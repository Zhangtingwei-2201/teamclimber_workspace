#include "shooter_search_armor.h"
#include "shape_tools.h"
#include "YOLOv11.h"

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <string>

Logger logger;

struct DetectedArmor
{
    std::string type;
    cv::Point2f TLcorner;
    double width;
    double height;
};

void shooter_node::callback_search_armor(sensor_msgs::msg::Image::SharedPtr msg)
{
    try
    {
        // 1. 图像转换
        cv_bridge::CvImagePtr cv_ptr;
        if (msg->encoding == "rgb8" || msg->encoding == "R8G8B8")
        {
            cv::Mat image(msg->height, msg->width, CV_8UC3,
                          const_cast<unsigned char *>(msg->data.data()));
            cv::Mat bgr_image;
            cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
            cv_ptr = std::make_shared<cv_bridge::CvImage>();
            cv_ptr->header = msg->header;
            cv_ptr->encoding = "bgr8";
            cv_ptr->image = bgr_image;
        }
        else
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }

        cv::Mat image = cv_ptr->image;
        if (image.empty())
            return;

        cv::Mat result_image = image.clone();

        // 2. 模型检测
        std::vector<Detection> armor_objects = model->detect(image);
        model->draw(image, result_image, armor_objects);

        // 3. 处理检测结果
        for (const auto &obj : armor_objects)
        {
            DetectedArmor armor_obj;

            int class_id = obj.class_id;
            std::string class_name = CLASS_NAMES[class_id];
            float confidence = obj.conf;
            cv::Rect bounding_box = obj.bbox;

            armor_obj.type = class_name;
            armor_obj.TLcorner.x = bounding_box.x;
            armor_obj.TLcorner.y = bounding_box.y;
            armor_obj.width = bounding_box.width;
            armor_obj.height = bounding_box.height;

            RCLCPP_INFO(this->get_logger(), "Found Armor:%s, Box=[%.2f, %.2f, %.2f, %.2f]",
                        class_name.c_str(), armor_obj.TLcorner.x, armor_obj.TLcorner.y, armor_obj.width, armor_obj.height);

            // 获取 2D 关键点
            std::vector<cv::Point2f> opencv_corners = shape_tools::calculateArmor2DCorners(
                armor_obj.TLcorner.x, armor_obj.TLcorner.y, armor_obj.width, armor_obj.height);

            double real_width = 0.705;
            double real_height = 0.52;
            double half_width = real_width / 2.0;
            double half_height = real_height / 2.0;

            std::vector<cv::Point3f> object_3d_points;
            object_3d_points.emplace_back(-half_width, -half_height, 0); // TL (左上)
            object_3d_points.emplace_back(half_width, -half_height, 0);  // TR (右上)
            object_3d_points.emplace_back(half_width, half_height, 0);   // BR (右下)
            object_3d_points.emplace_back(-half_width, half_height, 0);  // BL (左下)

            // 相机内参
            static const cv::Mat camera_matrix =
                (cv::Mat_<double>(3, 3) << 381.36, 0.0, 320.0,
                 0.0, 381.36, 240.0,
                 0.0, 0.0, 1.0);
            static const cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);

            cv::Mat rvec, tvec;
            
            // 2. 求解 PnP
            bool success = cv::solvePnP(
                object_3d_points,
                opencv_corners,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                false,
                cv::SOLVEPNP_IPPE
            );

            if (success)
            {
                tvec.convertTo(tvec, CV_64F);
                
                double x_c = tvec.at<double>(0); 
                double y_c = tvec.at<double>(1); 
                double z_c = tvec.at<double>(2);

                double gazebo_x = x_c;
                double gazebo_y = z_c;
                double gazebo_z = -y_c;

                RCLCPP_INFO(this->get_logger(), "[Gazebo]:[x=%.2f, y=%.2f, z=%.2f]",
                    gazebo_x, gazebo_y, gazebo_z);
                
                double bullet_speed = 15.0;
                double gravity_a = 9.8;

                double TanElevation = shape_tools::calculateLowTanElevation(
                    gazebo_x, gazebo_y, gazebo_z, bullet_speed, gravity_a);

                RCLCPP_INFO(this->get_logger(), "Calculated Elevation (v=%.1f): %.4f",
                            bullet_speed, TanElevation);
            }
        }

        cv::imshow("Detection Result", result_image);
        cv::waitKey(1);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
}
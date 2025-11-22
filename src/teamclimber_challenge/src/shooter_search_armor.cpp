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
    int width;
    int height;
};

void shooter_node::callback_search_armor(sensor_msgs::msg::Image::SharedPtr msg)
{
    try
    {
        // 图像转换：从ROS的Img到opencv的Mat
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
        {
            RCLCPP_WARN(this->get_logger(), "Received empty image");
            return;
        }

        // 创建结果图像
        cv::Mat result_image = image.clone();

        // 用模型识别
        std::vector<Detection> armor_objects = model->detect(image);

        // 绘制识别框
        model->draw(image, result_image, armor_objects);

        // 检测到的单个装甲板
        if (!armor_objects.empty())
        {
            DetectedArmor armor_obj;

            int class_id = obj.class_id;                    // 类别id
            std::string class_name = CLASS_NAMES[class_id]; // 类别名
            float confidence = obj.conf;                    // 置信度
            cv::Rect bounding_box = obj.bbox;               // 边界框

            armor_obj.type = class_name;
            armor_obj.TLcorner.x = bounding_box.x;
            armor_obj.TLcorner.y = bounding_box.y;
            armor_obj.width = bounding_box.width;
            armor_obj.height = bounding_box.height;

            RCLCPP_INFO(this->get_logger(), "Found Armor:%s, Confidence=%.2f, Box=[%d, %d, %d, %d]",
                        class_name.c_str(), confidence, armor_obj.TLcorner.x, armor_obj.TLcorner.y, armor_obj.width, armor_obj.height);

            // 计算装甲板的四个角点坐标（二维）
            std::vector<cv::Point2f> opencv_corners = shape_tools::calculateArmor2DCorners(
                armor_obj.TLcorner.x, armor_obj.TLcorner.y, armor_obj.width, armor_obj.height)

            // 定义仿真世界装甲板的尺寸
            float half_width = 0.705 / 2.0;
            float half_height = 0.230 / 2.0;

            // 定义相机内参
            static const cv::Mat camera_matrix =
                (cv::Mat_<double>(3, 3) << 381.36, 0.0, 320.0,
                 0.0, 381.36, 240.0,
                 0.0, 0.0, 1.0);

            // 定义四个点的 3D 坐标
            std::vector<cv::Point3f> armor_3Dcorners;
            armor_3Dcorners.push_back(cv::Point3f(-half_x, half_y, 0));  // 左下角点
            armor_3Dcorners.push_back(cv::Point3f(half_x, half_y, 0));   // 右下角点
            armor_3Dcorners.push_back(cv::Point3f(half_x, -half_y, 0));  // 右上角点
            armor_3Dcorners.push_back(cv::Point3f(-half_x, -half_y, 0)); // 左上角点

            cv::Mat rvec;                                                    // 旋转向量
            cv::Mat tvec;                                                    // 平移向量
            static const cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F); // 相机畸变系数

            // 调用PnP
            bool success = cv::solvePnP(
                armor_3Dcorners,
                opencv_corners,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                false,
                cv::SOLVEPNP_ITERATIVE);

            if (success)
            {
                // 获取旋转矩阵 (从旋转向量 rvec 转换而来)
                cv::Mat rot_matrix;
                cv::Rodrigues(rvec, rot_matrix);

                // 遍历 4 个角点，计算它们在 Gazebo 坐标系下的 3D 坐标
                std::vector<std::string> corner_names = {"左下", "右下", "右上", "左上"};
                for (size_t i = 0; i < opencv_corners.size(); ++i)
                {
                    // P_local 是装甲板自身的二维坐标
                    cv::Mat p_local = (cv::Mat_<double>(3, 1) << opencv_corners[i].x,
                                       opencv_corners[i].y,
                                       opencv_corners[i].z);

                    // 转换到相机坐标系: P_cam = R * P_local + T
                    cv::Mat p_cam = rot_matrix * p_local + tvec;

                    // 获取相机坐标系下的值
                    double cam_x = p_cam.at<double>(0);
                    double cam_y = p_cam.at<double>(1);
                    double cam_z = p_cam.at<double>(2);

                    // 转换到 Gazebo 坐标系
                    double gazebo_x = cam_z;
                    double gazebo_y = -cam_x;
                    double gazebo_z = -cam_y;

                    // 打印每个角点的 Gazebo 坐标
                    RCLCPP_INFO(this->get_logger(), "%s Corner: [x=%.2f, y=%.2f, z=%.2f]",
                                corner_names[i].c_str(), gazebo_x, gazebo_y, gazebo_z);
                }

                // 打印中心点坐标
                double center_gazebo_x = tvec.at<double>(2);
                double center_gazebo_y = -tvec.at<double>(0);
                double center_gazebo_z = -tvec.at<double>(1);
                RCLCPP_INFO(this->get_logger(), "Center Dist: %.2fm | Gazebo Pos: [%.2f, %.2f, %.2f]",
                            tvec.at<double>(2), center_gazebo_x, center_gazebo_y, center_gazebo_z);
            }
        }

        // 计算低弹道仰角
        double TanElevation = shape_tools::calculateLowTanElevation(center_gazebo_x, center_gazebo_y, center_gazebo_z, 1.5, 9.8);
    }
    // 显示结果图像
    cv::imshow("Detection Result", result_image);
    cv::waitKey(1);

    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
}

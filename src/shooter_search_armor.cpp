#include "shooter_search_armor.h"

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
            cv::Mat image(msg->height, msg->width, CV_8UC3, const_cast<unsigned char *>(msg->data.data()));
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
        if (image.empty())return;
        cv::Mat result_image = image.clone();

        // 2. YOLO模型识别检测
        std::vector<Detection> armor_objects = model->detect(image);
        model->draw(image, result_image, armor_objects);

        if (armor_objects.empty())
        {
            pred_x.reset();
            pred_y.reset();
            pred_z.reset();
            cv::imshow("Detection Result", result_image);
            cv::waitKey(1);
            return;
        }

        // 3. 处理检测结果
        const auto &obj = armor_objects[0];
        {
            DetectedArmor armor_obj;
            armor_obj.type = CLASS_NAMES[obj.class_id];
            armor_obj.TLcorner.x = obj.bbox.x;
            armor_obj.TLcorner.y = obj.bbox.y;
            armor_obj.width = obj.bbox.width;
            armor_obj.height = obj.bbox.height;

            // PnP 准备
            std::vector<cv::Point2f> opencv_corners = shape_tools::calculateArmor2DCorners(
                armor_obj.TLcorner.x, armor_obj.TLcorner.y, armor_obj.width, armor_obj.height);
            std::vector<cv::Point3f> object_3d_points = shape_tools::calculateArmor3DCorners(
                real_width / 2.0, real_height / 2.0);

            cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
            cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
            cv::Mat rvec, tvec;

            bool success = cv::solvePnP(object_3d_points, opencv_corners, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);

            if (success)
            {
                tvec.convertTo(tvec, CV_64F);
                double x_c = tvec.at<double>(0);
                double y_c = tvec.at<double>(1);
                double z_c = tvec.at<double>(2);

                // Cam -> Gazebo/World
                double raw_x = x_c;
                double raw_y = z_c;
                double raw_z = -y_c - 0.2; //0.2是z轴测量计算平均误差

                // 打印目标当前的仿真世界中的坐标
                RCLCPP_INFO(this->get_logger(), "Target:[%.2f,%.2f,%.2f]", raw_x, raw_y, raw_z);

                double current_time = this->now().seconds();

                // 1. 卡尔曼滤波更新 (只更新，不预测)
                // 这里的 0 表示 predict 0秒，也就是返回当前状态
                pred_x.update_and_predict(raw_x, current_time, 0);
                pred_y.update_and_predict(raw_y, current_time, 0);
                pred_z.update_and_predict(raw_z, current_time, 0);

                // 获取位置
                double curr_x = pred_x.get_current_pos();
                double curr_y = pred_y.get_current_pos();
                double curr_z = pred_z.get_current_pos();

                // 获取速度
                double v_x = pred_x.get_current_vel();
                double v_y = pred_y.get_current_vel();
                double v_z = pred_z.get_current_vel();

                // 获取加速度
                double a_x = pred_x.get_current_acc();
                double a_y = pred_y.get_current_acc();
                double a_z = pred_z.get_current_acc();

                // 2. 迭代求解
                double hit_x = curr_x;
                double hit_y = curr_y;
                double hit_z = curr_z;
                double t_flight = 0.2;
                double final_pitch = 0.0;
                double dist_horiz = 0.0;

                // 3次循环：粗算、修正、精修
                for (int i = 0; i < 3; i++)
                {
                    // 总时间 = 弹丸飞行时间 + 系统耗时（模型识别和卡尔曼滤波算法时间）
                    double t_total = t_flight + system_latency;

                    // 预测未来位置: P = P0 + v*t + 0.5*a*t^2
                    hit_x = curr_x + v_x * t_total + 0.5 * a_x * t_total * t_total;
                    hit_y = curr_y + v_y * t_total + 0.5 * a_y * t_total * t_total;
                    hit_z = curr_z + v_z * t_total + 0.5 * a_z * t_total * t_total;

                    // 计算目标在地面的投影到原点的距离
                    dist_horiz = std::sqrt(hit_x * hit_x + hit_y * hit_y);

                    BallisticSolution sol = ballistic_solver.solve(dist_horiz, hit_z, bullet_speed, gravity_a);

                    if (sol.has_solution)
                    {
                        t_flight = sol.time_of_flight;
                        final_pitch = sol.pitch;
                    }
                    else
                    {
                        break;
                    }
                }

                // 计算偏航角
                double final_yaw = std::atan2(hit_y, hit_x);

                RCLCPP_INFO(this->get_logger(), "Yaw:%.2f Pitch:%.2f Time:%.2f", final_yaw, final_pitch, t_flight);

                //================结果可视化（PnP反解）===================
                double aim_xc = hit_x;
                double aim_zc = hit_y;
                double aim_yc = -(hit_z + 0.2);

                // 绿色十字（预测击中点）
                cv::Point2f aim_point_green;
                if (aim_zc > 0.1)
                {
                    aim_point_green.x = fx * (aim_xc / aim_zc) + cx;
                    aim_point_green.y = fy * (aim_yc / aim_zc) + cy;
                }

                // 算出枪口指向的物理高度
                double aim_z_barrel = dist_horiz * std::tan(final_pitch);
                double aim_yc_blue = -(aim_z_barrel + 0.2);

                // 蓝色十字（枪口指向）
                cv::Point2f aim_point_blue;
                aim_point_blue.x = aim_point_green.x;
                if (aim_zc > 0.1)
                {
                    aim_point_blue.y = fy * (aim_yc_blue / aim_zc) + cy;
                }

                // 判断是否结果是否出框
                bool green_in = (aim_point_green.x >= 0 && aim_point_green.x < image.cols &&
                                 aim_point_green.y >= 0 && aim_point_green.y < image.rows);
                bool blue_in = (aim_point_blue.x >= 0 && aim_point_blue.x < image.cols &&
                                aim_point_blue.y >= 0 && aim_point_blue.y < image.rows);

                if (green_in)
                {
                    // 画绿色十字（预测击中点）
                    cv::drawMarker(result_image, aim_point_green, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 25, 2);
                    std::vector<cv::Point3f> target_p3d = {cv::Point3f(0, 0, 0)};
                    std::vector<cv::Point2f> target_p2d;
                    cv::projectPoints(target_p3d, rvec, tvec, camera_matrix, dist_coeffs, target_p2d);
                    if (!target_p2d.empty())
                    {
                        cv::circle(result_image, target_p2d[0], 4, cv::Scalar(0, 0, 255), -1);
                        cv::line(result_image, target_p2d[0], aim_point_green, cv::Scalar(0, 255, 255), 1);
                    }
                }

                if (blue_in)
                {
                    // 画蓝色十字（枪口指向）
                    cv::drawMarker(result_image, aim_point_blue, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 25, 2);
                }

                if (green_in && blue_in)
                {
                    // 画粉线（重力补偿量）
                    cv::line(result_image, aim_point_green, aim_point_blue, cv::Scalar(255, 0, 255), 2);
                }

                cv::imshow("Detection Result", result_image);
                cv::waitKey(1);
            }
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
}
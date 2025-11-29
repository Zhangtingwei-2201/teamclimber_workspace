#include "image_deal.h"
#include "shape_tools.h"

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

Logger logger;

struct DetectedObject
{
  std::string type;
  std::vector<cv::Point2f> points;
};

// 阶段回调函数
void vision_node::callback_stage(referee_pkg::msg::RaceStage::SharedPtr msg)
{
  this->latest_stage = msg->stage;
}

// 订阅摄像头回调函数
void vision_node::callback_camera(sensor_msgs::msg::Image::SharedPtr msg)
{
  if (latest_stage <= 4)
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

      std::vector<DetectedObject> all_detected_objects;

      std::vector<std::string> point_names = {"#1#", "#2#", "#3#", "#4#"};
      std::vector<cv::Scalar> point_colors = {
          cv::Scalar(255, 0, 0),   // 蓝色 - 1
          cv::Scalar(0, 255, 0),   // 绿色 - 2
          cv::Scalar(0, 255, 255), // 黄色 - 3
          cv::Scalar(255, 0, 255)  // 紫色 - 4
      };

      // 创建结果图像
      cv::Mat result_image = image.clone();

      // 转换到 HSV 空间
      cv::Mat hsv;
      cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

      // /*===========================red_ring===========================zp*/
      if (latest_stage == 1)
      {
        // 红色检测 - 使用稳定的范围
        cv::Mat mask1, mask2, red_mask;
        cv::inRange(hsv, sphere_red_low1, sphere_red_high1, mask1);
        cv::inRange(hsv, sphere_red_low2, sphere_red_high2, mask2);
        red_mask = mask1 | mask2;

        // 适度的形态学操作
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel);

        // 找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 统计球的数量
        int valid_spheres = 0;

        for (size_t i = 0; i < contours.size(); i++)
        {
          double area = cv::contourArea(contours[i]);
          if (area < 500)
            continue;

          // 计算最小外接圆
          cv::Point2f center;
          float radius = 0;
          cv::minEnclosingCircle(contours[i], center, radius);

          // 计算圆形度
          double perimeter = cv::arcLength(contours[i], true);
          double circularity = 4 * CV_PI * area / (perimeter * perimeter);

          if (circularity > 0.7 && radius > 15 && radius < 200)
          {
            valid_spheres++;

            // 求出四个点坐标
            std::vector<cv::Point2f> sphere_points =
                shape_tools::calculateStableSpherePoints(center, radius);

            RCLCPP_INFO(this->get_logger(), "Found sphere %d: (%.1f, %.1f) R=%.1f C=%.3f",
                        valid_spheres, center.x, center.y, radius, circularity);

            // 绘制检测到的球体
            cv::circle(result_image, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 2); // 绿色圆圈
            cv::circle(result_image, center, 3, cv::Scalar(0, 0, 255), -1);                       // 红色圆心

            // 绘制球体上的四个点
            for (int j = 0; j < 4; j++)
            {
              cv::circle(result_image, sphere_points[j], 6, point_colors[j], -1);
              cv::circle(result_image, sphere_points[j], 6, cv::Scalar(0, 0, 0), 2);

              // 标注序号
              std::string point_text = std::to_string(j + 1);
              cv::putText(
                  result_image, point_text,
                  cv::Point(sphere_points[j].x + 5, sphere_points[j].y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
              cv::putText(
                  result_image, point_text,
                  cv::Point(sphere_points[j].x + 5, sphere_points[j].y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

              RCLCPP_INFO(this->get_logger(), "Sphere %d, Point(%s): (%.1f, %.1f)",
                          valid_spheres, point_names[j].c_str(), sphere_points[j].x, sphere_points[j].y);
            }

            // 添加到发送列表
            DetectedObject sphere_obj;
            sphere_obj.type = "Ring_red";
            sphere_obj.points = sphere_points;
            all_detected_objects.push_back(sphere_obj);

            double small_radius = 0.68 * radius;

            if (circularity > 0.7 && radius > 15 && radius < 200)
            {
              valid_spheres++;

              // 求出四个点坐标
              std::vector<cv::Point2f> small_sphere_points =
                  shape_tools::calculateStableSpherePoints(center, small_radius);

              RCLCPP_INFO(this->get_logger(), "Found sphere %d: (%.1f, %.1f) R=%.1f",
                          valid_spheres, center.x, center.y, small_radius);

              // 绘制检测到的球体
              cv::circle(result_image, center, static_cast<int>(small_radius), cv::Scalar(255, 0, 0), 2); // 蓝色圆圈

              // 绘制球体上的四个点
              for (int j = 0; j < 4; j++)
              {
                cv::circle(result_image, small_sphere_points[j], 6, point_colors[j], -1);
                cv::circle(result_image, small_sphere_points[j], 6, cv::Scalar(0, 0, 0), 2);

                // 标注序号
                std::string point_text = std::to_string(j + 1);
                cv::putText(
                    result_image, point_text,
                    cv::Point(small_sphere_points[j].x + 5, small_sphere_points[j].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
                cv::putText(
                    result_image, point_text,
                    cv::Point(small_sphere_points[j].x + 5, small_sphere_points[j].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

                RCLCPP_INFO(this->get_logger(), "Sphere %d, Point(%s): (%.1f, %.1f)",
                            valid_spheres, point_names[j].c_str(), small_sphere_points[j].x, small_sphere_points[j].y);
              }

              // 添加到发送列表
              DetectedObject sphere_obj;
              sphere_obj.type = "Ring_red";
              sphere_obj.points = small_sphere_points;
              all_detected_objects.push_back(sphere_obj);
            }
          }
        }
      }

      // /*===========================Arrow===========================ztw*/
      if (latest_stage == 2)
      {
        // 红色检测 - 使用稳定的范围
        cv::Mat mask1, mask2, arrowred_mask;
        cv::inRange(hsv, sphere_red_low1, sphere_red_high1, mask1);
        cv::inRange(hsv, sphere_red_low2, sphere_red_high2, mask2);
        arrowred_mask = mask1 | mask2;

        // 找轮廓
        std::vector<std::vector<cv::Point>> arrowred_contours;
        cv::findContours(arrowred_mask, arrowred_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 只保留最像箭头的那个轮廓（面积大且细长）
        int best_idx = -1;
        double best_score = 0.0; // 这里用 ratio 做一个简单打分

        for (size_t i = 0; i < arrowred_contours.size(); i++)
        {
          double area = cv::contourArea(arrowred_contours[i]);
          if (area < 50.0) // 太小的噪声过滤掉，阈值按需要调
            continue;

          cv::RotatedRect rr = cv::minAreaRect(arrowred_contours[i]);
          float w = rr.size.width;
          float h = rr.size.height;
          if (w < 5 || h < 5) // 太细太小的也过滤
            continue;

          float long_side = std::max(w, h);
          float short_side = std::min(w, h);
          float ratio = long_side / short_side; // 长宽比

          if (ratio < 1.5f) // 箭头比较细长，长宽比至少大于 1.5，视情况调
            continue;

          double score = ratio; // 细长程度兼顾
          if (score > best_score)
          {
            best_score = score;
            best_idx = static_cast<int>(i);
          }
        }

        int valid_arrows = 0;

        if (best_idx >= 0)
        {
          valid_arrows++;

          if (best_idx >= 0)
          {
            valid_arrows++;

            const auto &cnt = arrowred_contours[best_idx];

            //============ 1. 用 PCA 求箭头主方向 ============//
            if (cnt.size() < 5)
              return; // 太少不做

            cv::Mat data_pts((int)cnt.size(), 2, CV_64F);
            for (int i = 0; i < (int)cnt.size(); ++i)
            {
              data_pts.at<double>(i, 0) = cnt[i].x;
              data_pts.at<double>(i, 1) = cnt[i].y;
            }

            cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

            // PCA 得到的中心点
            cv::Point2f center(
                (float)pca.mean.at<double>(0, 0),
                (float)pca.mean.at<double>(0, 1));

            // 第一特征向量：箭头主方向 (红线方向)
            cv::Point2f main_dir(
                (float)pca.eigenvectors.at<double>(0, 0),
                (float)pca.eigenvectors.at<double>(0, 1));
            // 单位化
            float len = std::sqrt(main_dir.x * main_dir.x + main_dir.y * main_dir.y);
            main_dir.x /= len;
            main_dir.y /= len;

            // 垂直方向（宽度方向，黄色线方向）
            cv::Point2f perp_dir(-main_dir.y, main_dir.x);

            //============ 2. 在主方向 / 垂直方向上做投影，算长度和宽度 ============//
            double min_t = 1e9, max_t = -1e9; // 沿主方向的投影范围
            double min_s = 1e9, max_s = -1e9; // 沿垂直方向的投影范围

            for (int i = 0; i < (int)cnt.size(); ++i)
            {
              cv::Point2f p = cnt[i];
              cv::Point2f rel = p - center;

              double t = rel.dot(main_dir); // 在箭头方向上的投影
              double s = rel.dot(perp_dir); // 在箭头宽度方向上的投影

              if (t < min_t)
                min_t = t;
              if (t > max_t)
                max_t = t;
              if (s < min_s)
                min_s = s;
              if (s > max_s)
                max_s = s;
            }

            // 箭头总长度 & 宽度（这时宽度就是“两侧分支端点之间的距离”）
            double half_len = 0.5 * (max_t - min_t);
            double half_w = 0.5 * (max_s - min_s);

            //============ 3. 按“主方向+垂直方向”构造旋转矩形四个角 ============//
            // 保留和之前一样的顺序：1左上 2右上 3右下 4左下（在局部坐标中定义）
            std::vector<cv::Point2f> arrow_points(4);
            arrow_points[0] = center + main_dir * (float)(-half_len) + perp_dir * (float)(-half_w); // TL
            arrow_points[1] = center + main_dir * (float)(+half_len) + perp_dir * (float)(-half_w); // TR
            arrow_points[2] = center + main_dir * (float)(+half_len) + perp_dir * (float)(+half_w); // BR
            arrow_points[3] = center + main_dir * (float)(-half_len) + perp_dir * (float)(+half_w); // BL

            //============ 4. 画框、画点、输出 ============//
            for (int j = 0; j < 4; j++)
            {
              cv::line(result_image,
                       arrow_points[j],
                       arrow_points[(j + 1) % 4],
                       cv::Scalar(0, 255, 0), // 绿色矩形，方便和原来区分
                       2);
            }

            for (int j = 0; j < 4; j++)
            {
              cv::circle(result_image, arrow_points[j], 5, point_colors[j], -1);
              cv::circle(result_image, arrow_points[j], 5, cv::Scalar(255, 255, 255), 2);

              std::string point_text = std::to_string(j + 1);
              cv::Point text_pos(arrow_points[j].x + 5, arrow_points[j].y - 5);

              cv::putText(result_image, point_text, text_pos,
                          cv::FONT_HERSHEY_SIMPLEX, 0.5,
                          cv::Scalar(255, 255, 255), 2);
              cv::putText(result_image, point_text, text_pos,
                          cv::FONT_HERSHEY_SIMPLEX, 0.5,
                          point_colors[j], 1);
            }

            double full_len = max_t - min_t;
            double full_w = max_s - min_s;
            double ratio = full_len / full_w;

            RCLCPP_INFO(this->get_logger(),
                        "Arrow %d: center=(%.1f,%.1f) len=%.1f width=%.1f ratio=%.2f",
                        valid_arrows, center.x, center.y, full_len, full_w, ratio);

            DetectedObject arrow_obj;
            arrow_obj.type = "arrow";
            arrow_obj.points = arrow_points;
            all_detected_objects.push_back(arrow_obj);
          }
        }
      }
      //==============================armor=================================//
      if (latest_stage == 3 || latest_stage == 4)
      {
        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();

        // 用模型识别
        std::vector<Detection> armor_objects = model->detect(image);

        // 记录结束时间
        auto end = std::chrono::high_resolution_clock::now();

        // 绘制识别框
        model->draw(image, result_image, armor_objects);

        RCLCPP_INFO(this->get_logger(), "Totally detected %zu armor objects.", armor_objects.size());

        // 遍历所有检测到的装甲板
        for (const auto &obj : armor_objects)
        {
          int class_id = obj.class_id;                    // 类别id
          std::string class_name = CLASS_NAMES[class_id]; // 类别名
          float confidence = obj.conf;                    // 置信度
          cv::Rect bounding_box = obj.bbox;               // 边界框

          float bound_tlx = bounding_box.x;
          float bound_tly = bounding_box.y;
          float width = bounding_box.width;
          float height = bounding_box.height;

          RCLCPP_INFO(this->get_logger(), "Found Armor:%s, Confidence=%.2f, Box=[%.2f, %.2f, %.2f, %.2f]",
                      class_name.c_str(), confidence, bound_tlx, bound_tly, width, height);

          // 求出四个点坐标
          std::vector<cv::Point2f> armor_points =
              shape_tools::calculateArmorPoints(bound_tlx, bound_tly, width, height);

          // 绘制四个点
          for (int j = 0; j < 4; j++)
          {
            cv::circle(result_image, armor_points[j], 6, point_colors[j], -1);
            cv::circle(result_image, armor_points[j], 6, cv::Scalar(0, 0, 0), 2);

            // 标注序号
            std::string point_text = std::to_string(j + 1);
            cv::putText(
                result_image, point_text, cv::Point(armor_points[j].x + 5, armor_points[j].y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            cv::putText(
                result_image, point_text, cv::Point(armor_points[j].x + 5, armor_points[j].y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 1);

            RCLCPP_INFO(this->get_logger(), "Armor:%s, Point(%s): (%.1f, %.1f)",
                        class_name.c_str(), point_names[j].c_str(),
                        armor_points[j].x, armor_points[j].y);
          }

          // 添加到发送列表
          DetectedObject armor_obj;
          armor_obj.type = class_name;
          armor_obj.points = armor_points;
          all_detected_objects.push_back(armor_obj);
        }

        // 测量用时
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        RCLCPP_INFO(this->get_logger(), "cost %2.4lf ms", tc);
      }
      /*=============================================================*/
      // 显示结果图像
      cv::imshow("Detection Result", result_image);
      cv::waitKey(1);

      // 创建并发布消息
      referee_pkg::msg::MultiObject msg_object;
      msg_object.header = msg->header;
      msg_object.num_objects = all_detected_objects.size();

      for (const auto &detected_obj : all_detected_objects)
      {
        referee_pkg::msg::Object obj_msg;

        // 放入目标类型
        obj_msg.target_type = detected_obj.type;

        // 放入目标四个点坐标
        for (const auto &point : detected_obj.points)
        {
          geometry_msgs::msg::Point corner;
          corner.x = point.x;
          corner.y = point.y;
          corner.z = 0.0;
          obj_msg.corners.push_back(corner);
        }

        // 放入单个目标信息
        msg_object.objects.push_back(obj_msg);
      }

      Target_pub->publish(msg_object);
      RCLCPP_INFO(this->get_logger(), "Published %lu total targets", all_detected_objects.size());
    }
    catch (const cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
  }
}
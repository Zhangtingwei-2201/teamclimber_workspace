#include "image_deal.h"
#include "shape_tools.h"
#include "YOLOv11.h"

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <string>

Logger logger;

struct DetectedObject
{
  std::string type;
  std::vector<cv::Point2f> points;
};

void vision_node::callback_camera(sensor_msgs::msg::Image::SharedPtr msg)
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

    /*===========================sphere===========================zp*/
    // 红色检测 - 使用稳定的范围
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), mask2);
    mask = mask1 | mask2;

    // 适度的形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

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

        // 显示半径信息
        std::string info_text = "R:" + std::to_string((int)radius);
        cv::putText(
            result_image, info_text, cv::Point(center.x - 15, center.y + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

        // 绘制球体上的四个点
        for (int j = 0; j < 4; j++)
        {
          cv::circle(result_image, sphere_points[j], 6, point_colors[j], -1);
          cv::circle(result_image, sphere_points[j], 6, cv::Scalar(0, 0, 0), 2);

          // 标注序号
          std::string point_text = std::to_string(j + 1);
          cv::putText(
              result_image, point_text,
              cv::Point(sphere_points[j].x + 10, sphere_points[j].y - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
          cv::putText(
              result_image, point_text,
              cv::Point(sphere_points[j].x + 10, sphere_points[j].y - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

          RCLCPP_INFO(this->get_logger(), "Sphere %d, Point(%s): (%.1f, %.1f)",
                      valid_spheres, point_names[j].c_str(), sphere_points[j].x, sphere_points[j].y);
        }

        // 添加到发送列表
        DetectedObject sphere_obj;
        sphere_obj.type = "sphere";
        sphere_obj.points = sphere_points;
        all_detected_objects.push_back(sphere_obj);
      }
    }

    /*===========================rect===========================zwt*/

    /*===========================armor===========================zp*/
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

      int bound_tlx = bounding_box.x;
      int bound_tly = bounding_box.y;
      int width = bounding_box.width;
      int height = bounding_box.height;

      RCLCPP_INFO(this->get_logger(), "Found Armor:%s, Confidence=%.2f, Box=[%d, %d, %d, %d]",
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
            result_image, point_text, cv::Point(armor_points[j].x + 10, armor_points[j].y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
        cv::putText(
            result_image, point_text, cv::Point(armor_points[j].x + 10, armor_points[j].y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

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

    /*=============================================================*/

    // 显示结果图像
    cv::imshow("Detection Result", result_image);
    cv::waitKey(1);

    // 创建并发布消息
    referee_pkg::msg::MultiObject msg_object;
    msg_object.header = msg->header;

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
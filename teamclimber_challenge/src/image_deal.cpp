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
    // 青色检测
    cv::Mat cyan_mask;
    cv::inRange(hsv, rect_cyan_low, rect_cyan_high, cyan_mask);

    // 形态学去噪（轻度）
    cv::Mat cyan_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(cyan_mask, cyan_mask, cv::MORPH_OPEN, cyan_kernel);
    cv::morphologyEx(cyan_mask, cyan_mask, cv::MORPH_CLOSE, cyan_kernel);

    // 找 cyan 的轮廓
    std::vector<std::vector<cv::Point>> cyan_contours;
    cv::findContours(cyan_mask, cyan_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 统计rect数量
    int valid_rects = 0;

    for (size_t i = 0; i < cyan_contours.size(); i++)
    {
      double area = cv::contourArea(cyan_contours[i]);
      if (area < rect_min_area)
        continue;

      double peri = cv::arcLength(cyan_contours[i], true);
      if (peri < 1e-3)
        continue;

      // 逼近四边形
      std::vector<cv::Point> poly;
      cv::approxPolyDP(cyan_contours[i], poly, approx_eps_ratio * peri, true);

      if (poly.size() > 7 && poly.size() < 3)
        continue;
      if (!cv::isContourConvex(poly))
        continue;

      // 用最小外接旋转矩形估计宽高/比例
      cv::RotatedRect rr = cv::minAreaRect(poly);
      float w = rr.size.width;
      float h = rr.size.height;
      if (w < 5 || h < 5)
        continue;

      float ratio = w > h ? w / h : h / w;
      if (ratio < rect_min_ratio || ratio > rect_max_ratio)
        continue;

      valid_rects++;

      // poly 转 Point2f
      std::vector<cv::Point2f> pts;
      for (auto &p : poly)
        pts.emplace_back(p.x, p.y);

      // 排序成：TL TR BR BL -> 输出顺序：1左下 2右下 3右上 4左上
      std::sort(pts.begin(), pts.end(),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.y < b.y; });

      std::vector<cv::Point2f> top{pts[0], pts[1]};
      std::vector<cv::Point2f> bottom{pts[2], pts[3]};

      std::sort(top.begin(), top.end(),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.x < b.x; });
      std::sort(bottom.begin(), bottom.end(),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.x < b.x; });

      cv::Point2f tl = top[0], tr = top[1];
      cv::Point2f bl = bottom[0], br = bottom[1];

      std::vector<cv::Point2f> rect_points = {bl, br, tr, tl};

      // 红色矩形边框
      for (int j = 0; j < 4; j++)
      {
        cv::line(result_image,
                 rect_points[j],
                 rect_points[(j + 1) % 4],
                 cv::Scalar(0, 0, 255), // BGR 红色
                 4);                    // 宽
      }

      for (int j = 0; j < 4; j++)
      {
        // 彩色实心点
        cv::circle(result_image, rect_points[j], 7, point_colors[j], -1);
        // 黑色描边
        cv::circle(result_image, rect_points[j], 7, cv::Scalar(0, 0, 0), 2);

        // 编号
        std::string point_text = std::to_string(j + 1);
        cv::Point text_pos(rect_points[j].x + 10, rect_points[j].y - 10);

        cv::putText(result_image, point_text, text_pos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 4); // 白色描边
        cv::putText(result_image, point_text, text_pos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    point_colors[j], 2); // 彩色字

        RCLCPP_INFO(this->get_logger(), "Rect %d, Point(%s): (%.1f, %.1f)",
                    valid_rects, point_names[j].c_str(), rect_points[j].x, rect_points[j].y);
      }

      RCLCPP_INFO(this->get_logger(), "Found rect %d: center=(%.1f,%.1f) w=%.1f h=%.1f ratio=%.2f",
                  valid_rects, rr.center.x, rr.center.y, w, h, ratio);

      DetectedObject rect_obj;
      rect_obj.type = "rect";
      rect_obj.points = rect_points;
      all_detected_objects.push_back(rect_obj);
    }

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
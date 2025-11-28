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

    // 形态学去噪
    cv::Mat cyan_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(cyan_mask, cyan_mask, cv::MORPH_OPEN, cyan_kernel);
    cv::morphologyEx(cyan_mask, cyan_mask, cv::MORPH_CLOSE, cyan_kernel);

    // 找cyan的轮廓
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

      // 逼近多边形
      std::vector<cv::Point> poly;
      cv::approxPolyDP(cyan_contours[i], poly, approx_eps_ratio * peri, true);

      if (poly.size() < 3 || poly.size() > 7)
        continue;
      if (!cv::isContourConvex(poly))
        continue;

      // 最小外接矩形
      cv::RotatedRect rr = cv::minAreaRect(poly);
      float w = rr.size.width;
      float h = rr.size.height;
      if (w < 5 || h < 5)
        continue;

      float ratio = (w > h ? w / h : h / w);
      if (ratio < rect_min_ratio || ratio > rect_max_ratio)
        continue;

      valid_rects++;

      // 至少要有4个点才能组成矩形
      if (poly.size() < 4)
        continue;

      // 把poly全部转成Point2f
      std::vector<cv::Point2f> all_pts;
      all_pts.reserve(poly.size());
      for (size_t i = 0; i < poly.size(); ++i)
      {
        cv::Point &p = poly[i];
        all_pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
      }

      // 在所有从n个点中选4个点的组合里，找误差最小的一组
      double best_cost = std::numeric_limits<double>::max();
      bool found_best = false;
      cv::Point2f best_tl, best_tr, best_bl, best_br;

      int n = static_cast<int>(all_pts.size());
      for (int a = 0; a <= n - 4; ++a)
      {
        for (int b = a + 1; b <= n - 3; ++b)
        {
          for (int c = b + 1; c <= n - 2; ++c)
          {
            for (int d = c + 1; d <= n - 1; ++d)
            {
              // 当前这一组4个点
              std::vector<cv::Point2f> cand = {
                  all_pts[a], all_pts[b], all_pts[c], all_pts[d]};

              // 对这4个点做一次“上/下 + 左/右”分类
              std::vector<cv::Point2f> tmp = cand;

              // 先按y从小到大排序：前2个是上面的，后2个是下面的
              std::sort(tmp.begin(), tmp.end(),
                        [](const cv::Point2f &p1, const cv::Point2f &p2)
                        { return p1.y < p2.y; });

              std::vector<cv::Point2f> top{tmp[0], tmp[1]};
              std::vector<cv::Point2f> bottom{tmp[2], tmp[3]};

              // 上下各自按x从小到大排序：从左到右
              std::sort(top.begin(), top.end(),
                        [](const cv::Point2f &p1, const cv::Point2f &p2)
                        { return p1.x < p2.x; });
              std::sort(bottom.begin(), bottom.end(),
                        [](const cv::Point2f &p1, const cv::Point2f &p2)
                        { return p1.x < p2.x; });

              cv::Point2f tl = top[0];    // 左上
              cv::Point2f tr = top[1];    // 右上
              cv::Point2f bl = bottom[0]; // 左下
              cv::Point2f br = bottom[1]; // 右下

              // 误差公式：
              //|左上x-左下x|+|右上x-右下x|+|左上y-右上y|+|左下y-右下y|
              double cost = std::fabs(tl.x - bl.x) + std::fabs(tr.x - br.x) +
                            std::fabs(tl.y - tr.y) + std::fabs(bl.y - br.y);

              if (cost < best_cost)
              {
                best_cost = cost;
                found_best = true;
                best_tl = tl;
                best_tr = tr;
                best_bl = bl;
                best_br = br;
              }
            }
          }
        }
      }

      std::vector<cv::Point2f> tmp4 = {best_bl, best_br, best_tr, best_tl};

      // 按y从小到大排序：tmp4[0]、tmp4[1] 是最高的两个点
      std::sort(tmp4.begin(), tmp4.end(),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.y < b.y; });

      // 最高的两个点
      cv::Point2f top0 = tmp4[0];
      cv::Point2f top1 = tmp4[1];

      // 上边y：取两者的较小值
      float top_y = std::min(top0.y, top1.y);

      // 第三高点的 y，当作矩形底边 y
      float third_y = tmp4[2].y;

      // 上下边长度只由最高两个点的 x 决定
      float left_x = std::min(top0.x, top1.x);
      float right_x = std::max(top0.x, top1.x);

      // 用这四个坐标构成“新矩形”的四个角
      cv::Point2f rect_tl(left_x, top_y);    // 左上
      cv::Point2f rect_tr(right_x, top_y);   // 右上
      cv::Point2f rect_bl(left_x, third_y);  // 左下
      cv::Point2f rect_br(right_x, third_y); // 右下

      // 定义 rect_points 顺序：1 左下，2 右下，3 右上，4 左上
      std::vector<cv::Point2f> rect_points = {
          rect_bl, rect_br, rect_tr, rect_tl};

      // 用这个矩形来画框
      cv::Rect box(cvRound(left_x),
                   cvRound(top_y),
                   cvRound(right_x - left_x),
                   cvRound(third_y - top_y));

      if (box.width <= 0 || box.height <= 0)
      {
        RCLCPP_WARN(this->get_logger(), "Invalid rect box (w<=0 or h<=0).");
        continue;
      }
      // 红色边框
      cv::rectangle(result_image, box, cv::Scalar(0, 0, 255), 4);
      for (int j = 0; j < 4; j++)
      {
        // 彩色实心点+黑边
        cv::circle(result_image, rect_points[j], 7, point_colors[j], -1);
        cv::circle(result_image, rect_points[j], 7, cv::Scalar(0, 0, 0), 2);

        // 编号
        std::string point_text = std::to_string(j + 1);
        cv::Point text_pos(rect_points[j].x + 10, rect_points[j].y - 10);

        cv::putText(result_image, point_text, text_pos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 4);
        cv::putText(result_image, point_text, text_pos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    point_colors[j], 2);

        RCLCPP_INFO(this->get_logger(),
                    "Rect %d, Point(%s): (%.1f, %.1f)",
                    valid_rects, point_names[j].c_str(),
                    rect_points[j].x, rect_points[j].y);
      }

      RCLCPP_INFO(this->get_logger(),
                  "Found rect %d: center=(%.1f,%.1f) w=%.1f h=%.1f ratio=%.2f",
                  valid_rects, rr.center.x, rr.center.y,
                  w, h, ratio);

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
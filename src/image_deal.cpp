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
        // 1. 红色检测 - 使用双阈值合并
        cv::Mat mask1, mask2, arrowred_mask;
        cv::inRange(hsv, sphere_red_low1, sphere_red_high1, mask1);
        cv::inRange(hsv, sphere_red_low2, sphere_red_high2, mask2);
        arrowred_mask = mask1 | mask2;

        // 2. 找轮廓
        std::vector<std::vector<cv::Point>> arrowred_contours;
        cv::findContours(arrowred_mask, arrowred_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int best_idx = -1;
        double best_score = 0.0;

        // 3. 筛选最佳轮廓 (面积 + 长宽比)
        for (size_t i = 0; i < arrowred_contours.size(); i++)
        {
          double area = cv::contourArea(arrowred_contours[i]);
          if (area < 100.0)
            continue; // 稍微调大一点阈值过滤噪声

          cv::RotatedRect rr = cv::minAreaRect(arrowred_contours[i]);
          float w = rr.size.width;
          float h = rr.size.height;
          if (w < 5 || h < 5)
            continue;

          float long_side = std::max(w, h);
          float short_side = std::min(w, h);
          float ratio = long_side / short_side;

          if (ratio < 1.5f)
            continue; // 箭头通常比较细长

          // 简单打分：面积 * 长宽比，优先选大且长的
          double score = area * ratio;
          if (score > best_score)
          {
            best_score = score;
            best_idx = static_cast<int>(i);
          }
        }

        if (best_idx >= 0)
        {
          const auto &cnt = arrowred_contours[best_idx];

          // ============ A. PCA 计算主轴 ============ //
          int sz = static_cast<int>(cnt.size());
          cv::Mat data_pts(sz, 2, CV_64F);
          for (int i = 0; i < sz; ++i)
          {
            data_pts.at<double>(i, 0) = cnt[i].x;
            data_pts.at<double>(i, 1) = cnt[i].y;
          }

          cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

          // 中心点
          cv::Point2f center((float)pca.mean.at<double>(0, 0), (float)pca.mean.at<double>(0, 1));

          // 初始主方向 (特征值最大的特征向量)
          cv::Point2f vec_main((float)pca.eigenvectors.at<double>(0, 0), (float)pca.eigenvectors.at<double>(0, 1));
          // 归一化
          float norm = std::sqrt(vec_main.x * vec_main.x + vec_main.y * vec_main.y);
          vec_main.x /= norm;
          vec_main.y /= norm;

          // 初始垂直方向 (顺时针旋转90度还是逆时针无所谓，后面会强制校正)
          cv::Point2f vec_perp(-vec_main.y, vec_main.x);

          // ============ B. 投影点以确定长度范围和方向 ============ //
          std::vector<double> proj_t(sz), proj_s(sz);
          double min_t = 1e9, max_t = -1e9;
          double min_s = 1e9, max_s = -1e9;

          for (int i = 0; i < sz; ++i)
          {
            cv::Point2f rel = cv::Point2f((float)cnt[i].x, (float)cnt[i].y) - center;
            double t = rel.dot(vec_main); // 沿主轴投影
            double s = rel.dot(vec_perp); // 沿垂轴投影

            proj_t[i] = t;
            proj_s[i] = s;

            if (t < min_t)
              min_t = t;
            if (t > max_t)
              max_t = t;
            if (s < min_s)
              min_s = s;
            if (s > max_s)
              max_s = s;
          }

          double length = max_t - min_t;
          double width = max_s - min_s;

          // ============ C. 关键步骤：判断箭头指向 ============ //
          // 逻辑：检查主轴两端（min_t端 和 max_t端）的点的分布宽度
          // 假设：箭头尖端（Tip）比尾部（Tail）更窄 (Spread更小)

          double threshold_range = length * 0.15; // 取两端 15% 的长度区域分析

          double min_end_spread_min = 1e9, min_end_spread_max = -1e9;
          double max_end_spread_min = 1e9, max_end_spread_max = -1e9;
          bool has_min_pts = false, has_max_pts = false;

          for (int i = 0; i < sz; ++i)
          {
            // 检查靠近 min_t 的区域
            if (proj_t[i] < min_t + threshold_range)
            {
              if (proj_s[i] < min_end_spread_min)
                min_end_spread_min = proj_s[i];
              if (proj_s[i] > min_end_spread_max)
                min_end_spread_max = proj_s[i];
              has_min_pts = true;
            }
            // 检查靠近 max_t 的区域
            if (proj_t[i] > max_t - threshold_range)
            {
              if (proj_s[i] < max_end_spread_min)
                max_end_spread_min = proj_s[i];
              if (proj_s[i] > max_end_spread_max)
                max_end_spread_max = proj_s[i];
              has_max_pts = true;
            }
          }

          // 计算两端的宽度
          double width_at_min = (has_min_pts) ? (min_end_spread_max - min_end_spread_min) : width;
          double width_at_max = (has_max_pts) ? (max_end_spread_max - max_end_spread_min) : width;

          // 真正的箭头方向向量（指向尖端）
          cv::Point2f head_dir = vec_main;

          // 如果 min 端比 max 端窄，说明 min 端是尖端，由于 vec_main 指向 max，所以需要反转
          // 如果 max 端比 min 端窄，说明 max 端是尖端，vec_main 方向正确
          if (width_at_min < width_at_max)
          {
            // min 端是尖头，当前 head_dir 指向 max (正t方向)，所以要反向
            head_dir = -vec_main;
          }
          // else: max 端是尖头，head_dir 保持不变

          // ============ D. 构建四个角点 (Left/Right definition) ============ //
          // 我们需要一个 "左" 向量 (Left Vector)，相对于箭头前进方向的左边
          // 在图像坐标系(y向下)中，如果 V=(x,y), 左向量 L=(y, -x)
          // 验证: V=(1,0)[右] -> L=(0,-1)[上] (在图像里上是y减小，正确)
          cv::Point2f left_dir(head_dir.y, -head_dir.x);

          // 重新计算中心位置（使用 minAreaRect 的几何中心思想，而不是质心）
          // 沿着最终的 head_dir，头部位置在 length/2，尾部在 -length/2
          // 注意：这里我们基于 PCA 的 center 平移

          // 为了更贴合边缘，我们基于 min_t/max_t 重新定位几何中心
          // 投影范围在 head_dir 上的值：
          // 如果 head_dir 未反转，范围是 [min_t, max_t] -> 几何中心偏移 (min_t + max_t)/2
          // 如果 head_dir 反转了，范围是 [-max_t, -min_t] -> 几何中心偏移 -(min_t + max_t)/2
          // 简便方法：直接用 head_dir 投影重新算一次极值，或者利用之前的 min_t/max_t

          // 让我们用最稳妥的方法：基于 head_dir 重新投影一次求几何中心，保证严谨
          double t_new_min = 1e9, t_new_max = -1e9;
          double w_new_min = 1e9, w_new_max = -1e9; // 宽度方向投影
          for (const auto &pt : cnt)
          {
            cv::Point2f rel = cv::Point2f((float)pt.x, (float)pt.y) - center;
            double t = rel.dot(head_dir);
            double w = rel.dot(left_dir);
            if (t < t_new_min)
              t_new_min = t;
            if (t > t_new_max)
              t_new_max = t;
            if (w < w_new_min)
              w_new_min = w;
            if (w > w_new_max)
              w_new_max = w;
          }

          double final_len = t_new_max - t_new_min;
          double final_wid = w_new_max - w_new_min;

          // 矩形的几何中心
          cv::Point2f geo_center = center + head_dir * (float)((t_new_max + t_new_min) / 2.0) + left_dir * (float)((w_new_max + w_new_min) / 2.0);

          // 计算半长和半宽
          float hl = (float)final_len / 2.0f;
          float hw = (float)final_wid / 2.0f;

          // 定义四个角点
          // 1. 正方向左 (Head-Left):  Center + HL*HeadDir + HW*LeftDir
          // 2. 正方向右 (Head-Right): Center + HL*HeadDir - HW*LeftDir
          // 3. 反方向右 (Tail-Right): Center - HL*HeadDir - HW*LeftDir
          // 4. 反方向左 (Tail-Left):  Center - HL*HeadDir + HW*LeftDir

          std::vector<cv::Point2f> arrow_points(4);
          arrow_points[0] = geo_center + head_dir * hl + left_dir * hw; // Point 1
          arrow_points[1] = geo_center + head_dir * hl - left_dir * hw; // Point 2
          arrow_points[2] = geo_center - head_dir * hl - left_dir * hw; // Point 3
          arrow_points[3] = geo_center - head_dir * hl + left_dir * hw; // Point 4

          // 交换1,3点位置,交换2,4点位置
          std::swap(arrow_points[0], arrow_points[2]);
          std::swap(arrow_points[1], arrow_points[3]);

          // ============ E. 可视化绘制 ============ //
          // 画矩形框
          for (int j = 0; j < 4; j++)
          {
            cv::line(result_image, arrow_points[j], arrow_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
          }

          // 画中心和主轴方向
          cv::circle(result_image, geo_center, 3, cv::Scalar(0, 255, 255), -1);
          cv::line(result_image, geo_center, geo_center + head_dir * (float)(final_len / 2), cv::Scalar(255, 0, 0), 2); // 蓝色线指向头

          // 标出 1, 2, 3, 4
          for (int j = 0; j < 4; j++)
          {
            // 这里的 point_colors 是假设你外部定义好的颜色数组，如果没有请自己定义
            // 如: std::vector<cv::Scalar> point_colors = {cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(255,0,0), cv::Scalar(0,255,255)};
            cv::Scalar color = cv::Scalar(0, 255, 0); // 默认绿色

            cv::circle(result_image, arrow_points[j], 5, color, -1);
            cv::putText(result_image, std::to_string(j + 1),
                        cv::Point(arrow_points[j].x + 5, arrow_points[j].y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
          }

          // 数据输出
          DetectedObject arrow_obj;
          arrow_obj.type = "arrow";
          arrow_obj.points = arrow_points;
          all_detected_objects.push_back(arrow_obj);

          RCLCPP_INFO(this->get_logger(), "Arrow Found: Center(%.1f, %.1f) Len:%.1f Wid:%.1f",
                      geo_center.x, geo_center.y, final_len, final_wid);
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
#ifndef IMAGE_DEAL_H
#define IMAGE_DEAL_H

#include <string>
#include <memory>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>

#include "YOLOv11.h"

extern Logger logger;

class vision_node : public rclcpp::Node
{
public:
  // 构造函数
  vision_node(std::string name) : Node(name)
  {
    RCLCPP_INFO(this->get_logger(), "%s节点已经启动.", name.c_str());

    // 初始化YOLO模型，使用智能指针
    const std::string engine_file_path = "/home/zoupeng/teamclimber_workspace/src/teamclimber_challenge/models/armorYOLO.engine";
    try
    {
      model = std::make_unique<YOLOv11>(engine_file_path, logger);
      RCLCPP_INFO(this->get_logger(), "YOLO模型加载成功");
    }
    catch (const std::exception &e)
    {
      RCLCPP_FATAL(this->get_logger(), "YOLO模型加载失败: %s", e.what());
      rclcpp::shutdown();
      return;
    }

    Image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        bind(&vision_node::callback_camera, this, std::placeholders::_1));

    Target_pub = this->create_publisher<referee_pkg::msg::MultiObject>(
        "/vision/target", 10);

    RCLCPP_INFO(this->get_logger(), "vision_node 启动成功");
  }

  // 析构函数
  ~vision_node() { cv::destroyWindow("Detection Result"); }

private:
  void callback_camera(sensor_msgs::msg::Image::SharedPtr msg);

  // 模型创建
  std::unique_ptr<YOLOv11> model;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
  rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr Target_pub;

  // sphere参数
  cv::Scalar sphere_red_low1 {0, 120, 70};
  cv::Scalar sphere_red_high1{10, 255, 255};
  cv::Scalar sphere_red_low2 {170, 120, 70};
  cv::Scalar sphere_red_high2{180, 255, 255};

  // rect参数
  // cyan的HSV 范围 //H在OpenCV是0~180，所以80~105是偏青色一段
  cv::Scalar rect_cyan_low{70, 60, 60};
  cv::Scalar rect_cyan_high{120, 255, 255};

  double rect_min_area = 100.0;   // 矩形最小面积
  double rect_max_ratio = 8.0;    // 最大长宽比（太细长不要），ratio=长边/短边
  double rect_min_ratio = 1.1;    // 最小长宽比（避免接近正方形时误判）
  double approx_eps_ratio = 0.02; // approxPolyDP系数
};

#endif
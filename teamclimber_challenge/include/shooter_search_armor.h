#ifndef SHOOTER_SEARCH_ARMOR_H
#define SHOOTER_SEARCH_ARMOR_H

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

#include "shape_tools.h"
#include "YOLOv11.h"

extern Logger logger;

class shooter_node : public rclcpp::Node
{
public:
    // 构造函数,有一个参数为节点名称
    shooter_node(std::string name) : Node(name)
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
            "/camera/image_raw", 10, bind(&shooter_node::callback_search_armor, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "shooter_node 启动成功");
    }

    // 析构函数
    ~shooter_node() { cv::destroyWindow("Detection Result"); }

private:
    void callback_search_armor(sensor_msgs::msg::Image::SharedPtr msg);

    // 模型创建
    std::unique_ptr<YOLOv11> model;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
};

#endif
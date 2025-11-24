#ifndef SHAPE_TOOLS_H
#define SHAPE_TOOLS_H

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

class shape_tools
{
public:
    // 稳定的球体点计算方法
    static std::vector<cv::Point2f> calculateStableSpherePoints(const cv::Point2f &center, float radius);

    // 求出矩形四个顶点
    static std::vector<cv::Point2f> calculateRectanglePoints(const cv::Point2f &center,float width,float height);

    // 求出装甲板的四个灯条坐标
    static std::vector<cv::Point2f> calculateArmorPoints(float bound_tlx, float bound_tly, float width, float height);

    // 求出装甲板的四个角点坐标
    static std::vector<cv::Point2f> calculateArmor2DCorners(float bound_tlx, float bound_tly, float width, float height);

    // 计算低弹道仰角
    static double calculateLowTanElevation(double x, double y, double z, double v0, double g);
};

#endif
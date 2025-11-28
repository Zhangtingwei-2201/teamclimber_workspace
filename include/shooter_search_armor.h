#ifndef SHOOTER_SEARCH_ARMOR_H
#define SHOOTER_SEARCH_ARMOR_H

#include <string>
#include <memory>
#include <cmath>
#include <deque>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include "shape_tools.h"
#include "YOLOv11.h"

// ===============简易卡尔曼滤波预测器（OpenCV）================
class Kalman1D
{
public:
    // 使用OpenCV封装好的简易卡尔曼滤波预测器
    cv::KalmanFilter kf;
    double last_time = 0.0;
    bool is_initialized = false;

    // 滤波器构造函数
    Kalman1D()
    {
        // 状态维数【位置, 速度, 加速度】、测量维数【位置】、控制维数
        kf.init(3, 1, 0);

        // 转移矩阵 F
        kf.transitionMatrix = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

        // 测量矩阵 H
        kf.measurementMatrix = cv::Mat::zeros(1, 3, CV_32F);
        kf.measurementMatrix.at<float>(0) = 1.0f;

        // 过程噪声 Q (根据测试调参) 
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        kf.processNoiseCov.at<float>(2, 2) = 1e-2;

        // 测量噪声 R（根据测试调参）
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));

        // 初始误差 P
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    }

    // 滤波器重置
    void reset()
    {
        is_initialized = false;
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    }

    // 更新和预测
    double update_and_predict(double input, double current_time, double t_ahead)
    {
        if (!is_initialized)
        {
            kf.statePost.at<float>(0) = (float)input; // 获取最优位置（唯一能从识别中获取信息）
            kf.statePost.at<float>(1) = 0;            // 获取最优速度
            kf.statePost.at<float>(2) = 0;            // 获取最优加速度
            last_time = current_time;
            is_initialized = true;
            return input;
        }

        // 计算时间差
        double dt = current_time - last_time;

        // 异常时间处理
        if (dt <= 0.00001 || dt > 1.0)
        {
            // 不更新卡尔曼滤波器，而是直接利用现有的、最后一次已知的状态来做一个最保守的推测
            return kf.statePost.at<float>(0) + kf.statePost.at<float>(1) * t_ahead;
        }

        // 1. 预测
        kf.transitionMatrix.at<float>(0, 1) = (float)dt;                // 速度对位置的影响
        kf.transitionMatrix.at<float>(0, 2) = 0.5f * (float)dt * dt;    // 加速度对位置的影响
        kf.transitionMatrix.at<float>(1, 2) = (float)dt;                // 加速度对速度的影响

        kf.predict();

        // 2. 校正
        cv::Mat measurement = cv::Mat(1, 1, CV_32F);
        measurement.at<float>(0) = (float)input; //把YOLO和PnP算出来的真实坐标填入这个矩阵
        kf.correct(measurement);

        last_time = current_time;

        // 3. 计算未来（使用当前最优状态）
        return get_current_pos() + get_current_vel() * t_ahead + 0.5 * get_current_acc() * t_ahead * t_ahead;
    }

    // 获取当前位置
    double get_current_pos() { return is_initialized ? kf.statePost.at<float>(0) : 0.0; }
    // 获取当前速度
    double get_current_vel() { return is_initialized ? kf.statePost.at<float>(1) : 0.0; }
    // 获取当前加速度
    double get_current_acc() { return is_initialized ? kf.statePost.at<float>(2) : 0.0; }
};

// ================弹道解算器==================
struct BallisticSolution
{
    double pitch;           // 仰角
    double time_of_flight;  // 弹丸飞行时间
    bool has_solution;      // 是否有解
};

class BallisticSolver
{
public:
    BallisticSolution solve(double dist_horiz, double height, double v, double g)
    {
        BallisticSolution sol;
        sol.has_solution = false;

        // 一元二次方程系数
        double A = (g * dist_horiz * dist_horiz) / (2.0 * v * v);
        double B = -dist_horiz;
        double C = height + A;

        double delta = B * B - 4 * A * C;

        // 无解判断
        if (delta < 0)return sol;

        double tan_theta = (-B - std::sqrt(delta)) / (2 * A);
        sol.pitch = std::atan(tan_theta);
        sol.time_of_flight = dist_horiz / (v * std::cos(sol.pitch));
        sol.has_solution = true;

        return sol;
    }
};

// ===================击打节点定义=======================
class shooter_node : public rclcpp::Node
{
public:
    shooter_node(std::string name) : Node(name)
    {
        RCLCPP_INFO(this->get_logger(), "%s节点已经启动.", name.c_str());

        const std::string engine_file_path = "/home/zoupeng/teamclimber_workspace/src/teamclimber_challenge/models/armorYOLO.engine";
        try
        {
            model = std::make_unique<YOLOv11>(engine_file_path, logger_);
            RCLCPP_INFO(this->get_logger(), "YOLO模型加载成功");
        }
        catch (const std::exception &e)
        {
            RCLCPP_FATAL(this->get_logger(), "YOLO模型加载失败: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        Image_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, std::bind(&shooter_node::callback_search_armor, this, std::placeholders::_1));
    }

    ~shooter_node() { cv::destroyWindow("Detection Result"); }

private:
    void callback_search_armor(sensor_msgs::msg::Image::SharedPtr msg);

    std::unique_ptr<YOLOv11> model;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
    Logger logger_;

    // 定义三个轴的卡尔曼预测器 (X, Y, Z 独立预测)
    Kalman1D pred_x;
    Kalman1D pred_y;
    Kalman1D pred_z;
    BallisticSolver ballistic_solver;

    // 参数管理
    double fx = 554.383, fy = 554.383, cx = 320.0, cy = 320.0; //相机内参
    double real_width = 0.705;      // 装甲板真实宽度 (米)
    double real_height = 0.520;     // 装甲板真实高度 (米)
    double bullet_speed = 10.0;    // 子弹初速度 (米/秒)
    double gravity_a = 9.8;         // 重力加速度
    double system_latency = 0.02;    // 系统总延迟 (图像处理+通信耗时)
};

#endif
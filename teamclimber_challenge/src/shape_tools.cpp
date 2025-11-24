#include "shape_tools.h"

std::vector<cv::Point2f> shape_tools::calculateStableSpherePoints(const cv::Point2f &center, float radius)
{
  std::vector<cv::Point2f> points;

  // 简单稳定的几何计算，避免漂移
  // 左、下、右、上（逆时针）
  points.push_back(cv::Point2f(center.x - radius, center.y)); // 左点 (1)
  points.push_back(cv::Point2f(center.x, center.y + radius)); // 下点 (2)
  points.push_back(cv::Point2f(center.x + radius, center.y)); // 右点 (3)
  points.push_back(cv::Point2f(center.x, center.y - radius)); // 上点 (4)

  return points;
}

// 矩形：四个顶点
// 左下，右下，右上，左上（逆时针）
std::vector<cv::Point2f> shape_tools::calculateRectanglePoints(
    const cv::Point2f &center,
    float width,
    float height)
{
  std::vector<cv::Point2f> points;

  float halfw = width * 0.5f;  // 一半的宽
  float halfh = height * 0.5f; // 一半的高

  // 图像坐标系：y 向下为正
  points.push_back(cv::Point2f(center.x - halfw, center.y + halfh)); // 左下
  points.push_back(cv::Point2f(center.x + halfw, center.y + halfh)); // 右下
  points.push_back(cv::Point2f(center.x + halfw, center.y - halfh)); // 右上
  points.push_back(cv::Point2f(center.x - halfw, center.y - halfh)); // 左上

  return points;
}

std::vector<cv::Point2f> shape_tools::calculateArmorPoints(float bound_tlx, float bound_tly, float width, float height)
{
  std::vector<cv::Point2f> points;

  // 左下、右下、右上、左上（逆时针）
  points.push_back(cv::Point2f(bound_tlx, bound_tly + 0.7 * height));           // 左下点 (1)
  points.push_back(cv::Point2f(bound_tlx + width, bound_tly + 0.7 * height));   // 右下点 (2)
  points.push_back(cv::Point2f(bound_tlx + width, bound_tly + 0.258 * height)); // 右上点 (3)
  points.push_back(cv::Point2f(bound_tlx, bound_tly + 0.258 * height));         // 左上点 (4)

  return points;
}

std::vector<cv::Point2f> shape_tools::calculateArmor2DCorners(float bound_tlx, float bound_tly, float width, float height)
{
  std::vector<cv::Point2f> points;

  // 左上、右上、右下、左下（顺时针）
  points.push_back(cv::Point2f(bound_tlx, bound_tly));                  // 左上角点 (4)
  points.push_back(cv::Point2f(bound_tlx + width, bound_tly));          // 右上角点 (3)
  points.push_back(cv::Point2f(bound_tlx + width, bound_tly + height)); // 右下角点 (2)
  points.push_back(cv::Point2f(bound_tlx, bound_tly + height));         // 左下角点 (1)

  return points;
}

double shape_tools::calculateLowTanElevation(double x, double y, double z, double v0, double g)
{
  double r = std::sqrt(x * x + y * y);
  double discriminant = (v0 * v0 * v0 * v0 - 2 * g * z * v0 * v0 - g * g * r * r);
  double tan_theta = (v0 * v0 - std::sqrt(discriminant)) / (g * r);

  return tan_theta;
}
#pragma once
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <Eigen/Dense>

// Only declaration, NO function body
sensor_msgs::msg::PointCloud2 eigenToPointCloud2(
    const Eigen::MatrixXd& points,
    const std_msgs::msg::Header& header);

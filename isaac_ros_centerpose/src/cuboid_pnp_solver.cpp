// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
#include "isaac_ros_centerpose/cuboid_pnp_solver.hpp"

#include <cmath>
#include <vector>

#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{
namespace
{

Eigen::Quaternionf ConvertRvecToQuaternion(cv::Mat rvec_cv)
{
  Eigen::Vector3f rvec = Eigen::Vector3f{
    static_cast<float>(rvec_cv.at<double>(0, 0)), static_cast<float>(rvec_cv.at<double>(1, 0)),
    static_cast<float>(rvec_cv.at<double>(2, 0))};
  const float theta{rvec.norm()};
  if (theta < std::numeric_limits<float>::epsilon()) {
    return Eigen::Quaternionf::Identity();
  }
  Eigen::Vector3f raxis = rvec / theta;
  return Eigen::Quaternionf{Eigen::AngleAxisf(theta, raxis)};
}

constexpr float kInvalidPoint{-5000};
constexpr int32_t kXIndex{0};
constexpr int32_t kYIndex{1};
constexpr int32_t kZIndex{2};

}  // namespace

std::optional<PnPResult> CuboidPnPSolver::solvePnP(
  const Eigen::MatrixXfRM & cuboid2d_points, const int pnp_algorithm)
{
  // Most of this code is just converting from Eigen to opencv
  const std::array<
    Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)> &
  cuboid3d_points = cuboid3d_.vertices();
  std::vector<cv::Point2f> obj_2d_points;
  std::vector<cv::Point3f> obj_3d_points;
  // Filter out any invalid readings
  for (size_t i = 0; i < static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT); ++i) {
    Eigen::Vector2f point_2d = cuboid2d_points.block<1, 2>(i, 0);
    if (point_2d.x() < kInvalidPoint || point_2d.y() < kInvalidPoint) {
      continue;
    }
    obj_2d_points.push_back(cv::Point2f{point_2d.x(), point_2d.y()});
    obj_3d_points.push_back(
        cv::Point3f{cuboid3d_points[i].x(), cuboid3d_points[i].y(), cuboid3d_points[i].z()});
  }

  if (obj_2d_points.size() < min_required_points_) {
    return std::nullopt;
  }

  // Convert from Eigen -> cv
  cv::Mat camera_matrix;
  cv::eigen2cv(camera_matrix_, camera_matrix);

  cv::Mat dist_coeffs;
  cv::eigen2cv(dist_coeffs_, dist_coeffs);

  cv::Mat rvec, tvec;
  bool ret = cv::solvePnP(
      obj_3d_points, obj_2d_points, camera_matrix, dist_coeffs, rvec, tvec, false, pnp_algorithm);
  if (!ret) {
    return std::nullopt;
  }

  std::vector<cv::Point2f> reprojected_points;
  cv::projectPoints(obj_3d_points, rvec, tvec, camera_matrix, dist_coeffs, reprojected_points);

  // Convert back to Eigen types
  PnPResult result;
  result.pose.position.x() = static_cast<float>(tvec.at<double>(kXIndex, 0));
  result.pose.position.y() = static_cast<float>(tvec.at<double>(kYIndex, 0));
  result.pose.position.z() = static_cast<float>(tvec.at<double>(kZIndex, 0));
  result.pose.orientation = ConvertRvecToQuaternion(rvec);

  result.projected_points = Eigen::MatrixXfRM(reprojected_points.size(), 2);
  for (size_t i = 0; i < reprojected_points.size(); ++i) {
    result.projected_points(i, kXIndex) = static_cast<float>(reprojected_points[i].x);
    result.projected_points(i, kYIndex) = static_cast<float>(reprojected_points[i].y);
  }

  return result;
}

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

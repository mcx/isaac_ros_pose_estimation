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
#pragma once

#include <cstddef>

#include "Eigen/Dense"

namespace nvidia
{
namespace isaac_ros
{
namespace centerpose
{

enum class CuboidVertexType : size_t
{
  REAR_BOTTOM_LEFT = 0,
  REAR_BOTTOM_RIGHT = 1,
  REAR_TOP_LEFT = 2,
  REAR_TOP_RIGHT = 3,
  FRONT_BOTTOM_LEFT = 4,
  FRONT_BOTTOM_RIGHT = 5,
  FRONT_TOP_LEFT = 6,
  FRONT_TOP_RIGHT = 7,
  CENTER = 8,
  TOTAL_CORNER_VERTEX_COUNT = 8,  // Corner vertices don't include the center
                                  // point
  TOTAL_VERTEX_COUNT = 9
};

class Cuboid3d {
public:
  Cuboid3d();

  explicit Cuboid3d(const Eigen::Vector3f & size_3d);

  const std::array<
    Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)> &
  vertices() const;

  std::array<Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)>
  vertices();

private:
  Eigen::Vector3f center_location_;
  Eigen::Vector3f size_3d_;

  std::array<Eigen::Vector3f, static_cast<size_t>(CuboidVertexType::TOTAL_CORNER_VERTEX_COUNT)>
  vertices_;
};

}  // namespace centerpose
}  // namespace isaac_ros
}  // namespace nvidia

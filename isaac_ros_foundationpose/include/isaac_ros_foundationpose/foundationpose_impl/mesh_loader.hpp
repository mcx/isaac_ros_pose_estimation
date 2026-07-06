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

#ifndef ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__MESH_LOADER_HPP_
#define ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__MESH_LOADER_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>

#include "Eigen/Dense"

struct aiMesh;

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

struct MeshData
{
  std::string texture_path;
  std::string mesh_file_path;

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mesh_vertices;
  float mesh_diameter{0.0f};

  Eigen::Vector3f mesh_model_center{0.0f, 0.0f, 0.0f};
  Eigen::Vector3f min_vertex{0.0f, 0.0f, 0.0f};
  Eigen::Vector3f max_vertex{0.0f, 0.0f, 0.0f};

  float * mesh_vertices_device{nullptr};
  float * mesh_normals_device{nullptr};
  int32_t * mesh_faces_device{nullptr};
  float * texcoords_device{nullptr};
  uint8_t * texture_map_device{nullptr};

  int num_vertices{0};
  int num_texcoords{0};
  int num_faces{0};

  bool has_tex{true};
  int texture_map_height{0};
  int texture_map_width{0};
  int texture_map_channels{0};

  void freeMeshDeviceMemory();
  void freeTextureDeviceMemory();
  ~MeshData();

  // Owns raw GPU pointers freed in ~MeshData(); copying would double-free.
  // Always pass via std::shared_ptr<const MeshData>.
  MeshData() = default;
  MeshData(const MeshData &) = delete;
  MeshData & operator=(const MeshData &) = delete;
  MeshData(MeshData &&) = delete;
  MeshData & operator=(MeshData &&) = delete;
};

// Loads a 3D mesh (OBJ/etc.) via Assimp, uploads vertices/faces/textures to GPU.
// No GXF dependency. Thread safety: NOT thread-safe.
class MeshLoader
{
public:
  explicit MeshLoader(cudaStream_t stream);
  ~MeshLoader();

  MeshLoader(const MeshLoader &) = delete;
  MeshLoader & operator=(const MeshLoader &) = delete;

  void load(const std::string & mesh_file_path);

  // Reload mesh only if the path changed since last load.
  void tryReload(const std::string & mesh_file_path);

  std::shared_ptr<const MeshData> getMeshData() const {return mesh_data_;}

private:
  void loadMeshData(const std::string & mesh_file_path);
  void loadTextureData(const std::string & texture_file_path);
  std::pair<Eigen::Vector3f, Eigen::Vector3f> findMinMaxVertex(const aiMesh * mesh);
  float calcMeshDiameter(const aiMesh * mesh);

  cudaStream_t stream_;
  std::shared_ptr<MeshData> mesh_data_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FOUNDATIONPOSE__FOUNDATIONPOSE_IMPL__MESH_LOADER_HPP_

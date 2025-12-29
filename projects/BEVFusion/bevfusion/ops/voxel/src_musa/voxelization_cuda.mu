#include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include <torch/types.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <cub/cub.cuh>

// ======================================================
// 判断 torch_musa 版本：
// 如果版本 > 2.0.0，使用新的 MUSAApplyUtils.muh
// 否则，使用 MUSA_Port_ApplyUtils.muh
// ======================================================
#if defined(USE_NEW_MUSA) && USE_NEW_MUSA
    #pragma message("[MUSA HELPER] Using <ATen/musa/MUSAApplyUtils.muh> for torch_musa > 2.0.0")
    #include <ATen/musa/MUSAApplyUtils.muh>
#else
    #pragma message("[MUSA HELPER] Using <ATen/musa/MUSA_PORT_ApplyUtils.muh> for torch_musa <= 2.0.0")
    #include <ATen/musa/MUSA_PORT_ApplyUtils.muh>
#endif

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_privateuseone(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_x;
      coors_offset[1] = c_y;
      coors_offset[2] = c_z;
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;

    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    // if (index >= num_points) return;
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if ((index >= num_points) || (coor_offset[0] == -1)) return;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          // point to the same coor that first show up
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          // out of boundary
          return;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
  }
}

template <typename T_int>
__global__ void determin_voxel_num(
    // const T_int* coor,
    T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,
    const int max_points, const int max_voxels, const int num_points) {
  // only calculate the coors before this coor[index]
  for (int i = 0; i < num_points; ++i) {
    // if (coor[i][0] == -1)
    //    continue;
    int point_pos_in_voxel = point_to_voxelidx[i];
    // record voxel
    if (point_pos_in_voxel == -1) {
      // out of max_points or invalid point
      continue;
    } else if (point_pos_in_voxel == 0) {
      // record new voxel
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

__global__ void nondisterministic_get_assign_pos(
    const int nthreads, const int32_t *coors_map, int32_t *pts_id,
    int32_t *coors_count, int32_t *reduce_count, int32_t *coors_order) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    if (coors_idx > -1) {
      int32_t coors_pts_pos = atomicAdd(&reduce_count[coors_idx], 1);
      pts_id[thread_idx] = coors_pts_pos;
      if (coors_pts_pos == 0) {
        coors_order[coors_idx] = atomicAdd(coors_count, 1);
      }
    }
  }
}

template<typename T>
__global__ void nondisterministic_assign_point_voxel(
    const int nthreads, const T *points, const int32_t *coors_map,
    const int32_t *pts_id, const int32_t *coors_in,
    const int32_t *reduce_count, const int32_t *coors_order,
    T *voxels, int32_t *coors, int32_t *pts_count, const int max_voxels,
    const int max_points, const int num_features, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    int coors_pts_pos = pts_id[thread_idx];
    if (coors_idx > -1) {
      int coors_pos = coors_order[coors_idx];
      if (coors_pos < max_voxels && coors_pts_pos < max_points) {
        auto voxels_offset =
            voxels + (coors_pos * max_points + coors_pts_pos) * num_features;
        auto points_offset = points + thread_idx * num_features;
        for (int k = 0; k < num_features; k++) {
          voxels_offset[k] = points_offset[k];
        }
        if (coors_pts_pos == 0) {
          pts_count[coors_pos] = min(reduce_count[coors_idx], max_points);
          auto coors_offset = coors + coors_pos * NDim;
          auto coors_in_offset = coors_in + coors_idx * NDim;
          for (int k = 0; k < NDim; k++) {
            coors_offset[k] = coors_in_offset[k];
          }
        }
      }
    }
  }
}

namespace voxelization {

/**
 * @brief Encodes 3D coordinates into a single 64-bit integer using bit manipulation.
 *
 * This function combines three integer coordinates (x, y, z) into a compact 64-bit
 * representation by assigning each coordinate a specific bit range within the output.
 * The encoding uses a spatial partitioning scheme where:
 * - x coordinate occupies bits 42-63 (21 bits)
 * - y coordinate occupies bits 21-41 (21 bits)
 * - z coordinate occupies bits 0-20 (21 bits)
 *
 * This allows efficient spatial hashing and voxel indexing in 3D applications such as
 * point cloud processing, voxel-based neural networks, and spatial data structures.
 *
 * @tparam T_int The integer type of input coordinates (typically int32_t or similar)
 * @param x The x-coordinate (will be shifted left by 42 bits)
 * @param y The y-coordinate (will be shifted left by 21 bits)
 * @param z The z-coordinate (occupies lowest 21 bits)
 * @return uint64_t The encoded coordinate as a single 64-bit integer
 *
 * @note All input coordinates should fit within 21 bits (range: 0 to 2,097,151).
 *       Values outside this range may cause overflow and incorrect encoding.
 * @note The function is force-inlined for performance optimization in critical paths.
*/
template <typename T_int>
__host__ __device__ __forceinline__ uint64_t encode_coor(T_int x, T_int y, T_int z) {
  return ((uint64_t)x << 42) | ((uint64_t)y << 21) | (uint64_t)z;
}

/**
 * @brief Generates head location markers for identifying the start of coordinate groups in sorted point clouds.
 *
 * This kernel processes sorted point indices to mark the beginning of each unique coordinate group.
 * It analyzes consecutive points in the sorted order and sets head_location flags to identify
 * where new coordinate groups begin. This is typically used in voxelization or point grouping
 * operations where we need to efficiently locate the first occurrence of each unique spatial position.
 *
 * The algorithm works by comparing each point with its predecessor in the sorted sequence:
 * - If coordinates differ from previous point: marks as group header (0)
 * - If coordinates match previous point: marks as continuation (1)
 *
 * @tparam T_int The integer type used for indexing (typically int)
 * @param coor Pointer to the original coordinate array [num_points * NDim]
 * @param sorted_indices Pointer to indices that sort the points by their coordinates
 * @param head_location Output array marking group headers (0) vs continuations (1)
 * @param num_points Total number of points to process
 * @param NDim Number of dimensions for each coordinate (typically 3 for x,y,z)
 *
 * @note The function expects points to be pre-sorted by their coordinates for correct operation.
 * @note For the first point (i=0), head_location is always set to 0 (new group start).
 * @note Only checks first 3 dimensions (x,y,z) even if NDim > 3, assuming spatial coordinates are stored first.
 * @note Uses CUDA 1D kernel loop pattern for parallel execution across GPU threads.
 *
 * @warning Points must be sorted by coordinates before calling this function, otherwise
 *          the head_location markers will be incorrect and meaningless.
 *
 * @example
 *   Given sorted points with coordinates: [(1,2,3), (1,2,3), (1,2,3), (4,5,6)]
 *   The head_location would be: [0, 1, 1, 0] indicating group starts at indices 0 and 3.
 *
 * @usage
 *   Typically called after sorting points and before voxel feature aggregation
 *   to enable efficient identification of unique spatial positions.
 */
template <typename T_int>
__global__ void generate_head(
    const int* coor,
    const int* sorted_indices,
    int* head_location,
    const int num_points,
    const int NDim
) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    if (i== 0) {
      head_location[sorted_indices[i]] = 0;
    } else {
      size_t sorted_idx = sorted_indices[i];
      size_t pre_sorted_idx = sorted_indices[i-1];
      if (coor[NDim*sorted_idx] != coor[NDim*pre_sorted_idx] || coor[NDim*sorted_idx+1] != coor[NDim*pre_sorted_idx+1] || coor[NDim*sorted_idx+2] != coor[NDim*pre_sorted_idx+2]) {
        head_location[sorted_idx] = 0;
      } else {
        head_location[sorted_idx] = 1;
      }
    }
  }
}

/**
 * @brief Generates spatial coordinate codes and initializes point indices for sorting.
 * 
 * This kernel prepares point cloud data for spatial sorting and grouping by:
 * 1. Initializing an identity mapping in the indices array [0, 1, 2, ..., num_points-1]
 * 2. Encoding valid 3D coordinates into 64-bit spatial hash codes
 * 3. Marking invalid points with a special code that sorts them last
 * 
 * The generated codes enable efficient spatial ordering of points, which is essential
 * for voxelization, duplicate point removal, and spatial data structure construction.
 * Invalid points (where the first coordinate equals -1) are excluded from spatial
 * grouping by assigning them UINT64_MAX, ensuring they appear at the end when sorted.
 * 
 * @tparam T_int Integer type for coordinate data (typically int32_t)
 * @param coor Input coordinate array storing [x,y,z,...] for each point
 * @param codes Output array for 64-bit spatial hash codes (one per point)
 * @param indices Output array initialized to point indices (identity permutation)
 * @param num_points Total number of points to process
 * @param NDim Number of dimensions per coordinate (only first 3 used for encoding)
 * 
 * @note Invalid points are identified by coor[i*NDim] == -1 and encoded as UINT64_MAX
 * @note Only the first 3 dimensions (x,y,z) are used for spatial encoding, additional
 *       dimensions are ignored regardless of NDim value
 * @note The indices array provides a stable mapping that can be sorted alongside
 *       codes to reorder points spatially while preserving original point identity
 * @note Uses encode_coor() helper function for bit-packing coordinates into 64-bit keys
 * 
 * @workflow
 *   This is typically the first step in spatial processing pipelines:
 *   1. generate_coor_code_kernel() - create codes and initialize indices
 *   2. Sort indices by codes (e.g., thrust::sort_by_key)
 *   3. Reorder points using sorted indices for spatial grouping
 * 
 * @example
 *   Input:  coor = [(1,2,3), (-1,0,0), (4,5,6)], NDim=3
 *   Output: codes  = [encode(1,2,3), UINT64_MAX, encode(4,5,6)]
 *           indices= [0, 1, 2]  // identity mapping
 */
template <typename T_int>
__global__ void generate_coor_code_kernel(
    const T_int* coor,
    uint64_t* codes,
    int* indices,
    const int num_points,
    const int NDim
) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    indices[i] = i; // 初始化原始索引
    const T_int* c = coor + i * NDim;
    if (c[0] == -1) {
      codes[i] = UINT64_MAX; // 无效点编码为最大值（排序到最后）
      return;
    }
    codes[i] = encode_coor(c[0], c[1], c[2]);
  }
}

/**
 * @brief Computes voxel indices for sorted point cloud data to establish spatial groupings.
 *
 * This kernel processes points in sorted order (by spatial hash codes) to determine:
 * 1. Which voxel each point belongs to (point_to_voxelidx)
 * 2. The index of the first point in the same voxel group (point_to_pointidx)
 *
 * The algorithm leverages pre-sorted point data to efficiently identify contiguous groups
 * of points with identical coordinates. By traversing backwards through sorted indices,
 * it counts how many consecutive points share the same spatial code and assigns
 * appropriate voxel-local indices.
 *
 * @tparam T_int Integer type for indexing and counters (typically int32_t)
 * @param coor Original coordinate array [num_points * NDim]
 * @param sorted_value Array of sorted spatial hash codes (uint64_t from encode_coor)
 * @param sorted_indices Indices that sort points by their spatial codes [num_points]
 * @param point_to_voxelidx Output: voxel-local index for each point [-1 for invalid]
 * @param point_to_pointidx Output: index of first point in same voxel group [num_points]
 * @param num_points Total number of points to process
 * @param NDim Number of dimensions per coordinate (typically 3 for x,y,z)
 * 
 * @algorithm
 *   For each point in sorted order:
 *   1. Handle invalid points (coordinate x = -1) → set both outputs to -1
 *   2. First point (sorted_i = 0) → always starts new voxel (index 0)
 *   3. First in sorted list (i = 0) → starts first voxel group
 *   4. Count consecutive duplicates backwards to determine voxel-local index
 *   5. Find first occurrence in group to set point_to_pointidx
 * 
 * @note Points MUST be pre-sorted by their spatial codes for correct grouping
 * @note Invalid points are immediately marked and skipped from further processing
 * @note The nested loops traverse backwards through sorted indices, providing O(k) 
 *       complexity where k is group size rather than O(n) for entire dataset
 * @note Only compares spatial hash equality, not raw coordinates directly
 * @note The outer loop iterates over sorted order, while inner logic works with
 *       the original point indices via sorted_indices[]
 * 
 * @workflow
 *   Typical usage in voxelization pipeline:
 *   1. encode_coor() → generate spatial hash codes
 *   2. sort points by codes → get sorted_value[], sorted_indices[]  
 *   3. compute_voxel_index_kernel() → establish voxel groupings
 *   4. aggregate features within each voxel group
 * 
 * @warning The backward traversal assumes sorted input - unsorted data will produce
 *          incorrect voxel assignments and grouping information.
 */
template <typename T_int>
__global__ void compute_voxel_index_kernel(
    const T_int* coor,
    const uint64_t* sorted_value,
    const int* sorted_indices,
    T_int* point_to_voxelidx,
    T_int* point_to_pointidx,
    const int num_points,
    const int NDim
) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    int sorted_i = sorted_indices[i];
    const T_int* c = coor + sorted_i * NDim;
    // 处理无效点
    if (c[0] == -1) {
      point_to_voxelidx[sorted_i] = -1;
      point_to_pointidx[sorted_i] = -1;
      continue;
    }
    // 初始化：第一个点 → 新体素
    if (sorted_i == 0) {
      point_to_voxelidx[sorted_i] = 0;
      point_to_pointidx[sorted_i] = 0;
      continue;
    }
    if (i == 0) {
      point_to_voxelidx[sorted_i] = 0;
      point_to_pointidx[sorted_i] = sorted_i;
      continue;
    }

    int voxel = 0;
    for (size_t j = i; j>0 && sorted_value[j] == sorted_value[j-1]; j--) {
      voxel += 1;
    }
    point_to_voxelidx[sorted_i] = voxel;

    size_t j = i;
    for (; j>=1 && sorted_value[j] == sorted_value[j-1];) {
      j--;
    }
    point_to_pointidx[sorted_i] = sorted_indices[j];
  }
}

/**
 * @brief Initializes final voxel indices and initializes per-voxel point counters.
 *
 * This kernel converts intermediate voxel indices to final voxel IDs using prefix sum offsets,
 * validates the results against the maximum voxel limit, and initializes the point count
 * array for the first max_voxels entries.
 *
 * For each point i:
 * 1. Calculates final voxel ID: coor_to_voxelidx[i] = point_to_voxelidx[i] - prefix_sum_voxel[i]
 * 2. Marks voxel IDs >= max_voxels as invalid (-1)
 * 3. Initializes num_points_per_voxel[i] = 1 for i < max_voxels (other entries unchanged)
 *
 * @tparam T_int Integer type for indexing (typically int)
 * @param num_points_per_voxel Array to store point counts per voxel (output)
 * @param point_to_voxelidx Intermediate voxel indices (input)
 * @param prefix_sum_voxel Prefix sum offsets for index conversion (input)
 * @param coor_to_voxelidx Final voxel IDs for each point (output, -1 if invalid)
 * @param max_voxels Maximum allowed voxels
 * @param num_points Total number of points
 *
 * @note Invalid voxel IDs (out of bounds) are set to -1
 * @note Only initializes the first max_voxels entries of num_points_per_voxel to 1
 */
template <typename T_int>
__global__ void determin_init_voxel_num_value(
    T_int* num_points_per_voxel,
    T_int* point_to_voxelidx,
    T_int* prefix_sum_voxel,
    T_int* coor_to_voxelidx,
    const int max_voxels,
    const int num_points) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    coor_to_voxelidx[i] = point_to_voxelidx[i] - prefix_sum_voxel[i];
    if (coor_to_voxelidx[i] >= max_voxels) {
      coor_to_voxelidx[i] = -1;
    }
    if (i < max_voxels) {
      num_points_per_voxel[i] = 1;
    }
  }
}

/**
 * @brief Assigns final voxel indices to points based on sorted spatial groups.
 *
 * This kernel processes pre-sorted point cloud data to ensure all points in the same
 * spatial group share the same voxel ID. It propagates the base voxel index from the
 * first point in each group to all subsequent points in that group.
 *
 * Key operations:
 * 1. For i=0: Calculates total voxel count using prefix sum and head location info
 * 2. For i>0: Finds the first point in current spatial group via backward traversal
 * 3. Propagates the base voxel ID from group leader (sorted_j) to current point
 *
 * @tparam T_int Integer type for indexing (typically int)
 * @param sorted_value Array of sorted spatial hash codes (points ordered by location)
 * @param sorted_indices Mapping from sorted order back to original point indices
 * @param coor_to_voxelidx Input/output: intermediate voxel IDs → final consistent IDs
 * @param voxel_num Output: total voxel count stored in voxel_num[0]
 * @param prefix_sum_voxel Prefix sum array used for count calculations
 * @param head_location Array marking group headers (0) vs continuations (1)
 * @param num_points Total number of points
 *
 * @note Points must be pre-sorted by spatial hash codes (sorted_value)
 * @note Only updates points with valid intermediate voxel IDs (!= -1)
 * @note Backward traversal efficiently finds group leaders in sorted sequence
 * @note The i=0 case handles special voxel count calculation for edge conditions
 *
 * @example
 *   Groups in sorted order: [A,A,A,B,C,C]
 *   All 'A' points get same voxel ID as first 'A' point
 *   All 'C' points get same voxel ID as first 'C' point
 */
template <typename T_int>
__global__ void determin_calculate_coor_to_voxelidx(
    const uint64_t* sorted_value,
    const int* sorted_indices,
    T_int* coor_to_voxelidx,
    T_int* voxel_num,
    T_int* prefix_sum_voxel,
    T_int* head_location,
    const int num_points) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    if (i == 0) {
      if (head_location[num_points - 1] == 1) {
        voxel_num[0] = num_points - prefix_sum_voxel[num_points-1] - 1;
      } else {
        voxel_num[0] = num_points - prefix_sum_voxel[num_points-1];
      }
    }

    int sorted_i = sorted_indices[i];
    if (i == 0) {
      continue;
    }
    size_t j = i;
    for (; j>=1 && sorted_value[j] == sorted_value[j-1];) {
      j--;
    }
    int sorted_j = sorted_indices[j];
    if (coor_to_voxelidx[sorted_i] != -1) {
      coor_to_voxelidx[sorted_i] = coor_to_voxelidx[sorted_j];
    }
  }
}

/**
 * @brief Counts the number of points in each voxel group and stores the counts.
 *
 * This kernel processes sorted point cloud data to calculate how many points belong
 * to each unique spatial group (voxel) and records these counts in the
 * num_points_per_voxel array. It identifies the last point in each spatial group
 * and then counts backwards to determine the total group size.
 *
 * The algorithm works by:
 * 1. Identifying group boundaries: points where sorted_value[i] != sorted_value[i+1]
 * 2. For each group end, counting backwards to find all consecutive duplicates
 * 3. Storing the count in num_points_per_voxel using the voxel ID from the first point
 *
 * @tparam T_int Integer type for indexing and counters (typically int32_t)
 * @param num_points_per_voxel Output array storing point counts per voxel [max_voxels]
 * @param sorted_value Array of sorted spatial hash codes (uint64_t)
 * @param sorted_indices Mapping from sorted order back to original point indices
 * @param coor_to_voxelidx Array mapping points to their voxel IDs
 * @param voxel_num_ptr Pointer to voxel counter (appears unused in this function)
 * @param max_points Maximum points allowed per voxel
 * @param max_voxels Maximum number of voxels
 * @param num_points Total number of points to process
 *
 * @algorithm
 *   For each point i in sorted order:
 *   1. Check if current point is the last in its spatial group (compare with next point)
 *   2. If NOT last in group: skip (continue to next point)
 *   3. If LAST in group:
 *      a. Count backwards through consecutive duplicates to get group size
 *      b. Get voxel ID from first point in group: coor_to_voxelidx[sorted_indices[i]]
 *      c. Store count in num_points_per_voxel[voxel_id]
 *
 * @note Points must be pre-sorted by spatial hash codes for correct group detection
 * @note Only processes the last point of each spatial group to avoid redundant counting
 * @note The loop variable 'i' is modified inside the loop (decremented in backward count)
 * @note Unused parameter: voxel_num_ptr appears to be unnecessary for this function
 * @note The backward counting ensures we find the true group size even for large groups
 *
 * @example
 *   Sorted groups: [A,A,A,B,C,C] where A,B,C are spatial codes
 *   Process:
 *   - Point 2 (last 'A'): count=3 → num_points_per_voxel[voxel_A] = 3
 *   - Point 3 (only 'B'): count=1 → num_points_per_voxel[voxel_B] = 1
 *   - Point 5 (last 'C'): count=2 → num_points_per_voxel[voxel_C] = 2
 *
 * @warning The loop modifies 'i' inside the iteration, which may affect loop progression.
 *          This is intentional for backward counting but requires careful analysis.
 *
 * @context
 *   This function is typically used near the end of voxelization pipelines to
 *   establish final point counts per voxel, enabling proper memory allocation
 *   and feature aggregation in subsequent processing stages.
 */
template <typename T_int>
__global__ void determin_get_num_points_per_voxel(
    T_int* num_points_per_voxel,
    const uint64_t* sorted_value,
    const int* sorted_indices,
    T_int* coor_to_voxelidx,
    T_int* voxel_num_ptr,
    const int max_points,
    const int max_voxels,
    const int num_points) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    if (i < (num_points-1) && sorted_value[i] == sorted_value[i+1]) {
      continue;
    } else {
      int num = 1;
      for (; i>0 && sorted_value[i] == sorted_value[i-1]; i--) {
        num++;
      }
      num_points_per_voxel[coor_to_voxelidx[sorted_indices[i]]] = num;
    }
  }
}

int hard_voxelize_gpu(const at::Tensor& points, at::Tensor& voxels,
                      at::Tensor& coors, at::Tensor& num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT(points);

  at::musa::MUSAGuard device_guard(points.device());

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  dim3 grid(std::min(at::musa::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int>
            <<<grid, block, 0, at::musa::getCurrentMUSAStream()>>>(
                points.contiguous().data_ptr<scalar_t>(),
                temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y,
                voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max,
                coors_y_max, coors_z_max, grid_x, grid_y, grid_z, num_points,
                num_features, NDim);
      }));
  musaDeviceSynchronize();
  AT_MUSA_CHECK(musaGetLastError());

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  // create some temporary variables
  auto point_to_pointidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto point_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));

  dim3 map_grid(std::min(at::musa::ATenCeilDiv(num_points, 512), 4096));
  dim3 map_block(512);
  uint64_t* d_codes;
  musaMalloc(&d_codes, sizeof(uint64_t) * num_points);
  auto d_sorted_indices_tensor = at::zeros(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto d_sorted_indices = d_sorted_indices_tensor.data_ptr<int>();

  AT_DISPATCH_ALL_TYPES(
      temp_coors.scalar_type(), "determin_duplicate", ([&] {
        auto d_coor = temp_coors.contiguous().data_ptr<int>();
        auto d_point_to_voxelidx = point_to_voxelidx.contiguous().data_ptr<int>();
        auto d_point_to_pointidx = point_to_pointidx.contiguous().data_ptr<int>();
#if 1
        generate_coor_code_kernel<int><<<map_grid, map_block>>>(
            d_coor, d_codes, d_sorted_indices, num_points, NDim
        );
        // 3. Thrust排序：按编码值升序排列（同一体素的点连续）
        thrust::device_ptr<uint64_t> dev_codes(d_codes);
        thrust::device_ptr<int> dev_indices(d_sorted_indices);
        thrust::sort_by_key(dev_codes, dev_codes + num_points, dev_indices);
        compute_voxel_index_kernel<int><<<map_grid, map_block>>>(
            d_coor, d_codes, d_sorted_indices, d_point_to_voxelidx, d_point_to_pointidx,
            num_points, NDim);
#endif
#if 0
        point_to_voxelidx_kernel<int>
            <<<map_grid, map_block, 0, at::musa::getCurrentMUSAStream()>>>(
                temp_coors.contiguous().data_ptr<int>(),
                point_to_voxelidx.contiguous().data_ptr<int>(),
                point_to_pointidx.contiguous().data_ptr<int>(), max_points,
                max_voxels, num_points, NDim);
#endif
      }));
  AT_MUSA_CHECK(musaGetLastError());

  // 3. determined voxel num and voxel's coor index
  // make the logic in the CUDA device could accelerate about 10 times
  auto coor_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto voxel_num = at::zeros(
      {
          1,
      },
      points.options().dtype(at::kInt));  // must be zero from the beginning

  AT_DISPATCH_ALL_TYPES(
      temp_coors.scalar_type(), "determin_duplicate", ([&] {
        auto point_to_voxelidx_ptr = point_to_voxelidx.contiguous().data_ptr<int>();
        auto point_to_pointidx_ptr = point_to_pointidx.contiguous().data_ptr<int>();
        auto num_points_per_voxel_ptr = num_points_per_voxel.contiguous().data_ptr<int>();
        auto coor_to_voxelidx_ptr = coor_to_voxelidx.contiguous().data_ptr<int>();
        auto voxel_num_ptr = voxel_num.contiguous().data_ptr<int>();
#if 1
        // 0. 给每个点做标记，is_head 还是 not head
        auto head_location_tensor = at::zeros(
            {
                num_points,
            },
            points.options().dtype(at::kInt));
        auto head_location = head_location_tensor.data_ptr<int>();

        generate_head<int><<<map_grid, map_block>>>(temp_coors.contiguous().data_ptr<int>(), d_sorted_indices, head_location, num_points, NDim);
        // 1. 计算voxelidx的前缀和
        auto voxel_exclusive_sum_tensor = at::zeros(
            {
                num_points,
            },
            points.options().dtype(at::kInt));
        auto voxel_exclusive_sum = voxel_exclusive_sum_tensor.data_ptr<int>();
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        // 先计算需要的临时内存大小（无需实际数据，传 nullptr 即可）
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, head_location, voxel_exclusive_sum, num_points);
        musaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, head_location, voxel_exclusive_sum, num_points);
        // 2. 初始化coor_to_voxelidx和num_points_per_voxel
        determin_init_voxel_num_value<int><<<map_grid, map_block, 0, at::musa::getCurrentMUSAStream()>>>(
            num_points_per_voxel_ptr,
            point_to_pointidx_ptr,
            voxel_exclusive_sum,
            coor_to_voxelidx_ptr,
            max_voxels,
            num_points);
        // 3. calculate coor_to_voxelidx value
        determin_calculate_coor_to_voxelidx<int><<<map_grid, map_block, 0, at::musa::getCurrentMUSAStream()>>>(
            d_codes/*sorted_value*/,
            d_sorted_indices/*sorted_indices*/,
            coor_to_voxelidx_ptr,
            voxel_num_ptr,
            voxel_exclusive_sum,
            head_location,
            num_points);
        // 4. calculate num_points_per_voxel
        determin_get_num_points_per_voxel<int><<<map_grid, map_block, 0, at::musa::getCurrentMUSAStream()>>>(
            num_points_per_voxel_ptr,
            d_codes/*sorted_value*/,
            d_sorted_indices/*sorted_indices*/,
            coor_to_voxelidx_ptr,
            voxel_num_ptr,
            max_points,
            max_voxels,
            num_points);
#endif
#if 0
        determin_voxel_num<int><<<1, 1, 0, at::musa::getCurrentMUSAStream()>>>(
            num_points_per_voxel.contiguous().data_ptr<int>(),
            point_to_voxelidx_ptr,
            point_to_pointidx_ptr,
            coor_to_voxelidx_ptr,
            voxel_num_ptr,
            max_points,
            max_voxels,
            num_points);
#endif
        musaFree(d_temp_storage);
      }));
  musaFree(d_codes);
  musaDeviceSynchronize();
  AT_MUSA_CHECK(musaGetLastError());

  // 4. copy point features to voxels
  // Step 4 & 5 could be parallel
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::musa::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_point_to_voxel<float, int>
            <<<cp_grid, cp_block, 0, at::musa::getCurrentMUSAStream()>>>(
                pts_output_size, points.contiguous().data_ptr<float>(),
                point_to_voxelidx.contiguous().data_ptr<int>(),
                coor_to_voxelidx.contiguous().data_ptr<int>(),
                voxels.contiguous().data_ptr<float>(), max_points, num_features,
                num_points, NDim);
      }));
  //   musaDeviceSynchronize();
  //   AT_MUSA_CHECK(musaGetLastError());

  // 5. copy coors of each voxels
  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(
      std::min(at::musa::ATenCeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block, 0,
                                         at::musa::getCurrentMUSAStream()>>>(
            coors_output_size, temp_coors.contiguous().data_ptr<int>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            coor_to_voxelidx.contiguous().data_ptr<int>(),
            coors.contiguous().data_ptr<int>(), num_points, NDim);
      }));
  musaDeviceSynchronize();
  AT_MUSA_CHECK(musaGetLastError());

  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];

  return voxel_num_int;
}

int nondisterministic_hard_voxelize_gpu(
    const at::Tensor &points, at::Tensor &voxels,
    at::Tensor &coors, at::Tensor &num_points_per_voxel,
    const std::vector<float> voxel_size,
    const std::vector<float> coors_range,
    const int max_points, const int max_voxels,
    const int NDim = 3) {

  CHECK_INPUT(points);

  at::musa::MUSAGuard device_guard(points.device());

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  if (num_points == 0)
    return 0;

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(torch::kInt32));

  dim3 grid(std::min(at::musa::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
    dynamic_voxelize_kernel<scalar_t, int>
    <<<grid, block, 0, at::musa::getCurrentMUSAStream()>>>(
        points.contiguous().data_ptr<scalar_t>(),
        temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y,
        voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max,
        coors_y_max, coors_z_max, grid_x, grid_y, grid_z, num_points,
        num_features, NDim);
  }));

  at::Tensor coors_map;
  at::Tensor coors_count;
  at::Tensor coors_order;
  at::Tensor reduce_count;
  at::Tensor pts_id;

  auto coors_clean = temp_coors.masked_fill(temp_coors.lt(0).any(-1, true), -1);

  std::tie(temp_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, false);

  if (temp_coors.index({0, 0}).lt(0).item<bool>()) {
    // the first element of temp_coors is (-1,-1,-1) and should be removed
    temp_coors = temp_coors.slice(0, 1);
    coors_map = coors_map - 1;
  }

  int num_coors = temp_coors.size(0);
  temp_coors = temp_coors.to(torch::kInt32);
  coors_map = coors_map.to(torch::kInt32);

  coors_count = coors_map.new_zeros(1);
  coors_order = coors_map.new_empty(num_coors);
  reduce_count = coors_map.new_zeros(num_coors);
  pts_id = coors_map.new_zeros(num_points);

  dim3 cp_grid(std::min(at::musa::ATenCeilDiv(num_points, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "get_assign_pos", ([&] {
    nondisterministic_get_assign_pos<<<cp_grid, cp_block, 0,
    at::musa::getCurrentMUSAStream()>>>(
        num_points,
        coors_map.contiguous().data_ptr<int32_t>(),
        pts_id.contiguous().data_ptr<int32_t>(),
        coors_count.contiguous().data_ptr<int32_t>(),
        reduce_count.contiguous().data_ptr<int32_t>(),
        coors_order.contiguous().data_ptr<int32_t>());
  }));

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
    nondisterministic_assign_point_voxel<scalar_t>
    <<<cp_grid, cp_block, 0, at::musa::getCurrentMUSAStream()>>>(
        num_points, points.contiguous().data_ptr<scalar_t>(),
        coors_map.contiguous().data_ptr<int32_t>(),
        pts_id.contiguous().data_ptr<int32_t>(),
        temp_coors.contiguous().data_ptr<int32_t>(),
        reduce_count.contiguous().data_ptr<int32_t>(),
        coors_order.contiguous().data_ptr<int32_t>(),
        voxels.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int32_t>(),
        num_points_per_voxel.contiguous().data_ptr<int32_t>(),
        max_voxels, max_points,
        num_features, NDim);
  }));
  AT_MUSA_CHECK(musaGetLastError());
  return max_voxels < num_coors ? max_voxels : num_coors;
}

void dynamic_voxelize_gpu(const at::Tensor& points, at::Tensor& coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT(points);

  at::musa::MUSAGuard device_guard(points.device());

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  const int col_blocks = at::musa::ATenCeilDiv(num_points, threadsPerBlock);
  dim3 blocks(col_blocks);
  dim3 threads(threadsPerBlock);
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_kernel", [&] {
    dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
  });
  musaDeviceSynchronize();
  AT_MUSA_CHECK(musaGetLastError());

  return;
}

}  // namespace voxelization

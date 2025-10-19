#include <pybind11/pybind11.h>
#include "point2voxel.h"

namespace py = pybind11;

PYBIND11_MODULE(point2voxel, m) {
    m.def("Point3d2VoxelMean",
        &points_to_voxel_3d_np_mean<float, 3>, "matrix tensor_square",
        "points"_a = 1, "voxels"_a = 2, "voxel_point_mask"_a = 3, "means"_a = 4,
        "coors"_a = 5, "num_points_per_voxel"_a = 6, "coor_to_voxelidx"_a = 7,
        "voxel_size"_a = 8, "coors_range"_a = 9, "max_points"_a = 10,
        "max_voxels"_a = 11);
  m.def("Point3d2VoxelMean",
        &points_to_voxel_3d_np_mean<double, 3>, "matrix tensor_square",
        "points"_a = 1, "voxels"_a = 2, "voxel_point_mask"_a = 3, "means"_a = 4,
        "coors"_a = 5, "num_points_per_voxel"_a = 6, "coor_to_voxelidx"_a = 7,
        "voxel_size"_a = 8, "coors_range"_a = 9, "max_points"_a = 10,
        "max_voxels"_a = 11);
  m.def("Point3d2Voxel", &points_to_voxel_3d_np<float, 3>,
        "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
        "voxel_point_mask"_a = 3, "coors"_a = 4, "num_points_per_voxel"_a = 5,
        "coor_to_voxelidx"_a = 6, "voxel_size"_a = 7, "coors_range"_a = 8,
        "max_points"_a = 9, "max_voxels"_a = 10);
  m.def("Point3d2Voxel", &points_to_voxel_3d_np<double, 3>,
        "matrix tensor_square", "points"_a = 1, "voxels"_a = 2,
        "voxel_point_mask"_a = 3, "coors"_a = 4, "num_points_per_voxel"_a = 5,
        "coor_to_voxelidx"_a = 6, "voxel_size"_a = 7, "coors_range"_a = 8,
        "max_points"_a = 9, "max_voxels"_a = 10);
}


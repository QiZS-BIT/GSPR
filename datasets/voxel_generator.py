import time
import torch
import torch.nn as nn
import numpy as np
try:
    from point2voxel import Point3d2VoxelMean, Point3d2Voxel
except ImportError:
    pass


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    coor_to_voxelidx,
                    max_points=35,
                    max_voxels=20000,
                    full_mean=True,
                    block_filtering=False):
    """convert 3d points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 0.8ms(~6k voxels)
    with c++ and 3.2ghz cpu.
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        coor_to_voxelidx: int array. used as a dense map.
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for voxelnet, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
        full_mean: bool. if true, all empty points in voxel will be filled with mean
            of exist points.
        block_filtering: filter voxels by height. used for lidar point cloud.
            use some visualization tool to see filtered result.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor. zyx format.
        num_points_per_voxel: [M] int32 tensor.
    """
    if full_mean:
        assert block_filtering is False
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    voxels = np.zeros(shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    voxel_point_mask = np.zeros(shape=(max_voxels, max_points), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    res = {
        "voxels": voxels,
        "coordinates": coors,
        "num_points_per_voxel": num_points_per_voxel,
        "voxel_point_mask": voxel_point_mask,
    }
    if full_mean:
        means = np.zeros(shape=(max_voxels, points.shape[-1]), dtype=points.dtype)
        voxel_num = Point3d2VoxelMean(points, voxels, voxel_point_mask, means, coors, num_points_per_voxel,
                                      coor_to_voxelidx, voxel_size.tolist(), coors_range.tolist(),
                                      max_points, max_voxels)
    else:
        voxel_num = Point3d2Voxel(points, voxels, voxel_point_mask, coors, num_points_per_voxel,
                                  coor_to_voxelidx, voxel_size.tolist(), coors_range.tolist(),
                                  max_points, max_voxels)
    res["voxel_num"] = voxel_num
    res["voxel_point_mask"] = res["voxel_point_mask"].reshape(-1, max_points, 1)
    return res


class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 full_mean=True):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._full_mean = full_mean

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(points, self._voxel_size,
                              self._point_cloud_range, self._coor_to_voxelidx,
                              self._max_num_points, max_voxels
                              or self._max_voxels, self._full_mean)
        voxels = res["voxels"]
        coors = res["coordinates"]
        num_points_per_voxel = res["num_points_per_voxel"]
        voxel_num = res["voxel_num"]
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]

        return voxels, coors, num_points_per_voxel

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size

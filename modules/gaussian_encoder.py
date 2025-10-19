import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gcn3d import Conv_layer, get_neighbor_index


class MeanVFE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_per_voxel):
        voxel_features = x
        points_mean = voxel_features[:, :, :, :].sum(axis=2)
        points_mean = points_mean / num_per_voxel[:, :, None]
        return points_mean


class GaussianEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_vfe = MeanVFE()

    def forward(self, x, num_per_voxel):
        weighted_x = x
        mean_x = self.mean_vfe(weighted_x, num_per_voxel)
        return mean_x

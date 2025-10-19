import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gcn3d import Conv_layer, Pool_layer, get_neighbor_index, indexing_neighbor


class GCNBackbone(nn.Module):
    def __init__(self, neighbor_num=5, support_num=3, input_feat=56):
        super().__init__()
        self.input_feat_dim = input_feat
        self.neighbor_num = neighbor_num

        self.conv_1 = Conv_layer(input_feat, 64, support_num=support_num)
        # self.conv_1_add = Conv_layer(64, 64, support_num=support_num)
        self.pool_1 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = Conv_layer(64, 128, support_num=support_num)
        self.conv_3 = Conv_layer(128, 256, support_num=support_num)
        self.pool_2 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = Conv_layer(256, 512, support_num=support_num)

    def forward(self, vertices: "(bs, vertice_num, 3)", features: "(bs, feature_num, n)"):
        bs, vertice_num, _ = vertices.size()

        # neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        # fm_1 = self.conv_1(neighbor_index, vertices, features)
        # fm_1 = F.relu(fm_1, inplace=True)
        # fm_1_add = self.conv_1_add(neighbor_index, vertices, fm_1)
        # fm_1_add = F.relu(fm_1_add, inplace=True)
        # vertices, fm_1_add, sample_1 = self.pool_1(vertices, fm_1_add)
        # neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        #
        # fm_2 = self.conv_2(neighbor_index, vertices, fm_1_add)
        # fm_2 = F.relu(fm_2, inplace=True)
        # fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        # fm_3 = F.relu(fm_3, inplace=True)
        # vertices, fm_3, sample_2 = self.pool_2(vertices, fm_3)
        # neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        fm_1 = self.conv_1(neighbor_index, vertices, features)
        fm_1 = F.relu(fm_1, inplace=True)
        vertices, fm_1, sample_1 = self.pool_1(vertices, fm_1)
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace=True)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace=True)
        vertices, fm_3, sample_2 = self.pool_2(vertices, fm_3)
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        fm_4 = F.relu(fm_4, inplace=True)

        return vertices, fm_4, neighbor_index, sample_1, sample_2


class SpatialEncodingLayer(nn.Module):
    def __init__(self, pos_in_channels, feat_in_channels, support_num=1):
        super().__init__()
        self.pos_in_channels = pos_in_channels
        self.feat_in_channels = feat_in_channels
        self.support_num = support_num

        self.pe = nn.Sequential(
            nn.Linear(self.pos_in_channels, self.feat_in_channels),
            nn.ReLU(),
            nn.Linear(self.feat_in_channels, self.feat_in_channels)
        )
        self.conv_5 = Conv_layer(self.feat_in_channels, self.feat_in_channels, self.support_num)
        self.conv_6 = Conv_layer(self.feat_in_channels, self.feat_in_channels, self.support_num)

    def forward(self, pos, feat, vertice, neighbor_index):
        positional_encoding = self.pe(pos)
        feat = feat + positional_encoding
        feat = self.conv_5(neighbor_index, vertice, feat)
        feat = F.relu(feat, inplace=True)
        feat = self.conv_6(neighbor_index, vertice, feat)
        feat = F.relu(feat, inplace=True)

        return feat

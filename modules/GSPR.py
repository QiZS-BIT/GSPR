import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gcn3d import pc_normalize, get_neighbor_direction_norm
from modules.gcn_backbone import GCNBackbone, SpatialEncodingLayer
from modules.netvlad import NetVLADLoupe


class GSPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GCNBackbone(neighbor_num=25, support_num=1)
        self.attention_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            activation='relu',
            batch_first=True,
            dropout=0.
        )
        self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
        self.vlad_layer = NetVLADLoupe(
            feature_size=512,
            cluster_size=64,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )

        self.spatial_encoding = SpatialEncodingLayer(3, 512)

    def forward(self, x):
        x = x.squeeze(0)
        x_norm = pc_normalize(x[:, :, :3])

        vertice, feat, neighbor_index, sp_1, sp_2 = self.backbone(x_norm, x[:, :, 3:])

        x_pool_1 = x[:, sp_1, :3]
        x_pool_2 = x_pool_1[:, sp_2, :]
        feat_with_emb = self.spatial_encoding(x_pool_2, feat, vertice, neighbor_index)

        feat_with_emb = self.attention(feat_with_emb)

        descriptor = self.vlad_layer(feat_with_emb)
        descriptor = F.normalize(descriptor, dim=1)
        return descriptor

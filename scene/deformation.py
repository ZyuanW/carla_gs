import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=None, args=None):
        super().__init__()
        if skips is None:
            skips = []
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.args = args
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.ratio = 0

        if getattr(args, 'empty_voxel', False):
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])

        if getattr(args, 'static_mlp', False):
            self.static_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.W, self.W),
                nn.ReLU(),
                nn.Linear(self.W, 1)
            )

        self._initialize_network()

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb", xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if getattr(self.args, 'empty_voxel', False):
            self.empty_voxel.set_aabb(xyz_max, xyz_min)

    def _initialize_network(self):
        if self.grid_pe != 0:
            grid_out_dim = self.grid.feat_dim * 3
        else:
            grid_out_dim = self.grid.feat_dim

        input_dim = 4 if self.no_grid else grid_out_dim

        layers = [nn.Linear(input_dim, self.W)]
        for _ in range(self.D - 1):
            layers.extend([nn.ReLU(), nn.Linear(self.W, self.W)])
        self.feature_out = nn.Sequential(*layers)

        self.pos_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3)
        )
        self.scales_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3)
        )
        self.rotations_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 4)
        )
        self.opacity_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 1)
        )
        self.shs_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 16 * 3)
        )

        if getattr(self.args, 'feat_head', False):
            self.dino_head = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:, :3], time_emb[:, :1]], dim=-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:, :3], time_emb[:, :1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            h = grid_feature

        hidden = self.feature_out(h)
        return hidden

    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self._forward_static(rays_pts_emb[:, :3])
        else:
            return self._forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def _forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb)
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb + dx

    def _forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        mask = self._compute_mask(hidden, rays_pts_emb)

        pts = self._compute_pts(hidden, rays_pts_emb, mask)
        scales = self._compute_scales(hidden, scales_emb, mask)
        rotations = self._compute_rotations(hidden, rotations_emb)
        opacity = self._compute_opacity(hidden, opacity_emb, mask)
        shs, dshs = self._compute_shs(hidden, shs_emb, mask)
        feat = self.dino_head(hidden) if getattr(self.args, 'feat_head', False) else None

        return pts, scales, rotations, opacity, shs, None, feat, dshs

    def _compute_mask(self, hidden, rays_pts_emb):
        if getattr(self.args, 'static_mlp', False):
            return self.static_mlp(hidden)
        elif getattr(self.args, 'empty_voxel', False):
            return self.empty_voxel(rays_pts_emb[:, :3])
        else:
            return torch.ones_like(rays_pts_emb[:, :1])

    def _compute_pts(self, hidden, rays_pts_emb, mask):
        if getattr(self.args, 'no_dx', False):
            return rays_pts_emb[:, :3]
        else:
            dx = self.pos_deform(hidden)
            return rays_pts_emb[:, :3] * mask + dx

    def _compute_scales(self, hidden, scales_emb, mask):
        if getattr(self.args, 'no_ds', False):
            return scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)
            return scales_emb[:, :3] * mask + ds

    def _compute_rotations(self, hidden, rotations_emb):
        if getattr(self.args, 'no_dr', False):
            return rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)
            if getattr(self.args, 'apply_rotation', False):
                return batch_quaternion_multiply(rotations_emb, dr)
            else:
                return rotations_emb[:, :4] + dr

    def _compute_opacity(self, hidden, opacity_emb, mask):
        if getattr(self.args, 'no_do', False):
            return opacity_emb[:, :1]
        else:
            do = self.opacity_deform(hidden)
            return opacity_emb[:, :1] * mask + do

    def _compute_shs(self, hidden, shs_emb, mask):
        if getattr(self.args, 'no_dshs', False):
            return shs_emb, None
        else:
            dshs = self.shs_deform(hidden).view(shs_emb.size(0), 16, 3)
            shs = shs_emb * mask.unsqueeze(-1) + dshs
            return shs, dshs

    def get_mlp_parameters(self):
        return [param for name, param in self.named_parameters() if "grid" not in name]

    def get_grid_parameters(self):
        return [param for name, param in self.named_parameters() if "grid" in name]
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        ## times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        
        ## time_emb = poc_fre(times_sel, self.time_poc)
        ## times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs, dx, feat, dshs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel) # [N, 1]
        return means3D, scales, rotations, opacity, shs, dx , feat, dshs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb
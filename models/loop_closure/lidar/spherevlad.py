'''
Author: Shiqi Zhao
SphereVLAD
'''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid

from ..aggregator.netvlad import NetVLAD


class SphereVLAD(nn.Module):
    '''
    sphvlad with bn
    '''
    def __init__(self, config):
        super(SphereVLAD, self).__init__()

        bandwidth = (int)(config.DATA.SPH_PROJ.IMAGE_SIZE[0]/2.0)

        grid_s2 = s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.conv1 = nn.Sequential(
                        S2Convolution(nfeature_in=1, nfeature_out=8,
                                      b_in=bandwidth, b_out=bandwidth//2, grid=grid_s2),
                        nn.BatchNorm3d(8),
                        nn.ReLU())

        self.conv2 = nn.Sequential(
                        SO3Convolution(nfeature_in=8, nfeature_out=8,
                                       b_in=bandwidth//2, b_out=bandwidth//4, grid=grid_so3_1),
                        nn.BatchNorm3d(8),
                        nn.ReLU())

        self.conv3 = nn.Sequential(
                        SO3Convolution(nfeature_in=8, nfeature_out=16,
                                       b_in=bandwidth//4, b_out=bandwidth//8, grid=grid_so3_2),
                        nn.BatchNorm3d(16),
                        nn.ReLU())

        self.conv4 = nn.Sequential(
                        SO3Convolution(nfeature_in=16, nfeature_out=16,
                                       b_in=bandwidth//8, b_out=bandwidth//8, grid=grid_so3_3),
                        nn.BatchNorm3d(16),
                        nn.ReLU())

        self.vlad = NetVLAD(num_clusters=config.MODEL.NETVLAD.CLUSTER_NUM, 
                            dim=config.MODEL.NETVLAD.FEATURE_DIM, 
                            normalize_input=config.MODEL.NETVLAD.NORMALIZE_INPUT,
                            output_dim=config.MODEL.NETVLAD.OUTPUT_DIM,
                            gate=config.MODEL.NETVLAD.GATE)

    
    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # reshape
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]*x.shape[4])
        # vlad
        x = self.vlad(x)
        return x
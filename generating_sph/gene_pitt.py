#
# Created on Sun Mar 07 2021
#
# The MIT License (MIT)
# Copyright (c) 2021 Max Yin
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import os
import sys
import numpy as np
from glob import glob
import open3d as o3d
import argparse
from torch import square
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


from config import cfg
from utils import log_print, LaserProjection, pcd_downsample


#!============================================================================================#
#! Parameters for Dataset and Network
#!============================================================================================#


def para_args():
    parser = argparse.ArgumentParser(description="Network configurations!")
    parser.add_argument("--config-network", default="../config/network/spherevlad.yaml", metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--config-dataset", default="../config/dataset/PITT.yaml", metavar="FILE",
                        help="path to config file", type=str)
    args = parser.parse_args()
    return args

def find_boundry(num):
    pose_list = glob(f'/data_hdd_1/SphereVLAD++/dataset/KITTI360/train_{num}/*_pose6d.npy')
    boundry = []
    num_list = [int(a.split('_pose6d.npy')[0].split('/')[-1]) for a in pose_list]
    num_list = sorted(num_list)
    for i in range(max(num_list)+1):
        if (i in num_list) and (i+1 not in num_list):
            boundry.append(i)
        if (i in num_list) and (i-1 not in num_list):
            boundry.append(i)
    return boundry

def generate_data(cfg, index, mode, slice_size=50, interval=1):

    path_base = '{}{}'.format(cfg.DATA.BENCHMARK_DIR, cfg.DATA.DATASET_NAME)
    out_path = '{}/{}_{}'.format(path_base, mode, index)
    os.system('rm -rf {}'.format(out_path))
    os.mkdir('{}'.format(out_path))

    log_print("Load data {}".format(out_path), 'b')

    #! Define range to image projection
    projection = LaserProjection(None, 
                                top_size=cfg.DATA.BEV_PROJ.IMAGE_SIZE,
                                z_range=cfg.DATA.BEV_PROJ.Z_RANGE,
                                sph_size=cfg.DATA.SPH_PROJ.IMAGE_SIZE,
                                fov_range=cfg.DATA.SPH_PROJ.FOV)
    
    trj_6D = np.loadtxt("{}/DATA/Train{}/poses.txt".format(path_base, index))

    #! Obtain Projections
    map_pcd = o3d.io.read_point_cloud(
        "{}/DATA/Train{}/cloudGlobal.pcd".format(path_base, index))
    pcd_tree = o3d.geometry.KDTreeFlann(map_pcd)
    
    #* generate submaps
    last_pose = np.array([9.9e9, 9.9e9, 9.9e9])
    count = 0

    for im_idx, pose in tqdm(enumerate(trj_6D), total=len(trj_6D)):
        # select interval
        if np.linalg.norm(pose[:3] - last_pose, ord=2) < interval:
            continue
        last_pose = pose[:3]

        # generate ball query pointcloud
        [k, p_idx, _] = pcd_tree.search_radius_vector_3d(
            pose[:3], cfg.DATA.SPH_PROJ.RADIUS)
        pcd_data = np.asarray(map_pcd.points)[p_idx, :]
        pcd_data -= pose[:3]
        if pcd_data.shape[0] == 0:
            continue

        # save pose files
        np.save(
            "{}/{:06d}_pose.npy".format(out_path, count+1), pose[:3])
        np.save(
            "{}/{:06d}_pose6d.npy".format(out_path, count+1), pose)

        # generate data for spherical view
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data)
        trans_matrix = np.eye(4)
        rot = R.from_rotvec(np.array([pose[3], pose[4], pose[5]]))
        trans_matrix[0:3, 0:3] = rot.inv().as_matrix()
        pcd.transform(trans_matrix)
        pcd = pcd.voxel_down_sample(0.5)
        sph_img = projection.do_sph_projection(np.array(pcd.points))
        im = Image.fromarray((sph_img * 255).astype(np.uint8))
        im.save("{}/{:06d}_sph.png".format(out_path, count+1))
        
        count += 1

if __name__ == "__main__":
    args = para_args()
    cfg.merge_from_file(args.config_network)
    cfg.merge_from_file(args.config_dataset)
    cfg.freeze()
    gen_data = generate_data

    # ANCHOR prepare train data
    for index in list(set(cfg.DATA.TRAIN_LIST + cfg.DATA.TEST_LIST)):
        gen_data(cfg, index, 'train')
    for index in cfg.DATA.VAL_LIST:
        gen_data(cfg, index, 'val')

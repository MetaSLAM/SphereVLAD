'''
Author: Shiqi Zhao
evaluate_utils.py
'''
import tqdm
from glob import glob
from copy import copy, deepcopy

import torch
import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.neighbors import KDTree
from PIL import Image
from scipy.spatial.transform import Rotation as R

from utils import pc_normalize, pcd_downsample, LaserProjection, log_print
from dataloader.data_augmentation import Augment_Point_Data, Augment_SPH_Data, Augment_RGB_Data


class EvaluationPitts():
    def __init__(self, config, model, device, number_neighbors=20) -> None:
        self.config = config
        self.number_neighbors = number_neighbors
        self.model = model
        self.device = device
        if config.MODEL.TYPE == "Lidar":
            self.trans_lidar = Augment_Point_Data(is_train=False)
        elif config.MODEL.TYPE == "LiSPH":
            self.trans_lidar_sph = Augment_SPH_Data(img_size=config.DATA.SPH_PROJ.IMAGE_SIZE,
                                                    is_train=False)
            self.projection = LaserProjection(device=device,
                                              top_size=config.DATA.BEV_PROJ.IMAGE_SIZE,
                                              z_range=config.DATA.BEV_PROJ.Z_RANGE,
                                              sph_size=config.DATA.SPH_PROJ.IMAGE_SIZE,
                                              fov_range=config.DATA.SPH_PROJ.FOV)
        else:
            raise NotImplementedError('Please Try Predefined Types')
        
    def get_features_recall(self, traj_num, trans_IDX, rot_IDX, rand_idx=0):
        database = []
        query = []
        running_time = 0
        
        # Set global map
        if self.config.DATA.DATASET_NAME == 'PITT':
            global_map = o3d.io.read_point_cloud('{}/{}/DATA/Train{}/cloudGlobal.pcd'.format(
                                                 self.config.DATA.BENCHMARK_DIR, self.config.DATA.DATASET_NAME,traj_num))
            bbox_pnv = o3d.geometry.AxisAlignedBoundingBox(
                        np.array([-20, -20, 0.8]),
                        np.array([ 20,  20, 100.0]))
        map_tree = o3d.geometry.KDTreeFlann(global_map)
        
        # Set filelist for test
        file_list = sorted(glob('{}/{}/train_{}/*_sph.png'.format(self.config.DATA.BENCHMARK_DIR, self.config.DATA.DATASET_NAME, traj_num)))
        file_list = file_list[::10]

        # load database
        for file_name in tqdm.tqdm(file_list):
            if self.config.MODEL.TYPE == "LiSPH":
                queries_img = Image.open(file_name)
                frame = [self.trans_lidar_sph(queries_img).to(self.device, dtype=torch.float)]
            embedding, t_gpu = self.model.infer_frame(frame, t=True)
            database.append(embedding)
            running_time += t_gpu

        # load the test query
        for file_name in tqdm.tqdm(file_list):
            # preprocess generate test query
            pose = np.load('{}_pose6d.npy'.format(
                    file_name.split('_sph.png')[0]))

            # * Add fixed transformation
            if rand_idx == 0:
                trans_idx = trans_IDX + np.random.random(1)[0]-0.5 # 0.5m noise
                rot_idx = rot_IDX + 5*(np.random.random(1)[0]-0.5) # 2.5° noise
            # * Add random transformation
            else:
                trans_idx = 2 * trans_IDX * (np.random.random(1)[0]-0.5)
                rot_idx = 2 * rot_IDX * (np.random.random(1)[0]-0.5)
            pose[0] += trans_idx
            pose[1] += trans_idx
            frame = []
            # generate sphrical images
            [k, p_idx, _] = map_tree.search_radius_vector_3d(
                    pose[:3], self.config.DATA.SPH_PROJ.RADIUS)
            pcd_data = np.asarray(global_map.points)[p_idx, :]
            pose[5] += rot_idx * np.pi/180
            pcd_data -= pose[:3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_data)
            rot = R.from_rotvec(np.array([pose[3], pose[4], pose[5]]))
            if self.config.MODEL.TYPE in ["LiSPH"]:
                trans_matrix = np.eye(4)
                trans_matrix[0:3, 0:3] = rot.inv().as_matrix()
                pcd.transform(trans_matrix)
                pcd = pcd.voxel_down_sample(0.5)
                sph_img = self.projection.do_sph_projection(np.asarray(pcd.points))
                im = Image.fromarray((sph_img * 255).astype(np.uint8))
                frame.append(self.trans_lidar_sph(im).to(self.device, dtype=torch.float))
            # load test query into network
            query.append(self.model.infer_frame(frame))
        log_print(f'{self.config.DATA.DATASET_NAME} {traj_num}:', 'b')
        print("Descriptor Generation Done!")
        print("Genrate Speed {}s/frame".format(running_time/len(file_list)))
        
        _,recall_nums,_,one_percent_recall = self.get_recall(np.array(database), np.array(query), num_neighbors=self.number_neighbors)
        log_print(f'Top one percent recall is {one_percent_recall}')

        return recall_nums, one_percent_recall, running_time
    
    def get_recall(self, database_feature, queries_feature, num_neighbors=30):
        database_output = database_feature
        queries_output = queries_feature

        database_nbrs = KDTree(database_output)
        # num_neighbors = (int)(queries_feature.shape[0]*0.01)
        recall = [0] * num_neighbors

        top1_similarity_score = []
        one_percent_retrieved = 0
        threshold = max(int(round(len(database_output) / 100.0)), 1)

        num_evaluated = 0
        topk_dict = {}
        top_recalls = np.zeros(num_neighbors+2)

        for i in range(len(queries_output)):

            true_neighbors = [i]
            if len(true_neighbors) == 0:
                continue
            num_evaluated += 1

            distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
            # indices = np.setdiff1d(indices[0], [i])
            indices = indices[0]

            for j in range(0, len(indices)):
                if indices[j] in true_neighbors:
                    if (j == 0):
                        similarity = np.dot(queries_output[i], database_output[indices[j]])
                        top1_similarity_score.append(similarity)
                    recall[j] += 1
                    break

            if len(list(set(indices[0:threshold]).intersection(set(true_neighbors)))) > 0:
                one_percent_retrieved += 1
            
            for recall_num in range(num_neighbors):
                if len(list(set(indices[0:recall_num+1]).intersection(set(true_neighbors)))) > 0:
                    top_recalls[recall_num] += 1

        top_recalls[-2] = one_percent_retrieved
        top_recalls[-1] = num_evaluated
        one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
        top_one_recall = (top_recalls[0] / float(num_evaluated)) * 100
        top_five_recall = (top_recalls[4] / float(num_evaluated)) * 100
        top_ten_recall = (top_recalls[9] / float(num_evaluated)) * 100
        recall = (np.cumsum(recall) / float(num_evaluated)) * 100

        return (top_one_recall, top_five_recall, top_ten_recall), \
            top_recalls, \
            top1_similarity_score, one_percent_recall

    @staticmethod
    def apply_noise(pcd, mu=0, sigma=0.1):
        noisy_pcd = deepcopy(pcd)
        points = np.asarray(noisy_pcd.points)
        points += np.clip(np.random.normal(mu, sigma, size=points.shape), -1, 1)
        noisy_pcd.points = o3d.utility.Vector3dVector(points)
        return noisy_pcd
'''
Author: Peng Yin, Shiqi Zhao
train.py

Dataset wrapper for Pittsburgh Dataset modified based on PointNetVLAD
Original Code: https://github.com/mikacuy/pointnetvlad
'''
import os
import random
from glob import glob

import pickle
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.neighbors import KDTree

from .triplet_dataloader import TripletDataLoader
from utils import log_print, pc_normalize


class PittsburghDataset(TripletDataLoader):
    """Dataloader Wrapper for Pittsburgh Dataset"""
    def __init__(self, config, is_train):
        super().__init__(config, is_train)
        self.dataset_dir = os.path.join(config.DATA.BENCHMARK_DIR, str(config.DATA.DATASET_NAME))
        if config.MODEL.TYPE not in ["Lidar", "LiSPH"]:
            raise ValueError("Pittsburgh dataset only provides Lidar data!")
        if is_train:
            self.generate_pickles()
            self.queries = self.get_queries_dict(os.path.join(self.dataset_dir, config.DATA.TRAIN_PICKLE))
            log_print("Number of training tuples: %d" % len(self.queries), "y")
        else:
            self.queries = self.get_queries_dict(os.path.join(self.dataset_dir, config.DATA.VAL_PICKLE))
            log_print("Number of val tuples: %d" % len(self.queries), "y")
        self.file_idxs = np.arange(0, len(self.queries.keys()))
    
    def load_file_func(self, filename):
        if self.config.MODEL.TYPE == "Lidar" or self.branch == "Lidar":
            pcd = np.asarray(o3d.io.read_point_cloud(filename + ".pcd").points)
            if pcd.shape[0] != 4096:
                raise ValueError('{}.pcd does not have sufficient points'.format(filename))
            pcd = pc_normalize(pcd)
            output = self.lidar_data_aug(pcd)
        elif self.config.MODEL.TYPE == "LiSPH" or self.branch == "LiSPH":
            sph_img = Image.open(filename + "_sph.png")
            output = self.lisph_data_aug(sph_img)
        return output
        
    def generate_pickles(self):
        if self.config.TRAINING.IS_TRAIN:
            pair_train = os.path.join(
                self.dataset_dir, self.config.DATA.TRAIN_PICKLE)
            pair_test = os.path.join(
                self.dataset_dir, self.config.DATA.VAL_PICKLE)
            #! Load pickles if exist
            if  os.path.exists(pair_train) and os.path.exists(pair_test):
                log_print("Load previous pickles", "b")
                return
            # ANCHOR generate train/val query for 
            pair_train = self.get_df(self.dataset_dir, 'train', is_shuffle=False)
            pair_train = pair_train.sample(frac=1).reset_index(drop=True)
            pair_val = self.get_df(self.dataset_dir, 'val', is_shuffle=False)
            pair_val = pair_val.sample(frac=1).reset_index(drop=True)

            self.construct_query_dict(pair_train, self.config.DATA.TRAIN_PICKLE)
            self.construct_query_dict(pair_val, self.config.DATA.VAL_PICKLE)
            log_print("Generated pickles!\n", "g")

    def get_from_folder(self, data_path, index):

        all_file_id = []
        pose_data = glob(data_path+'/*_pose.npy')
        for file_name in pose_data:
            all_file_id.append(file_name.split('_pose')[-2])
        all_file_id.sort()
        all_data_df = pd.DataFrame(all_file_id, columns=["file"])
        all_data_df["pcd_position_x"] = all_data_df["file"].apply(
            lambda x: np.load(x + '_pose.npy')[0])
        all_data_df["pcd_position_y"] = all_data_df["file"].apply(
            lambda x: np.load(x + '_pose.npy')[1])
        all_data_df["pcd_position_z"] = all_data_df["file"].apply(
            lambda x: np.load(x + '_pose.npy')[2])
        all_data_df["date"] = all_data_df["file"].apply(
            lambda x: x.split('/')[-2])
        all_data_df.reset_index(drop=True, inplace=True)
        return all_data_df

    def get_df(self, dataset_dir, type='train', is_shuffle=False):
        file_df = pd.DataFrame()
        file_dirs = []
        if type == 'train':
            for i in self.config.DATA.TRAIN_LIST:
                file_dirs.append("{}/train_{}/".format(dataset_dir, i))
        elif type == 'val':
            for i in self.config.DATA.VAL_LIST:
                file_dirs.append("{}/val_{}/".format(dataset_dir, i))
        elif type == 'test':
            for i in self.config.DATA.TEST_LIST:
                file_dirs.append("{}/test_{}/".format(dataset_dir, i))
        file_dirs.sort()
        for index, folder in enumerate(file_dirs):
            data_df = self.get_from_folder(folder, index)
            file_df = pd.concat([file_df, data_df], ignore_index=True)
        if is_shuffle:
            file_df.sample(frac=1).reset_index(drop=True)

        return file_df
    
    def construct_query_dict(self, data_df, filename):
        data_df.reset_index(drop=True, inplace=True)

        tree = KDTree(
            data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]])
        ind_nn = tree.query_radius(data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]],
                                   r=self.config.DATA.POSITIVES_RADIUS)
        ind_r = tree.query_radius(data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]],
                                  r=self.config.DATA.NEGATIVES_RADIUS)
        ind_traj = tree.query_radius(data_df[["pcd_position_x", "pcd_position_y", "pcd_position_z"]],
                                     r=self.config.DATA.TRAJ_RADIUS)

        queries = {}
        for i in tqdm(range(len(ind_nn)), total=len(ind_nn), desc='construct queries', leave=False):
            query = data_df.iloc[i]["file"]
            positives = np.setdiff1d(ind_nn[i], [i]).tolist()
            negatives = np.setdiff1d(ind_traj[i], ind_r[i]).tolist()

            random.shuffle(negatives)
            random.shuffle(positives)

            queries[i] = {"query": query, "positives": positives, "negatives": negatives}

        with open(os.path.join(self.dataset_dir, filename), 'wb') as handle:
            pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

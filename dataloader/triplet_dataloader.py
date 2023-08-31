'''
Author: Shiqi Zhao, Peng Yin
TripletDataLoader
'''

import os
import pickle
import random
from abc import abstractmethod

import torch
import numpy as np
from torch.utils.data import Dataset

from .data_augmentation import Augment_Point_Data, Augment_RGB_Data, Augment_SPH_Data

random.seed(12345)

        
class TripletDataLoader(Dataset):
    """Dataloader for tripletLoss or qradrupletLoss"""
    def __init__(self, config, is_train):
        self.config = config
        self.is_train = is_train
        self.data_count = 0
        self.branch = None
        
        if config.MODEL.TYPE == "Lidar":
            self.lidar_data_aug = Augment_Point_Data(config.TRAINING.BATCH.FRAME_TRANSFORM, is_train)
        elif config.MODEL.TYPE == "LiSPH":
            self.lisph_data_aug = Augment_SPH_Data(config.DATA.SPH_PROJ.IMAGE_SIZE,
                                                   config.TRAINING.BATCH.FRAME_TRANSFORM,  
                                                   is_train)
        elif config.MODEL.TYPE == "Image":
            self.image_data_aug = Augment_RGB_Data(config.TRAINING.BATCH.FRAME_TRANSFORM, is_train)
        else:
            raise ValueError("Wrong Model Type, Please Check config/config.py *MODEL* Section")
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        q_tuple = self.get_data()
        return q_tuple

    def get_data(self):
        q_tuple = []

        while(1):
            # if try data_count times still can not found a proper batch, resample
            if self.data_count > (len(self.file_idxs)-self.config.TRAINING.BATCH.BATCH_SIZE):
                self.shuffle_query()
                continue
            batch_keys = self.file_idxs[self.data_count]
            # select the batch with qualified number of positives and negatives for training
            if (len(self.queries[batch_keys]["positives"]) < self.config.TRAINING.BATCH.POSITIVES_PER_QUERY) or (len(self.queries[batch_keys]["negatives"]) < self.config.TRAINING.BATCH.NEGATIVES_PER_QUERY):
                self.data_count+=1
                continue
            q_tuple = self.get_tuple(self.queries[batch_keys], hard_neg=[], other_neg=True)
            self.data_count+=1
            break

        return q_tuple

    def shuffle_query(self):
        random.shuffle(self.file_idxs)
        self.data_count = 0

    def get_tuple(self, dict_value, hard_neg=None, other_neg=False):
        '''
        hard_neg: bool, add hard negative?
        other_neg: bool, add other negatives?(quadra loss fuction required)
        '''
        if hard_neg is None:
            hard_neg = []
        possible_negs = []

        num_pos = self.config.TRAINING.BATCH.POSITIVES_PER_QUERY
        num_neg = self.config.TRAINING.BATCH.NEGATIVES_PER_QUERY
        random.shuffle(dict_value["positives"])
        pos_files = []
        for i in range(num_pos):
            pos_files.append(self.queries[dict_value["positives"][i]]["query"])

        neg_files = []
        neg_indices = []
        if len(hard_neg) == 0:
            random.shuffle(dict_value["negatives"])
            for i in range(num_neg):
                neg_files.append(self.queries[dict_value["negatives"][i]]["query"])
                neg_indices.append(dict_value["negatives"][i])
        else:
            random.shuffle(dict_value["negatives"])
            for i in hard_neg:
                neg_files.append(self.queries[i]["query"])
                neg_indices.append(i)
            j = 0
            while len(neg_files) < num_neg:
                if not dict_value["negatives"][j] in hard_neg:
                    neg_files.append(self.queries[dict_value["negatives"][j]]["query"])
                    neg_indices.append(dict_value["negatives"][j])
                j += 1
        if other_neg:
            # get neighbors of negatives and query
            neighbors = []
            for pos in dict_value["positives"]:
                neighbors.append(pos)
            for neg in neg_indices:
                for pos in self.queries[neg]["positives"]:
                    neighbors.append(pos)
            possible_negs = list(set(self.queries.keys()) - set(neighbors))
            random.shuffle(possible_negs)

        query = self.load_file_func(dict_value["query"])  # Nx3
        query = np.expand_dims(query, axis=0)
        positives = self.load_files_func(pos_files)
        negatives = self.load_files_func(neg_files)

        output = [query, positives, negatives]
        if other_neg:
            neg2 = self.load_file_func(self.queries[possible_negs[0]]["query"])
            neg2 = np.expand_dims(neg2, axis=0)
            output.append(neg2)
        return output

    @abstractmethod
    def load_file_func(self, filename):
        pass

    def load_files_func(self, filenames):
        pcs = []
        for filename in zip(filenames):
            pc = self.load_file_func(filename[0])
            pcs.append(pc)
        pcs = torch.stack(pcs, axis=0)
        return pcs

    @staticmethod
    def get_queries_dict(filename):
        with open(filename, 'rb') as handle:
            queries = pickle.load(handle)
            return queries

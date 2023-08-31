'''
Author: Peng Yin, Shiqi Zhao
train.py
'''

import os
import sys
import argparse
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

from config import cfg
from models import set_lcd_model
from dataloader import make_data_loader
from eval import EvaluationPitts
from utils import setup_logger, log_print

#!============================================================================================#
#! Parameters for Dataset and Network
#!============================================================================================#


def para_args():
    parser = argparse.ArgumentParser(description="Network configurations!")
    parser.add_argument("--config-network", default="config/network/spherevlad.yaml", metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--config-dataset", default="config/dataset/PITT.yaml", metavar="FILE",
                        help="path to config file", type=str)
    args = parser.parse_args()
    return args


def train(config, logger, neptune):

    #! Define Model
    lcd, gpu_conf = set_lcd_model(config, logger, neptune)
    [_, device, gpu_ids] = gpu_conf

    #! Define Dataloader
    train_loader = make_data_loader(config, gpu_ids, is_train=True)
    if config.DATA.DATASET_NAME in ["PITT", "Campus"]:
        eval_loader = make_data_loader(config, gpu_ids, is_train=False)
    else:
        eval_loader = []
    log_print("train batch with {}, eval batch with {}".format(
        len(train_loader), len(eval_loader)), 'g')

    #! Define Tester
    if config.DATA.DATASET_NAME == "PITT":
        tester = EvaluationPitts(config, lcd, device)

    #! Main loop
    prev_time = time.time()
    best_recall = 0
    for epoch in range(lcd.epoch, config.TRAINING.EPOCH+1):

        log_print("Train epoch {}".format(epoch), "g")
        
        #! Do Training
        for i, batch in enumerate(train_loader):

            # * Determine approximate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = config.TRAINING.EPOCH * \
                len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()
            data = [x.to(device, dtype=torch.float) for x in batch]
            loss_lidar = lcd.train_lcd(data)
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [LiDAR loss: %f], ETA: %s"
                % (epoch, config.TRAINING.EPOCH, i, len(train_loader), loss_lidar, time_left)
            )

        #! Do Evaluation
        if config.DATA.DATASET_NAME in ["PITT", "Campus"]:
            for i, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
                data = [x.to(device, dtype=torch.float) for x in batch]
                lcd.eval_lcd(data)
        
        #! Do Test
        if config.DATA.DATASET_NAME in ["PITT", "Campus"]:
            test_stats = []
            for traj_num in config.DATA.TEST_LIST:
                if config.DATA.DATASET_NAME == "PITT":
                    recall, _, _ = tester.get_features_recall(traj_num, 2, 0, 0)
                else:
                    recall, _, _ = tester.infer(traj_num)
                test_stats.append(recall)
            test_stats = np.array(test_stats).sum(axis=0)
            curr_recall = test_stats[0]/test_stats[-1]
            if curr_recall > best_recall:
                lcd.save_checkpoint(666)
            logger.info(f"[Epoch {epoch}/{config.TRAINING.EPOCH}] Recall@1 {curr_recall}")
            log_print(f"[Epoch {epoch}/{config.TRAINING.EPOCH}] Recall@1 {curr_recall}", 'b')
        
        #! Adjust Learning Rate
        lcd.adjust_learning_rate()

        #! Save Model
        if epoch % 2 == 0:
            lcd.save_checkpoint(epoch)


if __name__ == "__main__":
    args = para_args()
    cfg.merge_from_file(args.config_network)
    cfg.merge_from_file(args.config_dataset)
    cfg.TRAINING.IS_TRAIN = True
    logger, neptune, cfg = setup_logger(
        "Loop closure benchmark", cfg, args, is_train=True)
    cfg.freeze()
    train(cfg, logger, neptune)

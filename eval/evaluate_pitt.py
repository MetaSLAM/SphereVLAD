'''
Author: Shiqi Zhao
evaluate_pitt.py
'''
import os
import argparse
import numpy as np
from config import cfg

from models import set_lcd_model
from eval.eval_utils import EvaluationPitts
from utils import log_print


def para_args():
    parser = argparse.ArgumentParser(description="Network configurations!")
    parser.add_argument("--model-path", default="data/results/default_model", metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--dataset", default="PITT", type=str)
    parser.add_argument("--epoch", default="62", type=int)
    parser.add_argument("--noise", default=0, type=int, help="apply noise during test")
    parser.add_argument("--type", default="rot", type=str, choices=["recall", "rot"], help="choose do top recall or rot analysis")
    parser.add_argument("--trans-noise", type=int, required=True)
    parser.add_argument("--rot-noise", type=int, required=True)
    parser.add_argument("--log", type=bool, required=False, default=False)
    args = parser.parse_args()
    return args


def val(config, type, noise, trans_noise, rot_noise, log):
    #! Log
    if log:
        save_dir = 'log/{}/{}/'.format(config.MODEL.NAME, config.DATA.DATASET_NAME)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    #! Define Model
    model, gpu_conf = set_lcd_model(config)
    [_, device, gpu_ids] = gpu_conf
    #! Define Evaluation Class
    valPitt = EvaluationPitts(config, model, device)
    #! Evaluation
    if type == "recall":
        total_recalls = []
        total_time = 0
        for traj_num in range(21, 22):
            recalls, _, running_time = valPitt.get_features_recall(traj_num, trans_noise, rot_noise, noise)
            total_recalls.append(recalls)
            total_time += running_time
        total_recalls = np.array(total_recalls).sum(axis=0)
        log_print('Total Top One Recall is {}'.format(total_recalls[0]/total_recalls[-1]), 'r')
        log_print('Total Running Time is {}'.format(total_time/total_recalls[-1]), 'r')
        if log:
            stats = total_recalls[:-1]/total_recalls[-1]
            save_file = save_dir + 'recall.txt'
            file_tosave = open(save_file, 'w+')
            file_tosave.write('Average recall @1 is: {}\n'.format(stats[0]))
            file_tosave.write('Average recall @1% is: {}\n'.format(stats[-1]))
            file_tosave.write('\n')
            for index, item in enumerate(stats[:-1]):
                file_tosave.write("Average recall @{} is: {}\n".format(index+1, item))
            file_tosave.write('\t\n\t\n')
            file_tosave.close()
    elif type == "rot":
        for trans in [1,2,3]:
            for rot in [30,60,90,120,150,180]:
                total_recalls = []
                total_time = 0
                for traj_num in range(21, 22):
                    recalls, _, running_time = valPitt.get_features_recall(traj_num, trans, rot, noise)
                    total_recalls.append(recalls)
                    total_time += running_time
                total_recalls = np.array(total_recalls).sum(axis=0)
                log_print('Total Top One Recall is {}'.format(total_recalls[0]/total_recalls[-1]), 'r')
                log_print('Total Running Time is {}'.format(total_time/total_recalls[-1]), 'r')
                if log:
                    stats = total_recalls[:-1]/total_recalls[-1]
                    save_file = save_dir + f'rot_{trans}_{rot}.txt'
                    file_tosave = open(save_file, 'w+')
                    file_tosave.write('Average recall @1 is: {}\n'.format(stats[0]))
                    file_tosave.write('Average recall @1% is: {}\n'.format(stats[-1]))
                    file_tosave.write('\n')
                    for index, item in enumerate(stats[:-1]):
                        file_tosave.write("Average recall @{} is: {}\n".format(index+1, item))
                    file_tosave.write('\t\n\t\n')
                    file_tosave.close()


if __name__ == "__main__":
    args = para_args()
    cfg.merge_from_file("{}/DATA.yaml".format(args.model_path))
    cfg.merge_from_file("{}/MODEL.yaml".format(args.model_path))
    cfg.DATA.DATASET_NAME = args.dataset
    cfg.TRAINING.IS_TRAIN = False
    cfg.TRAINING.GPU.IDS = [0] # default to use only one gpu when testing
    cfg.WEIGHT.LOAD_ADDRESS = "{}/pth/model_{}.pth".format(args.model_path, args.epoch)
    cfg.freeze()
    val(cfg, args.type, args.noise, args.trans_noise, args.rot_noise, args.log)
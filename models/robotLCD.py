import sys
sys.path.append('../')
sys.path.append('../../')

import time

import torch
import torch.nn as nn

from models.loop_closure import make_models
from models.loop_closure.losses import make_losses
from utils import log_print


#!=======================================================================#
#!                           Traning                                     #
#!=======================================================================#
class LcdNet(object):
    def __init__(self, config, logger, use_cuda, device, neptune=None, gpu_ids=None):
        super(LcdNet, self).__init__()

        self.neptune = neptune
        self.config = config
        self.logger = logger

        #! Let's define different models
        self.model = make_models(config)
        log_print("Runing {}".format(config.MODEL.NAME), 'r')
        log_print("#params of model: {}".format(sum([x.numel()
                                        for x in self.model.parameters()])), 'b')
        log_print("#trainable params of model: {}".format(sum([x.numel()
                                        for x in self.model.parameters() if x.requires_grad])), 'b')
        
        #! Data Parallel
        if use_cuda:
            if gpu_ids is not None and len(gpu_ids)>1 :
                print("Let's use", len(gpu_ids), "GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
            self.model = self.model.to(device)

        #! Loss Function
        self.criterion = make_losses(config).to(device)
        
        #! Optimizer and Scheduler
        if config.TRAINING.OPTIMIZER.NAME == 'Adam':
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=config.TRAINING.OPTIMIZER.INIT_LEARNING_RATE)
        else:
            raise NotImplementedError(f"Unrecognized Optimizer {config.TRAINING.OPTIMIZER.NAME}")
        
        if config.TRAINING.SCHEDULER.NAME == None:
            self.scheduler = None
        elif config.TRAINING.SCHEDULER.NAME == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                             step_size=config.TRAINING.SCHEDULER.STEP_SIZE, 
                                                             gamma=config.TRAINING.SCHEDULER.GAMMA)
        elif config.TRAINING.SCHEDULER.NAME == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                                  milestones=config.TRAINING.SCHEDULER.MILESTONES, 
                                                                  gamma=config.TRAINING.SCHEDULER.GAMMA)
        else:
            raise NotImplementedError(f"Unrecognized Scheduler {config.TRAINING.SCHEDULER.NAME}")
        
        #! Resume Training & Testing
        if config.TRAINING.IS_TRAIN:
            if config.TRAINING.RESUME:
                self.epoch = self.load_checkpoint(config.WEIGHT.LOAD_ADDRESS, config.TRAINING.RESUME) + 1
            else:
                self.epoch = 1
        else:
            self.load_checkpoint(config.WEIGHT.LOAD_ADDRESS)

    def train_lcd(self, x):
        """[summary]
        Args:
            x ([type]): [description]
        """

        self.model.train()
        data = torch.cat(x, dim=1)
        B = data.shape[0]
        N = data.shape[1]
        lidar_data = data.view(B*N, -1, data.shape[3], data.shape[4])
        self.optimizer.zero_grad()
        feature_lidar = self.model(lidar_data).view(B, N, -1)
        
        loss_lidar, losses = self.criterion(feature_lidar)
        if self.neptune is not None:
            self.neptune['Sphere/training_lidar_loss'].append(loss_lidar.item())
            self.neptune['Sphere/training_lidar_trip'].append(losses[0].item())
            self.neptune['Sphere/training_lidar_secd'].append(losses[1].item())
        loss_lidar.backward()
        self.optimizer.step()
        return loss_lidar.item()

    def eval_lcd(self, x):
        self.model.eval()
        data = torch.cat(x, dim=1)
        B = data.shape[0]
        N = data.shape[1]
        lidar_data = data.view(B*N, -1, data.shape[3], data.shape[4])
        feature_lidar = self.model(lidar_data).view(B, N, -1)
        with torch.no_grad():
            loss, (trip, secd) = self.criterion(feature_lidar)
            if self.neptune is not None:
                self.neptune['Sphere/eval_lidar_loss'].append(loss.item())
                self.neptune['Sphere/eval_lidar_trip'].append(trip.item())
                self.neptune['Sphere/eval_lidar_secd'].append(secd.item())
    
    def adjust_learning_rate(self):
        if self.scheduler != None:
            self.scheduler.step()

    def save_checkpoint(self, epoch):
        state_dict = {"state_dict": self.model.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "epoch": epoch}
        torch.save(state_dict,
                   '{}/pth/model_{}.pth'.format(self.config.OUTPUT.DIR, epoch))
    
    def load_checkpoint(self, weight_path, resume=False):
        checkpoint = torch.load(weight_path)
        # load model parameters
        try:
            model_dict = checkpoint["state_dict"]
        except:
            model_dict = checkpoint # for pointnetvlad
        try:
            self.model.load_state_dict(model_dict)
        except:
            from collections import OrderedDict
            model_dict = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                new_key = key.split('module.')[-1]
                model_dict[new_key] = value
            self.model.load_state_dict(model_dict)
        log_print("Load models from {}!".format(weight_path), 'g')
        # load optimizer parameters
        if resume:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log_print("Load optimizer parameters from {}!".format(weight_path), 'g')
            return checkpoint["epoch"]
        else:
            return


#!=======================================================================#
#!                           Inference                                   #
#!=======================================================================#
    def infer_frame(self, x, t=False):
        if self.config.MODEL.TYPE in ["Lidar", "LiSPH"]:
            return self.infer_lidar(*x, t)
        elif self.config.MODEL.TYPE == "Image":
            return self.infer_image(*x, t)
    
    def infer_image(self, x, t=False):
        self.model.eval()
        x = x.view(1, 3, x.shape[-2], x.shape[-1])
        if t == True:
            t1 = time.time()
        with torch.no_grad():
            x = self.model(x).view(-1)
        if t == True:
            t2 = time.time()
            t_delta = t2-t1
            return x.detach().cpu().numpy(), t_delta
        return x.detach().cpu().numpy()
    
    def infer_lidar(self, x, t=False):
        self.model.eval()
        x = x.view(1, self.config.MODEL.S2CNN.INPUT_CHANNEL_NUM, x.shape[-2], x.shape[-1])
        if t == True:
            t1 = time.time()
        with torch.no_grad():
            x = self.model(x).view(-1)
        if t == True:
            t2 = time.time()
            t_delta = t2-t1
            return x.detach().cpu().numpy(), t_delta
        return x.detach().cpu().numpy()

import numpy as np
import torch

from .robotLCD import LcdNet


def set_lcd_model(cfg, logger=None, neptune=None):
    #* set up GPU usage
    use_cuda = torch.cuda.is_available()
    gpu_ids = []
    if use_cuda:
        gpu_c = torch.cuda.device_count()
        if cfg.TRAINING.GPU.IDS is not None:
            if len(cfg.TRAINING.GPU.IDS) <= gpu_c:
                gpu_ids = cfg.TRAINING.GPU.IDS
            else:
                raise ValueError("Incorrect GPU Index, Please Check!")
        else:
            gpu_ids = np.arange(gpu_c).tolist()
        cuda = "cuda"
    device = torch.device(cuda if use_cuda else 'cpu')
    lcd = LcdNet(cfg, logger, use_cuda, device, neptune, gpu_ids)
    return lcd, [use_cuda, device, gpu_ids]
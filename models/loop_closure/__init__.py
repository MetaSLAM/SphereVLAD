import torch


def make_models(config):
    if config.MODEL.NAME == "SphereVLAD":
        if config.MODEL.TYPE == "LiSPH":
            from .lidar.spherevlad import SphereVLAD
        elif config.MODEL.TYPE == "Image":
            from .visual.spherevlad import SphereVLAD
        model = SphereVLAD(config)
    elif config.MODEL.NAME == "SphereVLAD2":
        from .lidar.spherevlad2 import SphereVLAD2
        model = SphereVLAD2(config)
    else:
        raise ValueError("Wrong Model Name!")
    
    if config.TRAINING.IS_TRAIN and config.WEIGHT.FREEZE_WEIGHT:
        pass
    return model

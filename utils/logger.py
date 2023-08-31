import os
import sys
import logging
from datetime import datetime
import neptune

def setup_logger(name, cfg, args, is_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    if is_train:
        Output_dir = os.path.join(cfg.OUTPUT.DIR, "{}-{}-{}".format(cfg.MODEL.NAME, cfg.DATA.DATASET_NAME, dt_string))
    else:
        Output_dir = os.path.join(cfg.OUTPUT.DIR, "Inference-{}-{}-{}".format(cfg.MODEL.NAME, cfg.DATA.DATASET_NAME, dt_string))

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    if not os.path.exists(Output_dir):
        os.makedirs(Output_dir)
        os.makedirs("{}/img".format(Output_dir))
        os.makedirs("{}/pth".format(Output_dir))

    fh = logging.FileHandler(os.path.join(Output_dir, "log.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
        
    logger.info(args)

    logger.info("Loaded network configuration file {}".format(args.config_network))
    logger.info("Loaded dataset configuration file {}".format(args.config_dataset))

    os.system('cp {} {}/MODEL.yaml'.format(args.config_network, Output_dir))
    os.system('cp {} {}/DATA.yaml'.format(args.config_dataset, Output_dir))

    cfg.OUTPUT.DIR = Output_dir

    # neptune_run = neptune.init_run(project=cfg.PROJECT_NAME,
    #                  api_token=cfg.API_TOKEN)
    # params = {"Learning_rate": cfg.TRAINING.OPTIMIZER.INIT_LEARNING_RATE,
    #           "Optimizer": cfg.TRAINING.OPTIMIZER.NAME,
    #           "Scheduler": cfg.TRAINING.SCHEDULER.NAME,
    #           "Positive_Dis": cfg.DATA.POSITIVES_RADIUS,
    #           "Negtive_Dis": cfg.DATA.NEGATIVES_RADIUS,
    #           "Loss": cfg.LOSS.NAME,
    #           "Loss_Margin0": cfg.LOSS.MARGIN0,
    #           "Loss_Margin1": cfg.LOSS.MARGIN1,
    #           "Batch_Size": cfg.TRAINING.BATCH.BATCH_SIZE}
    # neptune_run["parameters"] = params
    neptune_run = None

    return logger, neptune_run, cfg

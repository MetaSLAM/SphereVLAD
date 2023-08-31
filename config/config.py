import numpy as np
from yacs.config import CfgNode as CN

_C = CN()

#! -----------------------------------------------------------------------------
#! NEPTUNE
#! -----------------------------------------------------------------------------
_C.PROJECT_NAME = ''
_C.API_TOKEN = ''

#! -----------------------------------------------------------------------------
#! INPUT
#! -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.DATASET_NAME = "" 
_C.DATA.BENCHMARK_DIR = "" # The parent folder of dataset
_C.DATA.DATASET_TYPE = "baseline" # For Oxford Dataset
# Projection for LiDAR point cloud
# Spherical projection
_C.DATA.SPH_PROJ = CN()
_C.DATA.SPH_PROJ.RADIUS = 50.0 # Radius of ball query in submap generation
_C.DATA.SPH_PROJ.IMAGE_SIZE = [64, 64] # Define the size of generated panorama
_C.DATA.SPH_PROJ.FOV = [-90, 90] # Define the fov when genenrating panorama
_C.DATA.SPH_PROJ.VIS_THRESH = 3.0 # Not used for now
_C.DATA.SPH_PROJ.VIS_RADIUS = 3 # Not used for now
_C.DATA.SPH_PROJ.OCCLUSION = 16  # Not used for now
# BEV projection
_C.DATA.BEV_PROJ = CN()
_C.DATA.BEV_PROJ.IMAGE_SIZE = [64, 64] # Define the size of generated BEV
_C.DATA.BEV_PROJ.Z_RANGE = [-100, 100] # Confine the points used to generate BEV
_C.DATA.BEV_PROJ.MAX_DIS = 30 # The x and y max range of the point cloud, eg: a 
                              # point cloud x in [-100,100], this value will be 
                              # 100
# Image
_C.DATA.SPH_IM = CN()
_C.DATA.SPH_IM.IMAGE_SIZE = [224, 224]
_C.DATA.IM = CN()
_C.DATA.IM.IMAGE_SIZE = [224, 224]
# Distance settings for positive and negative frame
_C.DATA.POSITIVES_RADIUS = 10.0
_C.DATA.NEGATIVES_RADIUS = 50.0
_C.DATA.TRAJ_RADIUS = 5.0e+3
# Data spliting
_C.DATA.TRAIN_LIST = []
_C.DATA.VAL_LIST = []
_C.DATA.TEST_LIST = []
# Pickles
_C.DATA.TRAIN_PICKLE = 'train.pickle'
_C.DATA.VAL_PICKLE = 'val.pickle'

#! -----------------------------------------------------------------------------
#! DATASET
#! -----------------------------------------------------------------------------
# Alita Campus dataset
_C.DATA.ALITA_CAMPUS = CN()
_C.DATA.ALITA_CAMPUS.DIRECTION = None
_C.DATA.ALITA_CAMPUS.COLLECTED_TIME = None

#! -----------------------------------------------------------------------------
#! MODEL
#! -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.TYPE = "Lidar" # "Lidar", "LiSPH", "Image"

# S2CNN
_C.MODEL.S2CNN = CN()
_C.MODEL.S2CNN.INPUT_CHANNEL_NUM = 1

# NetVLAD
_C.MODEL.NETVLAD = CN()
_C.MODEL.NETVLAD.ENCODER = 'vgg16'
_C.MODEL.NETVLAD.CLUSTER_NUM = 64
_C.MODEL.NETVLAD.FEATURE_DIM = 128
_C.MODEL.NETVLAD.NORMALIZE_INPUT = True
_C.MODEL.NETVLAD.OUTPUT_DIM = None
_C.MODEL.NETVLAD.GATE = False

# PointNetVLAD
_C.MODEL.POINTNETVLAD = CN()
_C.MODEL.POINTNETVLAD.MAX_SAMPLES = 4096
_C.MODEL.POINTNETVLAD.GLOBAL_FEAT = True
_C.MODEL.POINTNETVLAD.FEATURE_TRANSFORM = True
_C.MODEL.POINTNETVLAD.MAX_POOL = False
_C.MODEL.POINTNETVLAD.CLUSTER_SIZE = 64
_C.MODEL.POINTNETVLAD.OUTPUT_DIM = 256
_C.MODEL.POINTNETVLAD.CONTEXT_GATING = True
_C.MODEL.POINTNETVLAD.ADD_BATCH_NORM = True

# AutoMerge
_C.MODEL.AUTOMERGE = CN()
_C.MODEL.AUTOMERGE.FEATURE_SIZE = 6
_C.MODEL.AUTOMERGE.SELF_ATTEN = 'add'
_C.MODEL.AUTOMERGE.CROSS_ATTEN = 'reweight'

#! -----------------------------------------------------------------------------
#! LOSS
#! -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.NAME = ""
_C.LOSS.MARGIN0 = 0.5
_C.LOSS.MARGIN1 = 0.2
_C.LOSS.MARGIN2 = 0.3

#! -----------------------------------------------------------------------------
#! TRAINING CONFIGURATION
#! -----------------------------------------------------------------------------
_C.TRAINING = CN()
# Training or Test
_C.TRAINING.IS_TRAIN = False
# Number of workers
_C.TRAINING.NUM_WORKERS = 8 # Number of workers when training
# Batch
_C.TRAINING.BATCH = CN()
_C.TRAINING.BATCH.BATCH_SIZE = 2 # Batch size per gpu
# Data Augment
_C.TRAINING.BATCH.FRAME_TRANSFORM = 0 # several mode
_C.TRAINING.BATCH.BATCH_TRANSFORM = False # Data argmentation for the whole batch
# Number of positive and negative in a batch
_C.TRAINING.BATCH.POSITIVES_PER_QUERY = 2
_C.TRAINING.BATCH.NEGATIVES_PER_QUERY = 18
# Optimizer
_C.TRAINING.OPTIMIZER = CN()
_C.TRAINING.OPTIMIZER.NAME = "Adam"
_C.TRAINING.OPTIMIZER.INIT_LEARNING_RATE = 1e-4
_C.TRAINING.OPTIMIZER.WEIGHT_DECAY = 0.001 # Not used for now
# Scheduler
_C.TRAINING.SCHEDULER = CN()
_C.TRAINING.SCHEDULER.NAME = None
_C.TRAINING.SCHEDULER.MILESTONES = [10] # For MultiStepLR
_C.TRAINING.SCHEDULER.STEP_SIZE = None # For StepLR
_C.TRAINING.SCHEDULER.GAMMA = 0.1 # For StepLR & MultiStepLR
# GPU Index for Parallel Training
_C.TRAINING.GPU = CN()
_C.TRAINING.GPU.IDS = None # Choose to use which gpus
# Resume
_C.TRAINING.RESUME = False
# Epoch
_C.TRAINING.EPOCH = 20

#! -----------------------------------------------------------------------------
#! TRAINED WEIGHT
#! -----------------------------------------------------------------------------
_C.WEIGHT = CN()
_C.WEIGHT.LOAD_ADDRESS = ""
_C.WEIGHT.SAVE_ADDRESS = ""
_C.WEIGHT.LOAD_OPTIMIZER = True
_C.WEIGHT.FREEZE_WEIGHT = False

#! -----------------------------------------------------------------------------
#! OUTPUT
#! -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.DIR = "data/results"

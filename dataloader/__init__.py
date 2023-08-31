from torch.utils.data import DataLoader

from .utils import make_collate_fn
from utils import log_print

def make_data_loader(config, gpu_ids, is_train):
    '''
    Args:
        config: parameter configurations
        gpu_ids: gpu indexes for training
        is_train: training or testing
    Returns:
        dataloader(type:class)
    '''
    if is_train:
        log_print("Using dataset: %s" % config.DATA.DATASET_NAME, "g")
    
    if config.DATA.DATASET_NAME == "PITT":
        from .pittsburgh import PittsburghDataset
        dataloader = PittsburghDataset(config, is_train)
    else:
        raise ValueError(f"Unrecognized Dataset Name {config.DATA.DATASET_NAME}")
    
    #TODO collate functions
    if config.TRAINING.BATCH.BATCH_TRANSFORM:
        collate_fn = make_collate_fn(config)
    else:
        collate_fn = None
    # collate_fn=collate_fn,
    
    loader = DataLoader(dataloader,
                        batch_size=config.TRAINING.BATCH.BATCH_SIZE*len(gpu_ids),
                        num_workers=config.TRAINING.NUM_WORKERS if is_train else 4,
                        pin_memory=True,
                        shuffle=True if is_train else False,
                        drop_last=True)
    return loader
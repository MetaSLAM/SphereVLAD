def make_collate_fn(config):
    """Generate Collate_functions

    Args:
        config (class): parameter configurations
    """
    def collate_fn(batch):
        """Collate_functions

        Args:
            batch (list): input batch for training
        """
        if config.TRAINING.IS_TRAIN and config.TRAINING.BATCH.BATCH_TRANSFORM:
            pass
        return batch
    return collate_fn
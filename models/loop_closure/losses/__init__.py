from .lazy_quadruplet_loss import LazyTripletLoss, LazyQuadrupletLoss

def make_losses(config):
    if config.LOSS.NAME == "LazyQuadrupletLoss":
        loss = LazyQuadrupletLoss(
                margin_dis=config.LOSS.MARGIN0, 
                margin_sec=config.LOSS.MARGIN1)
    elif config.LOSS.NAME == "LazyTripletLoss":
        loss = LazyTripletLoss(margin=config.LOSS.MARGIN0)
    else:
        raise NotImplementedError(f"Unrecognized Loss Function {config.LOSS.NAME}!")
    return loss
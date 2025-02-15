from configs.default_configs import get_default_configs


def get_config():
    config = get_default_configs()

    config.method = "TPT"

    # training
    training = config.training
    training.selection_p = 0.1 # confidence selection percentile
    training.tta_steps = 1 

    # inference
    inference = config.inference
    inference.batch_size = 64

    optim = config.optim
    optim.optimizer = "AdamW" # SGD, Adam, AdamW
    optim.lr = 5e-3

    return config

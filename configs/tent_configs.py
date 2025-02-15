from configs.default_configs import get_default_configs


def get_config():
    config = get_default_configs()

    config.method = "Tent"

    # training
    training = config.training

    # inference
    inference = config.inference
    inference.batch_size = 64

    optim = config.optim
    optim.lr = 0.0001 # 0.001
    optim.weight_decay = 0
    optim.optimizer = "Adam" # SGD, Adam
    optim.beta1 = 0.9

    return config
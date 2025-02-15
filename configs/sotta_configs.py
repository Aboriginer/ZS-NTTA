from configs.default_configs import get_default_configs


def get_config():
    config = get_default_configs()

    config.method = "SoTTA"

    # training
    training = config.training

    # inference
    inference = config.inference
    inference.use_learned_stats = True
    inference.bn_momentum = 0.2
    inference.update_every_x = 64
    inference.high_threshold = 0.5
    inference.memory_size = 64
    
    inference.batch_size = 1

    inference.HUS_batch_size = 128
    inference.hloss_temperature = 1.0

    optim = config.optim
    optim.lr = 0.0001 # performance is poor at 0.001
    optim.weight_decay = 0.0005

    return config
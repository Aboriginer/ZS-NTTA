import ml_collections
from configs.default_configs import get_default_configs


def get_config():
    config = get_default_configs()

    config.method = "TDA"

    # training
    training = config.training

    # inference
    inference = config.inference
    config.positive = positive = ml_collections.ConfigDict()
    config.negative = negative = ml_collections.ConfigDict()
    inference.batch_size = 1

    positive.enabled = True
    positive.shot_capacity = 3
    positive.alpha = 2.0
    positive.beta = 5.0 
    
    negative.enabled = True
    negative.shot_capacity = 2
    negative.alpha = 0.117
    negative.beta = 1.0
    negative.entropy_threshold_lower = 0.2
    negative.entropy_threshold_upper = 0.5 
    negative.mask_threshold_lower = 0.03
    negative.mask_threshold_upper = 1.0

    return config
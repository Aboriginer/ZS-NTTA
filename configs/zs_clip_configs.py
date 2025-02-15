from configs.default_configs import get_default_configs


def get_config():
    config = get_default_configs()

    config.method = "ZS-CLIP"

    inference = config.inference
    inference.batch_size = 64

    return config

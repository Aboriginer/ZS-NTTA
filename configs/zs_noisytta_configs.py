from configs.default_configs import get_default_configs


def get_config():
    config = get_default_configs()

    config.method = "ZS-NTTA"

    # training
    training = config.training
    training.save_ckpt = True
    training.gaussian_sampling = False

    # inference
    inference = config.inference
    inference.ttda_queue_length = 64
    inference.top = 1.
    inference.update_classifier = 'None'

    inference.inject_noise_type = 'gaussian' # gaussian, uniform, salt_and_pepper, poisson
    inference.using_ttda_step = 10
    inference.gaussian_rate = 0.125
    inference.batch_size = 128

    optim = config.optim
    optim.classifier_lr = 0.0005 # 0.001
    optim.lr = 0.0005 # 0.001
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    
    return config
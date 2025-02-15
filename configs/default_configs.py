import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    
    # training
    config.training = training = ml_collections.ConfigDict()

    # inference
    config.inference = inference = ml_collections.ConfigDict()
    inference.queue_length = 512
    inference.threshold_type = "adaptive" # adaptive, fixed
    if inference.threshold_type == "fixed":
        inference.fixed_threshold = 0.5
    inference.top = 1. # 0.5

    # data
    config.data = data = ml_collections.ConfigDict()
    data.path = "/data2/chentao/data/check_data"
    # 1. -> 1./(1.+1.)=50%, 0.33 -> 0.33/(1.+0.33)=25%, 3. -> 3/(1.+3.)=75%
    data.OOD_ratio = 1.
    data.workers = 4

    # model
    config.model = model = ml_collections.ConfigDict()
    model.arch = "ViT-B/16" # ViT-B/16, RN50, ViT-L/14
    model.resolution = 224 # CLIP image resolution
    model.n_ctx = 4
    model.ctx_init = "a_photo_of_a"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()

    # logs
    config.logs = logs = ml_collections.ConfigDict()
    logs.project = 'zeroshot_noisyTTDA'
    logs.path = 'results/'
    logs.visualization = 'data_analysis/score'
    logs.conf_path = 'data_analysis/conf_pkl'
    logs.grads_path = 'data_analysis/grads_pkl'
    
    logs.experiment_group = 'main_results'


    config.anlysis_mode = 'normal_mode' # for Table 1 in our paper: normal_mode, all_gt_mode (unavailable), all_update_mode
    config.seed = 0
    config.print_freq = 200
    config.device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    return config
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .zs_clip import ZeroShotCLIP
from utils.utils import *


class Tent(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(Tent, self).__init__(*args, **kwargs)

        for name, module in self.model.named_modules():
            if "image_encoder" in name:
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
                if isinstance(module, nn.BatchNorm2d):
                    module.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
        
        self.writer = SummaryWriter(f'data_analysis/tensorboard/{self.args.logs.experiment_group}/{self.args.method}/{self.args.logs.experiment_id}')
        params = collect_params(self.model, self.args)
        self.optimizer = get_optimizer(self.args, params, lr=self.args.optim.lr)

    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        # assert image.shape[0] != 1 # tent: batch size > 1
        unseen_mask = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)
        self.model.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()

            output = self.model.inference_enable_grad(image)

            if self.args.anlysis_mode == "all_gt_mode":
                selected_output = output[target != self.args.class_num]
            elif self.args.anlysis_mode == "normal_mode":
                selected_output = output[~unseen_mask]
            elif self.args.anlysis_mode == "all_update_mode":
                selected_output = output
            else:
                raise NotImplementedError

            loss = softmax_entropy(selected_output).mean(0)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('TTDA Loss', loss.item(), step)

        return unseen_mask


def collect_params(model, args):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if "image_encoder" in nm:
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)

            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)

    return params
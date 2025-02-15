import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import numpy as np
from copy import deepcopy

from utils.utils import *
from clip.get_model import get_model
from data.dataset_class_names import get_classnames


class ZeroShotCLIP:
    def __init__(self, args):
        self.args = args
        self.model = get_model(self.args)
        self.model_state = None
            
        for param in self.model.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        print("=> Model created: visual backbone {}".format(self.args.model.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            assert self.args.gpu is not None
            torch.cuda.set_device(args.gpu)
            self.model = self.model.cuda(args.gpu)

        print('=> Using native Torch AMP. Training in mixed precision.')

        cudnn.benchmark = True

        classnames = get_classnames(self.args.data.test_set)
        print(f'=> loaded {self.args.data.test_set} classname')

        self.model.reset_classnames(classnames, self.args.model.arch)

        self.os_training_queue = []
        self.os_inference_queue = []

    def setup(self, images, target):
        self.model.eval()

    @torch.no_grad()
    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        assert clip_output.dim() == 2
        logits = F.softmax(clip_output, dim=1)
        ood_score, _ = logits.max(1) # [bs] 
        ood_score = 1 - ood_score
        self.os_inference_queue.extend(ood_score.detach().cpu().tolist())
        self.os_inference_queue = self.os_inference_queue[-self.args.inference.queue_length:]

        if self.args.inference.threshold_type == 'adaptive':
            threshold_range = np.arange(0, 1, 0.01)
            criterias = [compute_os_variance(np.array(self.os_inference_queue), th) for th in threshold_range]
            best_threshold = threshold_range[np.argmin(criterias)]
        else:
            best_threshold = self.args.inference.fixed_threshold
        print(best_threshold)
        unseen_mask = (ood_score > best_threshold)

        return unseen_mask

    @torch.no_grad()
    def get_output(self, image):
        output, image_feature_raw = self.model(image)
        
        return output, image_feature_raw
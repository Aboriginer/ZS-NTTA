import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from .zs_clip import ZeroShotCLIP
from utils.utils import *
from copy import deepcopy


class TPT(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(TPT, self).__init__(*args, **kwargs)

        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad_(True)
        self.os_training_queue = []
        trainable_param = self.model.prompt_learner.parameters()
        self.optimizer = get_optimizer(self.args, trainable_param, self.args.optim.lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    def setup(self, images, target):
        self.model.eval()
        images = torch.cat(images, dim=0)
        assert images.shape[0] != 1 # tpt: batch size > 1 (data augmentation)

        # reset the tunable prompt to its initial state
        if self.args.training.tta_steps > 0:
            with torch.no_grad():
                self.model.reset()
        self.optimizer.load_state_dict(self.optim_state)
        self._test_time_tuning(images, target)

    def _test_time_tuning(self, inputs, target):        
        selected_idx = None
        update_flag = True
        first_iteration = True
        for j in range(self.args.training.tta_steps):
            with torch.cuda.amp.autocast():
                output, _ = self.model(inputs) 

                if self.args.data.OOD_set != 'None' and first_iteration:
                    first_iteration = False
                    logits = F.softmax(output, dim=1)
                    logit_input0 = logits[0]
                    logit_input0 = torch.unsqueeze(logit_input0, 0)

                    ood_score, _ = logit_input0.max(1)
                    ood_score = 1 - ood_score
                    self.os_training_queue.extend(ood_score.detach().cpu().tolist())
                    self.os_training_queue = self.os_training_queue[-self.args.inference.queue_length:]

                    if self.args.inference.threshold_type == 'adaptive':
                        threshold_range = np.arange(0, 1, 0.01)
                        criterias = [compute_os_variance(np.array(self.os_training_queue), th) for th in threshold_range]
                        best_threshold = threshold_range[np.argmin(criterias)]
                    else:
                        best_threshold = self.args.inference.fixed_threshold

                    print('TPT, id:', self.args.data.test_set, 'ood:', self.args.data.OOD_set, best_threshold)
                    if self.args.anlysis_mode == "all_gt_mode":
                        if target == self.args.class_num:
                            update_flag = False
                    elif self.args.anlysis_mode == "normal_mode":
                        if ood_score >= best_threshold:
                            update_flag = False
                    elif self.args.anlysis_mode == "all_update_mode":
                        update_flag = True
                    else:
                        raise NotImplementedError

                if update_flag:
                    if selected_idx is not None:
                        output = output[selected_idx]
                    else:
                        output, selected_idx = select_confident_samples(output, self.args.training.selection_p)

                    assert len(output) > 0
                    loss = avg_entropy(output)
            
            if update_flag:
                self.optimizer.zero_grad()
                # compute gradient and do SGD step
                self.scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.step(self.optimizer)
                self.scaler.update()
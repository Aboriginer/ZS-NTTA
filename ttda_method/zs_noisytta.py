import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from collections import deque
import copy

from .zs_clip import ZeroShotCLIP
from clip.classifier import OODDetector
from utils.utils import *


class ZeroShotNTTA(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(ZeroShotNTTA, self).__init__(*args, **kwargs)

        if self.args.model.arch == "ViT-L/14":
            feat_dim = 768
        elif self.args.model.arch == "ViT-B/16":
            feat_dim = 512
        else:
            feat_dim = 1024
        self.ood_net = OODDetector(feat_dim).to(self.model.image_encoder.conv1.weight.device)
        self.criterion = nn.CrossEntropyLoss() 

        self.optimizer = get_optimizer(self.args, self.ood_net.parameters(), lr=self.args.optim.lr)
        self.os_detector_queue = []
        self.loss_history = []

        queue_length = self.args.inference.ttda_queue_length
        self.queues = {
            'ood_detector_out_queue': deque(maxlen=queue_length),
            'unseen_mask_queue': deque(maxlen=queue_length),
            'target_queue': deque(maxlen=queue_length),
            'clip_output_queue': deque(maxlen=queue_length)
        }
        self.ttda_queue = []

    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        unseen_mask = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)

        self.model.eval()
        self.ood_net.train()

        with torch.enable_grad():   
            if self.args.inference.batch_size == 1:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.ttda_queue.extend(ood_detector_out)
                self.queues['ood_detector_out_queue'].append(ood_detector_out)
                self.queues['unseen_mask_queue'].append(unseen_mask)
                self.queues['target_queue'].append(target)
                self.queues['clip_output_queue'].append(clip_output)
                if step != 0 and step % self.args.inference.ttda_queue_length == 0:                    
                    batch_ood_detector_out = torch.stack(list(self.queues['ood_detector_out_queue']), dim=0).squeeze(1)
                    batch_unseen_mask = torch.stack(list(self.queues['unseen_mask_queue']), dim=0).squeeze(1)
                    batch_target = torch.stack(list(self.queues['target_queue']), dim=0).squeeze(1)
                    batch_clip = torch.stack(list(self.queues['clip_output_queue']), dim=0).squeeze(1)

                    self.update_detector(batch_ood_detector_out, batch_unseen_mask, batch_target, step, batch_clip)
            else:
                ood_detector_out = self.ood_net(image_feature_raw)
                self.update_detector(ood_detector_out, unseen_mask, target, step, clip_output)

        if self.args.inference.batch_size == 1:
            ttda_queue_length = self.args.inference.ttda_queue_length
        else:
            ttda_queue_length = self.args.inference.batch_size

        if step * self.args.inference.batch_size > self.args.inference.using_ttda_step * ttda_queue_length:
            predict = F.softmax(ood_detector_out, 1) # [bs, 2]

            ood_score = predict[:, 0]
            self.os_detector_queue.extend(ood_score.detach().cpu().tolist())
            self.os_detector_queue = self.os_detector_queue[-self.args.inference.queue_length:]
            if self.args.inference.threshold_type == 'adaptive':
                threshold_range = np.arange(0, 1, 0.01)
                criterias = [compute_os_variance(np.array(self.os_detector_queue), th) for th in threshold_range]
                best_threshold = threshold_range[np.argmin(criterias)]
            else:
                best_threshold = self.args.inference.fixed_threshold
            print(best_threshold)
            unseen_mask = (ood_score > best_threshold)

            conf = predict[:, 1]
            return unseen_mask, conf
        
        return unseen_mask
    
    
    def update_detector(self, ood_detector_out, unseen_mask, target, step, clip_output=None):
        logit = F.softmax(clip_output, dim=1)
        conf, _ = logit.max(1) # [bs] 
        
        # NOTE that we don't use ID/OOD samples' target here, we just use Gaussian noise samples' target
        # Artificially added Gaussian noise samples: target = -1000, we can obtain the target of these samples
        unseen_mask[target == -1000] = True

        select_ood_indices = torch.where(unseen_mask)[0]
        select_id_indices = torch.where(~unseen_mask)[0]
        selected_indices = torch.cat([select_id_indices, select_ood_indices])
        selected_labels = torch.cat([torch.ones(len(select_id_indices)), torch.zeros(len(select_ood_indices))]).cuda()
      
        loss = self.criterion(ood_detector_out[selected_indices], selected_labels.long())

        self.optimizer.zero_grad()
        loss.backward()
        self.loss_history.append(loss.item())
        self.optimizer.step()
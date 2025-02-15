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
import operator
import math


class TDA(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(TDA, self).__init__(*args, **kwargs)
        self.pos_cache, self.neg_cache, self.accuracies = {}, {}, []        
        # Unpack all hyperparameters
        self.pos_enabled, self.neg_enabled = self.args.positive.enabled, self.args.negative.enabled
        
    
    def run_test_tda(self, image_feature_raw, clip_logits):
        with torch.no_grad():
            
            image_features = image_feature_raw / image_feature_raw.norm(dim=-1, keepdim=True)
            clip_weights = self.model.get_text_features().t()
            if image_features.size(0) > 1:
                batch_entropy = softmax_entropy(clip_logits)
                selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
                output = clip_logits[selected_idx]
                image_features = image_features[selected_idx].mean(0).unsqueeze(0)
                clip_logits = output.mean(0).unsqueeze(0)

                loss = avg_entropy(output)
                prob_map = output.softmax(1).mean(0).unsqueeze(0)
                pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
            else:
                loss = softmax_entropy(clip_logits)
                prob_map = clip_logits.softmax(1)
                pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

            #Test-time adaptation
            # image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
            prop_entropy = get_entropy(loss, clip_weights)

            if self.pos_enabled:
                update_cache(self.pos_cache, pred, [image_features, loss], self.args.positive.shot_capacity)

            if self.neg_enabled and self.args.negative.entropy_threshold_lower < prop_entropy < self.args.negative.entropy_threshold_upper:
                update_cache(self.neg_cache, pred, [image_features, loss, prob_map], self.args.negative.shot_capacity, True)

            final_logits = clip_logits.clone()
            if self.pos_enabled and self.pos_cache:
                final_logits += compute_cache_logits(image_features, self.pos_cache, self.args.positive.alpha, self.args.positive.beta, clip_weights)
            if self.neg_enabled and self.neg_cache:
                final_logits -= compute_cache_logits(image_features, self.neg_cache, self.args.negative.alpha, self.args.negative.beta, clip_weights, (self.args.negative.mask_threshold_lower, self.args.negative.mask_threshold_upper))

            return final_logits



def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .zs_clip import ZeroShotCLIP
from utils.utils import *


class SoTTA(ZeroShotCLIP):
    def __init__(self, *args, **kwargs):
        super(SoTTA, self).__init__(*args, **kwargs)

        params, _ = sam_collect_params(self.model, freeze_top=True)
        self.optimizer = SAM(params, torch.optim.Adam, rho=0.05, lr=self.args.optim.lr,
                                weight_decay=self.args.optim.weight_decay)

        for param in self.model.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for name, module in self.model.named_modules():
            if "image_encoder" in name:
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                    if self.args.inference.use_learned_stats:
                        module.track_running_stats = True
                        module.momentum = self.args.inference.bn_momentum
                    else:
                        # With below, this module always uses the test batch statistics (no momentum)
                        module.track_running_stats = False
                        module.running_mean = None
                        module.running_var = None

                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

                elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d): 
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

                elif isinstance(module, nn.LayerNorm):  
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

        self.mem = HUS(args=self.args, capacity=self.args.inference.memory_size, threshold=self.args.inference.high_threshold)
        self.mem_state = self.mem.save_state_dict()


    def get_unseen_mask(self, clip_output, image, image_feature_raw, step, target):
        assert image.shape[0] == 1 # sotta: batch size = 1
        unseen_mask = super().get_unseen_mask(clip_output, image, image_feature_raw, step, target)
        dls = torch.zeros(1) # domain_labels in SoTTA, not used here

        current_sample = torch.squeeze(image, 0), target, dls

        with torch.no_grad():
            self.model.eval()
            f, c, d = current_sample[0].cuda(), current_sample[1].cuda(), current_sample[2].cuda()
            logit = F.softmax(clip_output, dim=1)
            pseudo_cls = logit.max(1, keepdim=False)[1][0].cpu().numpy() # TODO
            pseudo_conf = logit.max(1, keepdim=False)[0][0].cpu().numpy()

            if self.args.anlysis_mode == "all_gt_mode":
                if target != self.args.class_num:
                    self.mem.add_instance([f, pseudo_cls, d, pseudo_conf])
            elif self.args.anlysis_mode == "normal_mode":
                if not unseen_mask:
                    self.mem.add_instance([f, pseudo_cls, d, pseudo_conf])
            elif self.args.anlysis_mode == "all_update_mode":
                self.mem.add_instance([f, pseudo_cls, d, pseudo_conf])
            else:
                raise NotImplementedError

        if step % self.args.inference.update_every_x != 0 or self.args.inference.update_every_x > step:  # train only when enough samples are collected
            return unseen_mask

        # setup models
        self.model.train()
        with torch.enable_grad():

            feats, _, _ = self.mem.get_memory()

            if len(feats) == 0:
                return unseen_mask

            feats = torch.stack(feats)
            dataset = torch.utils.data.TensorDataset(feats)
            data_loader = DataLoader(dataset, batch_size=self.args.inference.HUS_batch_size,
                                    shuffle=True, drop_last=False, pin_memory=False)

            entropy_loss = HLoss(self.args.inference.hloss_temperature)

            for batch_idx, (feats,) in enumerate(data_loader):
                self.step(loss_fn=entropy_loss, feats=feats)

        return unseen_mask


    def step(self, loss_fn, feats=None):
        assert (feats is not None)

        self.model.train()
        feats = feats.cuda()
        preds_of_data = self.model.inference_enable_grad(feats)

        loss_first = loss_fn(preds_of_data)

        self.optimizer.zero_grad()

        loss_first.backward()

        if not isinstance(self.optimizer, SAM):
            self.optimizer.step()
        else:
            # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
            self.optimizer.first_step(zero_grad=True)

            preds_of_data = self.model.inference_enable_grad(feats)

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second = loss_fn(preds_of_data)

            loss_second.backward()

            self.optimizer.second_step(zero_grad=True)


def sam_collect_params(model, freeze_top=False):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if freeze_top:
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # print(self.base_optimizer, self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class HUS:
    def __init__(self, args, capacity, threshold=None):
        self.args = args
        self.data = [[[], [], [], []] for _ in range(self.args.class_num)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * self.args.class_num
        self.marker = [''] * self.args.class_num
        self.capacity = capacity
        self.threshold = threshold

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [[l[:] for l in ls] for ls in state_dict['data']]
        self.counter = state_dict['counter'][:]
        self.marker = state_dict['marker'][:]
        self.capacity = state_dict['capacity']
        self.threshold = state_dict['threshold']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [[l[:] for l in ls] for ls in self.data]
        dic['counter'] = self.counter[:]
        dic['marker'] = self.marker[:]
        dic['capacity'] = self.capacity
        dic['threshold'] = self.threshold

        return dic

    def print_class_dist(self):
        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):
        occupancy_per_class = [0] * self.args.class_num
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):
        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.args.class_num
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 4)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[3] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_average_confidence(self):
        conf_list = []
        for i, data_per_cls in enumerate(self.data):
            for confidence in data_per_cls[3]:
                conf_list.append(confidence)
        if len(conf_list) > 0:
            return np.average(conf_list)
        else:
            return 0

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][3])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][3])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], [], []] for _ in range(self.args.class_num)]  # feat, pseudo_cls, domain, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][2].append(0)
            self.data[tgt_idx][3].append(aux[i])


class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.sum(dim=1).mean()

        return b

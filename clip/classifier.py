import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn


class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True, head='linear'):
        super(ResClassifier_MME, self).__init__()
        if head == "linear":
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Sequential(
                    nn.Linear(input_size, input_size, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(input_size, num_classes, bias=False)
                )
            # self.fc = nn.Sequential(nn.Linear(input_size, input_size // 4, bias=False),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(input_size // 4, num_classes, bias=False))
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
        
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)


class OODDetector(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super(OODDetector, self).__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x
    

class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp 
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


# class DINOHead(nn.Module):
#     def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
#                  nlayers=3, hidden_dim=2048, bottleneck_dim=256):
#         super().__init__()
#         nlayers = max(nlayers, 1)
#         if nlayers == 1:
#             self.mlp = nn.Linear(in_dim, bottleneck_dim)
#         elif nlayers != 0:
#             layers = [nn.Linear(in_dim, hidden_dim)]
#             if use_bn:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.GELU())
#             for _ in range(nlayers - 2):
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#                 if use_bn:
#                     layers.append(nn.BatchNorm1d(hidden_dim))
#                 layers.append(nn.GELU())
#             layers.append(nn.Linear(hidden_dim, bottleneck_dim))
#             self.mlp = nn.Sequential(*layers)
#         self.apply(self._init_weights)
#         self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
#         self.last_layer.weight_g.data.fill_(1)
#         if norm_last_layer:
#             self.last_layer.weight_g.requires_grad = False

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x_proj = self.mlp(x)
#         x = nn.functional.normalize(x, dim=-1, p=2)
#         # x = x.detach()
#         logits = self.last_layer(x)
#         logits = logits / 0.1
#         return logits
        

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        logits = logits / 0.1
        return logits
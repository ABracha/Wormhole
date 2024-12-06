import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PartialGeodesicLoss(nn.Module):
    def __init__(self, loss_weight=1.0, w_p = 0.0 ,is_binary = False):
        super().__init__()
        self.loss_weight = loss_weight
        self.is_binary = is_binary
        self.w_p = w_p

    def forward(self, Pyx, Dx, Dy, Ay, mask):
        valid_points = Dy < 4 # remove distances with errors
        losses = dict()
        Ay_05 = Ay**0.5
        if self.loss_weight > 0.0:
            weights = torch.clip(mask/(Dy + 1e-7),0.0,1.0) if not self.is_binary else torch.where(mask > Dy, 1.0 , 0.0).float()
            loss = torch.sum((((torch.abs(Ay_05*(torch.matmul(Pyx,torch.matmul(Dx,Pyx.transpose(-2,-1))) - Dy)*Ay_05.unsqueeze(1))) **2)*weights)[valid_points])

            l_geo = self.loss_weight * loss
            losses['l_geo'] = l_geo
        return losses

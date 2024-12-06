import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return self.loss_weight * torch.mean(loss)



@LOSS_REGISTRY.register()
class PartialOrthoLoss(nn.Module):
    def __init__(self, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super(PartialOrthoLoss, self).__init__()
        self.w_orth = w_orth


    def forward(self, C_fp, evals_full, evals_partial):
        assert C_fp.shape[0] == 1, 'Currently, only support batch size = 1'
        criterion = SquaredFrobeniusLoss()
        C_fp = C_fp[0]
        evals_full, evals_partial = evals_full[0], evals_partial[0]
        losses = dict()
        # compute area ratio between full shape and partial shape r
        r = min((evals_partial < evals_full.max()).sum(), C_fp.shape[0] - 1)
        eye = torch.zeros_like(C_fp)
        eye[torch.arange(0, r + 1), torch.arange(0, r + 1)] = 1.0

        if self.w_orth > 0:
            losses['l_orth'] = self.w_orth * criterion(torch.matmul(C_fp, C_fp.t()), eye)
        return losses

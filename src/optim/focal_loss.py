import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.reduction = "mean"

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs, targets, reduction=self.reduction)

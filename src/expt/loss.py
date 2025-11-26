import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from expt.config import LossConfig
from expt.utils import to_tensor


class FocalLoss(nn.Module):
    """
    Wrapper around torchvision.ops.sigmoid_focal_loss to make it compatible
    with the standard CrossEntropyLoss interface (accepting class indices).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (N, C) tensor of raw scores
            targets: (N,) tensor of class indices (0, 1, ..., C-1)
        """
        num_classes = logits.size(1) if self.num_classes is None else self.num_classes

        # targets: (N) -> (N, C)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        loss = sigmoid_focal_loss(
            inputs=logits,
            targets=targets_one_hot,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss


def create_loss_function(config: LossConfig) -> nn.Module:
    """
    Factory function to create loss function based on config.

    Args:
        config: LossConfig object
        class_weights: Optional tensor of shape (num_classes,) for weighted loss
    """
    loss_name = config.name.lower()
    class_weights = (
        to_tensor(config.class_weights) if config.class_weights is not None else None
    )
    if loss_name == "cross_entropy":
        # CrossEntropyLoss handles softmax internally
        return nn.CrossEntropyLoss(weight=class_weights, reduction=config.reduction)

    elif loss_name == "focal":
        # Handle defaults if config values are None
        alpha = config.alpha if config.alpha is not None else 0.25
        gamma = config.gamma if config.gamma is not None else 2.0

        return FocalLoss(alpha=alpha, gamma=gamma, reduction=config.reduction)

    elif loss_name == "bce":
        # Usually requires sigmoid applied beforehand or BCEWithLogitsLoss
        # Here assuming Multilabel or Binary where logits are passed
        pos_weight = class_weights if class_weights is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=config.reduction)

    else:
        raise NotImplementedError(f"Loss function {loss_name} not implemented.")

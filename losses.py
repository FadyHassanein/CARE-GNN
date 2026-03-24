import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fraud detection.
    Paper: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Focuses training on hard-to-classify examples by down-weighting
    well-classified examples with a modulating factor (1-p_t)^gamma.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        :param alpha: class weights (tensor of size num_classes) or scalar for binary
        :param gamma: focusing parameter, higher gamma = more focus on hard examples
        :param reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([1 - alpha, alpha])
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets)
            focal_weight = focal_weight * at

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedCrossEntropy(nn.Module):
    """
    Weighted cross-entropy loss that automatically computes class weights
    from training labels to handle class imbalance.
    """

    def __init__(self, labels=None, num_classes=2):
        super(WeightedCrossEntropy, self).__init__()
        if labels is not None:
            class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
            weights = len(labels) / (num_classes * class_counts)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.criterion(inputs.to(self.criterion.weight.device), targets.to(self.criterion.weight.device))

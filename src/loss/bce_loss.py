import torch
import torch.nn as nn


class BCEWithLogitsLossWrapper(nn.Module):
    """
    Wrapper over nn.BCEWithLogitsLoss.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Args:
            logits (torch.Tensor): Output logits from LCNN, shape (B, 1) or (B,).
            target (torch.Tensor): Ground truth labels, shape (B, 1) or (B,).
        
        Returns:
            dict: Dictionary with key 'loss'.
        """
        logits = logits.squeeze(-1)
        target = target.squeeze(-1).float()

        loss = self.loss(logits, target)
        
        return {"loss": loss}

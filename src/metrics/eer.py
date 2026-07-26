import numpy as np
import torch
from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # False rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size)
    )
    # False acceptance rates
    far = np.concatenate(
        (
            np.atleast_1d(1),
            nontarget_trial_sums / nontarget_scores.size,
        )
    )

    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    """ 
    Returns equal error rate (EER) and the corresponding threshold.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    eer *= 100
    return eer, thresholds[min_index]


class EERMetric(BaseMetric):
    """Equal Error Rate (EER) metric."""

    def __init__(self, name="EER", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.reset()

    def reset(self):
        self.bonafide_scores = []
        self.other_scores = []

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        """Compute scores from all batches."""
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labels = target.detach().cpu().numpy().reshape(-1)

        self.bonafide_scores.extend(probs[labels == 1.0])
        self.other_scores.extend(probs[labels == 0.0])
        
        return 0.0 

    def compute(self):
        """Compute EER in whole epoch."""
        if len(self.bonafide_scores) == 0 or len(self.other_scores) == 0:
            return 0.0
            
        eer, _ = compute_eer(
            np.array(self.bonafide_scores), 
            np.array(self.other_scores)
        )
        return eer

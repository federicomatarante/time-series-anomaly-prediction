from operator import index
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


def any_predicted_anomaly(preds: Tensor, targets: Tensor, threshold: float):
    pred_exists = (preds.sum(dim=2) >= threshold)
    target_exists = (targets.sum(dim=2) >= threshold)

    tp_mask = (pred_exists & target_exists)
    return tp_mask


class ExistenceOfAnomaly(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        # Use float tensors for better performance
        self.add_state("true_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if len(preds.shape) != 3 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 3D tensors of shape [batch_size, channels, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")

        # Vectorized operations across batch
        pred_exists = (preds.sum(dim=(2)) >= self.threshold)
        target_exists = (targets.sum(dim=(2)) >= self.threshold)

        self.true_positives += (pred_exists & target_exists).float().sum(dim=1).mean()
        self.false_positives += (pred_exists & ~target_exists).float().sum(dim=1).mean()
        self.false_negatives += (~pred_exists & target_exists).float().sum(dim=1).mean()

    def compute(self) -> float:
        epsilon = 1e-7  # Prevent division by zero
        numerator = 2 * self.true_positives
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives + epsilon
        return (numerator / denominator)


class DensityOfAnomalies(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("density_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_density_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape

        tp_mask = any_predicted_anomaly(preds, targets, self.threshold)
        sum_mask = tp_mask.float().sum()  # Number of TP channels

        if sum_mask == 0:
            return

        # Compute density per channel (only for TP channels)
        diff = (targets.sum(dim=2) - preds.sum(dim=2)).abs() / sum_mask  # [B, C]
        ones = torch.ones_like(diff)

        density_per_channel = (ones - diff) * tp_mask  # Zero out non-TP

        self.density_sum += density_per_channel.sum()
        self.total_density_samples += tp_mask.sum()

    def compute(self):
        return self.density_sum / (self.total_density_samples + 1e-7)


class LeadTime(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("lead_time_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape

        tp_mask = any_predicted_anomaly(preds, targets, self.threshold)
        sum_mask = tp_mask.int().sum()  # Number of TP channels

        if sum_mask == 0:
            return

        preds_values = (preds >= self.threshold).int()
        targets_values = (targets >= self.threshold).int()
        first_pred_indices = torch.argmax(preds_values, dim=2)
        first_target_indices = torch.argmax(targets_values, dim=2)
        diff = (first_pred_indices - first_target_indices).abs() / sum_mask  # [B, C]
        ones = torch.ones_like(diff)
        lead_time = (ones - diff) * tp_mask  # Zero out non-TP
        # Accumulate results
        self.lead_time_sum += lead_time.sum()
        self.num_valid += tp_mask.sum()

    def compute(self):
        if self.num_valid == 0:
            return torch.tensor(0.0, device=self.lead_time_sum.device)
        return self.lead_time_sum / self.num_valid


class DiceScore(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valid_channels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        B, C, T = preds.shape

        # TP mask [B, C]
        tp_mask = any_predicted_anomaly(preds, targets, self.threshold)
        if tp_mask.sum() == 0:
            return

        pred_sums = preds.sum(dim=2)
        target_sums = targets.sum(dim=2)
        # Compute Dice only for TP channels
        pred_mask = (pred_sums >= self.threshold)  # [B, C, T]
        target_mask = (target_sums >= self.threshold)

        intersection = (pred_mask * tp_mask & target_mask * tp_mask).sum(dim=-1)  # [B, C]
        pred_sum_elements = (pred_mask * tp_mask).sum(dim=-1)
        target_sum_elements = (target_mask * tp_mask).sum(dim=-1)
        union = pred_sum_elements + target_sum_elements

        dice = (2 * intersection) / (union + 1e-7)  # [B, C]
        # dice = dice * tp_mask  # Zero out non-TP

        self.dice_sum += dice.sum()
        self.valid_channels += B

    def compute(self):
        return self.dice_sum / (self.valid_channels + 1e-7)


def create_tensors_with_some_zeroes(batch_size, channels, window_size):
    # TODO test: channels with all 0
    #   channels with values not exactly 1 but lower
    #   not exact values
    #   There are too many ones for now
    preds = torch.rand(batch_size, channels, window_size)  # Genera tensor casuale con 0 e 1
    # Scegli casualmente il 10% dei canali da azzerare
    num_zero_channels = int(channels * 0.1)
    indices = torch.randperm(channels)[:num_zero_channels]  # Indici dei canali da azzerare

    # Azzeriamo i canali selezionati
    preds[:, indices, :] = 0

    targets = torch.round(preds)  # Crea una copia identica

    return preds, targets

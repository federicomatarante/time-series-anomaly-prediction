import math
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class ExistenceOfAnomaly(Metric):
    """
    Evaluates if model correctly predicts existence of at least one anomaly within prediction range.
    Adapted for batch processing with shape [batch_size, window_size].

    Score ranges from 0 (worst) to 1 (best).

    :param threshold: Value above which a prediction is considered an anomaly
    :param kwargs: Additional arguments passed to parent Metric class
    """

    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update metric states from batch of predictions and targets.

        :param preds: Model predictions [batch_size, window_size]
        :param targets: Ground truth labels [batch_size, window_size]
        :raises ValueError: If inputs have incorrect shapes
        """
        """if len(preds.shape) != 2 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 2D tensors of shape [batch_size, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")"""
        # TODO check shape
        preds = preds.flatten(0, 1)
        targets = targets.flatten(0, 1)
        # Check existence of anomalies in each sequence
        pred_exists = (preds >= self.threshold)
        target_exists = (targets >= self.threshold)

        self.true_positives += torch.sum(pred_exists & target_exists)
        self.false_positives += torch.sum(pred_exists & ~target_exists)
        self.false_negatives += torch.sum(~pred_exists & target_exists)

    def compute(self) -> float:
        """
        Compute F1 score from accumulated statistics.

        :return: F1 score between 0 and 1
        """
        return 2 * self.true_positives / (2 * self.true_positives + self.false_positives + self.false_negatives)

    def reset(self):
        """Reset metric states to initial values."""
        self.true_positives.zero_()
        self.false_positives.zero_()
        self.false_negatives.zero_()


class DensityOfAnomalies(Metric):
    """
    Measures difference between predicted and actual anomaly densities across batches.

    Score ranges from 0 (worst) to 1 (best).

    :param kwargs: Additional arguments passed to parent Metric class
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("cumulative_density", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def reset(self):
        """Reset metric states to initial values."""
        self.cumulative_density.zero_()
        self.total.zero_()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update metric states from batch of predictions and targets.

        :param preds: Model predictions [batch_size, window_size]
        :param targets: Ground truth labels [batch_size, window_size]
        :raises ValueError: If inputs have incorrect shapes
        """
        """if len(preds.shape) != 2 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 2D tensors of shape [batch_size, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")"""
        # TODO check shape

        # Compute density difference for each sequence in batch
        self.cumulative_density += torch.sum(torch.abs(torch.sum(targets - preds, dim=1)))
        self.total += targets.numel()

    def compute(self) -> float:
        """
        Compute density similarity score from accumulated statistics.

        :return: Score between 0 and 1
        """

        return 1 - self.cumulative_density / self.total


class LeadTime(Metric):
    """
    Measures temporal distance between first predicted and first actual anomaly in each sequence.

    Score ranges from 0 (worst) to 1 (best), or inf if no anomalies found.

    :param threshold: Value above which a prediction is considered an anomaly
    :param kwargs: Additional arguments passed to parent Metric class
    """

    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("cumulative_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("valid_sequences", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")

    def reset(self) -> None:
        """Reset metric states to initial values."""
        self.cumulative_distance.zero_()
        self.valid_sequences.zero_()
        self.total_length.zero_()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update metric states from batch of predictions and targets.

        :param preds: Model predictions [batch_size, window_size]
        :param targets: Ground truth labels [batch_size, window_size]
        :raises ValueError: If inputs have incorrect shapes
        """
        """if len(preds.shape) != 2 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 2D tensors of shape [batch_size, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")"""
        # TODO check shape

        batch_size, window_size = preds.shape
        self.total_length += window_size

        for i in range(batch_size):
            pred_indices = torch.nonzero(preds[i] >= self.threshold)
            target_indices = torch.nonzero(targets[i] >= self.threshold)

            if len(pred_indices) > 0 and len(target_indices) > 0:
                # Get first occurrence of anomaly in both sequences
                first_pred = pred_indices[0]
                first_target = target_indices[0]
                self.cumulative_distance += torch.abs(first_pred - first_target).item()
                self.valid_sequences += 1

    def compute(self) -> float:
        """
        Compute lead time score from accumulated statistics.

        :return: Score between 0 and 1, or inf if no valid sequences found
        """
        if self.valid_sequences == 0:
            return math.inf
        return 1 - self.cumulative_distance / (self.valid_sequences * self.total_length)


class DiceScore(Metric):
    """
    Measures overlap between predicted and actual anomalies using Dice coefficient.
    Adapted for batch processing.

    Score ranges from 0 (no overlap) to 1 (perfect overlap).

    :param threshold: Value above which a prediction is considered an anomaly
    :param kwargs: Additional arguments passed to parent Metric class
    """

    def __init__(self, threshold: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("pred_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("target_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("common_predicted_positives", default=torch.tensor(0), dist_reduce_fx="sum")

    def reset(self) -> None:
        """Reset metric states to initial values."""
        self.pred_positives.zero_()
        self.target_positives.zero_()
        self.common_predicted_positives.zero_()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Update metric states from batch of predictions and targets.

        :param preds: Model predictions [batch_size, window_size]
        :param targets: Ground truth labels [batch_size, window_size]
        :raises ValueError: If inputs have incorrect shapes
        """
        """if len(preds.shape) != 2 or preds.shape != targets.shape:
            raise ValueError(
                f"Inputs must be 2D tensors of shape [batch_size, window_size]. "
                f"Got predictions shape {preds.shape} and targets shape {targets.shape}")"""
        # TODO check shape

        pred_mask = preds >= self.threshold
        target_mask = targets >= self.threshold

        self.pred_positives += torch.sum(pred_mask)
        self.target_positives += torch.sum(target_mask)
        self.common_predicted_positives += torch.sum(pred_mask & target_mask)

    def compute(self) -> float:
        """
        Compute Dice score from accumulated statistics.

        :return: Score between 0 and 1
        """
        return 2 * self.common_predicted_positives / (self.pred_positives + self.target_positives)

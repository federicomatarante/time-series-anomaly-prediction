import random
from pathlib import Path
from typing import Any, Tuple, List

import numpy as np
import torch

from src.dataset.msl import MSLDataset


class MSLAnomalyPredictionDataset(MSLDataset):
    """
    This dataset processes telemetry data for anomaly detection by creating sliding windows
    of input sequences and their corresponding anomaly labels.

    See super class for more reference.
    """

    def __init__(self, ds_type: str, window_size: int, horizon_size: int, stride: int):
        super().__init__(ds_type, window_size, horizon_size, stride)

    def _load_dataset(self, anomalies: dict[Path, dict[str, Any]]):
        """Load and process the raw data into sliding windows.

         :param anomalies: dict mapping file paths to metadata containing:
             - num_values: number of values in time series
             - sequences: list of anomaly start/end indices

         Processing steps:
         1. Loads raw numpy arrays and converts to tensors
         2. Creates binary anomaly tensors from metadata
         3. Segments data into overlapping windows using stride
         4. Creates (sequence, anomaly) pairs for each window

         Shapes:
             - sequence_tensor: [seq_len, channels] - Raw input sequence
             - anomalies_tensor: [seq_len, channels] - Binary anomaly indicators
             - sequence_piece: [window_size, channels] - Input window
             - anomaly_piece: [horizon_size, channels] - Target window
         """
        data = []
        # Parsed data and converted to tensors
        for file, meta in anomalies.items():
            sequence = np.load(file)
            num_values = meta['num_values']
            sequence_tensor = torch.from_numpy(sequence)  # Shape [seq_len, channels]
            anomalies_tensor = torch.zeros_like(sequence_tensor)  # Creating anomaly tensor
            anomalies = meta['sequences']  # List of [start_anomaly,end_anomaly] indexes
            for anomaly in anomalies:
                start_anomaly, end_anomaly = anomaly[0], anomaly[1]
                anomalies_tensor[start_anomaly: end_anomaly] = 1
            data.append((num_values, sequence_tensor, anomalies_tensor))

        # Divide it in pieces
        dataset = []
        for num_values, sequence_tensor, anomalies_tensor in data:
            num_pieces = (sequence_tensor.shape[
                              0] - self.window_size - self.horizon_size) // self.stride + 1  # Fixed calculation
            for i in range(num_pieces):
                seq_start = i * self.stride
                seq_end = seq_start + self.window_size
                pred_end = seq_end + self.horizon_size

                # Check if pred_end index is past sequence length
                if pred_end > num_values:
                    break
                # Cut and also converted to shape (channels, seq_len)
                sequence_piece = sequence_tensor[seq_start:seq_end, :].transpose(0, 1).float()
                anomaly_piece = anomalies_tensor[seq_end:pred_end, :].transpose(0, 1).float()

                dataset.append((sequence_piece, anomaly_piece))
        return dataset

    def normalize(self):
        """
        Normalize the dataset using min-max normalization to scale each channel
        to the range [0, 1].
        """
        # Compute the min and max for each channel across the entire dataset
        all_sequences = [seq for seq, _ in self.dataset]  # Extract the sequences (input windows)
        all_sequences = torch.stack(all_sequences)  # Shape: [num_samples, window_size, channels]

        # Compute the min and max per channel (over all samples)
        min_vals, max_vals = self.get_min_max()

        # Normalize each sequence in the dataset
        for i, (sequence, anomaly) in enumerate(self.dataset):
            # Min-Max normalization: (value - min) / (max - min)
            normalized_sequence = (sequence - min_vals) / (
                    max_vals - min_vals + 1e-7)  # Adding small epsilon to avoid div by 0
            self.dataset[i] = (normalized_sequence, anomaly)

    def _get_anomalies(self) -> Tuple[int, List[int]]:
        """
        Identifies and counts the anomalous samples in the dataset.
        :return: A tuple containing the number of anomalies and a list of indices of "not anomalies".
        """
        not_anomalies_indices = []
        num_anomalies = 0
        for i, (_, labels) in enumerate(self.dataset):
            if (labels == 1).any():
                num_anomalies += 1
            else:
                not_anomalies_indices.append(i)

        return num_anomalies, not_anomalies_indices

    @property
    def anomalies_ratio(self) -> float:
        """
        Computes and returns the proportion of anomalous samples in the dataset.
        :return: A float representing the ratio of anomalous samples to total samples in the dataset.
        """
        num_anomalies, _ = self._get_anomalies()
        return num_anomalies / len(self.dataset)

    def get_min_max(self) -> Tuple[float, float]:
        """
        Returns the minimum and maximum values from the sequence data in the dataset.
        :return: A tuple (min_value, max_value)
        """
        min_value = float('inf')
        max_value = float('-inf')

        for sequence, _ in self.dataset:
            # Assuming sequence is a tensor and we are interested in the raw sequence values
            min_value = min(min_value, sequence.min().item())
            max_value = max(max_value, sequence.max().item())

        return min_value, max_value

    def balance(self, ratio: float):
        """
        Balances the dataset by randomly eliminating a quantity of samples without anomalies in the target window.
        :param ratio: goal ratio between samples with anomalies respect to samples without anomalies.
        :raise ValueError: if ratio is not between 0 and 1
        """

        num_anomalies, free_spaces_indices = self._get_anomalies()
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between 0 and 1!")
        current_free_samples = len(self.dataset) - num_anomalies
        num_to_remove = int((current_free_samples - (1 - ratio) * len(self.dataset)) / (1 - (1 - ratio)))
        anomalies_to_remove = random.sample(free_spaces_indices, num_to_remove)
        self.dataset = [elem for i, elem in enumerate(self.dataset) if i not in anomalies_to_remove]

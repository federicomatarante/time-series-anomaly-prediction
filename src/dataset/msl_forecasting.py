from pathlib import Path
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataset.msl import MSLDataset


class MSLForecastingDataset(MSLDataset):
    """
    This dataset processes telemetry data for anomaly detection by creating sliding windows
    of input sequences and their corresponding forecasting values.

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
         4. Creates (sequence, prediction) pairs for each window

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
            data.append((num_values, sequence_tensor))

        # Divide it in pieces
        dataset = []
        for num_values, sequence_tensor in data:
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
                prediction_piece = sequence_tensor[seq_end:pred_end, :].transpose(0, 1).float()

                dataset.append((sequence_piece, prediction_piece))
        return dataset

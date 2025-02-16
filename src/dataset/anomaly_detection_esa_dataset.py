from typing import Union

import numpy as np
import torch

from src.dataset.esa import ESADataset


class AnomalyDetectionESADataset(ESADataset):


    def __init__(self, folder: str, mission: Union[int, str], period: str, ds_type: str, window_size: int = 2,
                 horizon_size: int = 2, stride: int = 1):

        super().__init__(folder, mission, period, ds_type, window_size, horizon_size, stride)

    def __getitem__(self, idx):
        # Calculate start and end indices for the input window
        start_index = self.start_time + self.delta_index * idx * self.stride
        end_index = start_index + self.delta_index * (self.window_size - 1)

        # Calculate indices for the horizon (future) window
        start_horizon_index = end_index + self.delta_index
        end_horizon_index = start_horizon_index + self.delta_index * (self.horizon_size - 1)

        # Get input signals - shape [channels, window_size]
        signals = torch.from_numpy(
            self.channels.loc[start_index:end_index].to_numpy(dtype=np.float32)
        ).transpose(1, 0)  # Transpose to get [channels, window_size]

        if signals.shape[1] < self.window_size:
            filler = torch.zeros([self.n_channels, self.window_size - signals.shape[1]])
            signals = torch.cat([signals, filler], dim=1)

        # Get future timesteps - shape [channels, horizon_size]
        future_signals = torch.from_numpy(
            self.channels.loc[start_horizon_index:end_horizon_index].to_numpy(dtype=np.float32)
        ).transpose(1, 0)  # Transpose to get [channels, horizon_size]

        if future_signals.shape[1] < self.horizon_size:
            filler = torch.zeros([self.n_channels, self.horizon_size - future_signals.shape[1]])
            future_signals = torch.cat([future_signals, filler], dim=1)

        # Get labels - shape [channels, horizon_size]
        labels = torch.from_numpy(
            self.anomalies.loc[start_horizon_index:end_horizon_index].to_numpy(dtype=np.float32)
        ).transpose(1, 0)  # Transpose to get [channels, horizon_size]

        if labels.shape[1] < self.horizon_size:
            filler = torch.zeros([self.n_channels, self.horizon_size - labels.shape[1]])
            labels = torch.cat([labels, filler], dim=1)
        return (
            signals,  # Shape: [channels, window_size]
            future_signals,  # Shape: [channels, horizon_size]
            torch.where(labels == 1.0, 1.0, 0.0)  # Shape: [channels, horizon_size]
        )

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Union
import numpy as np


class ESADataset(Dataset):
    """Dataset class for ESA-ADB dataset.

    The data must be in the folder `folder/ESA-Mission1/` as preprocessed by the official scripts in the repository.
    
    Args:
    folder (str): The root directory containing the ESA mission data files.
    mission (Union[int, str]): The mission number or identifier. 
    period (str): The time period of data to use (e.g. "3_months").
    ds_type (str): The dataset type, either 'train', 'val', or 'test'.
    window_size (int, optional): Number of past time steps to use as input. Defaults to 2.
    horizon_size (int, optional): Number of future time steps to predict. Defaults to 2. 
    stride (int, optional): Step size between consecutive samples. Defaults to 3.
    
    Example usage of ESADataset with DataLoader:

    >>> esa = ESADataset("./data", 1, "3_months", 'train', window_size=2, horizon_size=1, stride=3)
    >>> dataloader = DataLoader(esa, batch_size=4, shuffle=False, num_workers=0)
    >>> for x in dataloader:
    ...     print(x)
    ...     break

    This demonstrates how to create an ESADataset instance and iterate through it using
    a DataLoader with specified batch size and no data shuffling.
    """
    def __init__(self, folder: str,
                mission: Union[int, str],
                period: str,
                ds_type: str,
                window_size: int = 2,
                horizon_size: int = 2,
                stride:int = 1):
        self.window_size = window_size
        self.horizon_size = horizon_size
        self.stride = stride
        self.delta_t_x = None
        self.delta_t_y = None
        if isinstance(mission, int):
            mission = str(mission)
        self.dataset = pd.read_csv(f"{folder}/ESA-Mission{mission}/{period}.{ds_type}.csv")
        self.n_channels = len(list(filter(lambda x: x.startswith('channel_'),  self.dataset.columns)))
        self.dataset.set_index("timestamp", inplace=True)
        self.dataset.index = pd.to_datetime(self.dataset.index)
        self.channels = pd.concat([
            self.dataset[f'channel_{i}'] for i in range(1, self.n_channels+1)
        ], axis=1)
        self.anomalies = pd.concat([
            self.dataset[f'is_anomaly_channel_{i}'] for i in range(1, self.n_channels+1)
        ], axis=1)
        first_index = self.channels.index[0]
        second_index = self.channels.index[1]

        self.length = self.channels.loc[first_index]
        self.delta_index = second_index - first_index
        self.start_time = first_index
        self.end_time = self.channels.index[-1]
    
    def __len__(self):
        return len(self.dataset.index) - (self.window_size+self.horizon_size+1)
    
    def __getitem__(self, idx):
        start_index = self.start_time + self.delta_index*idx
        end_index = start_index + self.delta_index*self.window_size
        start_horizon_index = end_index + self.delta_index
        end_horizon_index = start_horizon_index + self.delta_index*self.horizon_size
        
        signals = torch.from_numpy(self.channels.loc[start_index:end_index].to_numpy(dtype=np.float32)).transpose(1,0)
        if signals.shape[1] < self.window_size:
            filler = torch.zeros([self.n_channels, self.window_size-signals.shape[1]])
            signals = torch.cat([signals, filler], dim=1)

        labels = torch.from_numpy(self.anomalies.loc[start_horizon_index:end_horizon_index].to_numpy(dtype=np.float32)).transpose(1,0)
        if labels.shape[1] < self.horizon_size:
            filler = torch.zeros([self.n_channels, self.horizon_size-labels.shape[1]])
            labels = torch.cat([labels, filler], dim=1)

        return (
            signals,
            labels
        )


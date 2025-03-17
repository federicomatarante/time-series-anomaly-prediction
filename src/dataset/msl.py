import ast
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import kagglehub

# Download latest version
path = kagglehub.dataset_download("patrickfleith/nasa-anomaly-detection-dataset-smap-msl")


# Read the data

def get_labeled_anomalies(dataset_name: str, dataset_type: str):
    """Load labeled anomalies from the NASA spacecraft telemetry dataset.

    :param dataset_name: which dataset to use. Must be in [SMAP, MSL]
    :param dataset_type: type of dataset to load. Must be in ['train', 'test']
    :return: A dict for which the key is the path to the np file and the value is a dict containing:
        - num_values: number of values for the time series
        - sequences: list of tuples [(start_anomaly_idx, end_anomaly_idx),...]
    :raises ValueError: if dataset_name not in [SMAP, MSL] or dataset_type not in ['train', 'test']
    """
    if dataset_name not in ['SMAP', 'MSL']:
        raise ValueError("dataset_name must be in [SMAP, MSL] ")
    if dataset_type not in ['train', 'test']:
        raise ValueError("dataset_type must be in ['train', 'test']")
    labeled_anomalies_file = path / Path('labeled_anomalies.csv')
    labeled_anomalies = pd.read_csv(labeled_anomalies_file)
    filtered_anomalies_labels = labeled_anomalies[labeled_anomalies['spacecraft'] == dataset_name]
    anomalies = {}
    for i in range(len(filtered_anomalies_labels)):
        row = filtered_anomalies_labels.iloc[i]
        file = row['chan_id']
        anomaly_sequences = row['anomaly_sequences']
        anomaly_sequences = ast.literal_eval(anomaly_sequences)  # Convert to list
        # anomaly_classes = row['class'] # Not used
        # sequences = []
        # for j in range(len(anomaly_sequences)):
        # sequences.append((anomaly_sequences[j], anomaly_classes[j]))
        num_values = row['num_values']
        anomalies[path / Path("data/data") / dataset_type / Path(file + ".npy")] = {
            "num_values": num_values,
            "sequences": anomaly_sequences
        }
    return anomalies


class MSLDataset(Dataset):
    """PyTorch Dataset for NASA's MSL spacecraft telemetry data.



    :param ds_type: dataset type to load. Must be in ['train', 'test']
    :param window_size: size of the input sequence window
    :param horizon_size: size of the prediction horizon window
    :param stride: step size for sliding window

    Shapes:
        - sequence: torch.Tensor of shape [window_size, channels]
        - anomaly: torch.Tensor of shape [horizon_size, channels]
    """

    def __init__(self, ds_type: str, window_size: int, horizon_size: int, stride: int):
        if ds_type not in ['test', 'train']:
            raise ValueError("ds_type must be in ['test', 'train'] ")

        self.window_size = window_size
        self.horizon_size = horizon_size
        self.stride = stride
        anomalies = get_labeled_anomalies("MSL", ds_type)
        self.dataset = self._load_dataset(anomalies)

    def _load_dataset(self, anomalies: dict[Path, dict[str, Any]]):
        """Load and process the raw data into sliding windows.

         :param anomalies: dict mapping file paths to metadata containing:
             - num_values: number of values in time series
             - sequences: list of anomaly start/end indices

         Processing steps:
         1. Loads raw numpy arrays and converts to tensors
         2. Creates binary anomaly tensors from metadata
         3. Segments data into overlapping windows using stride
         4. Creates (in_window, out_window) pairs for each window

         Shapes:
             - sequence_tensor: [seq_len, channels] - Raw input sequence
             - anomalies_tensor: [seq_len, channels] - Binary anomaly indicators
             - sequence_piece: [window_size, channels] - Input window
             - anomaly_piece: [horizon_size, channels] - Target window
         """
        raise NotImplementedError()

    def __len__(self):
        """Return the total number of windows in the dataset.

        :return: number of (sequence, anomaly) pairs in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Return the sequence and anomaly tensors for given index.

        :param idx: index of the window to retrieve
        :return: tuple containing:
            - sequence: input sequence window of shape [window_size, channels]
            - anomaly: target anomaly window of shape [horizon_size, channels]
        """
        return self.dataset[idx]

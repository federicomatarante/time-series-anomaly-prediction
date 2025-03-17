import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Type

import torch

from src.dataset.msl_anomaly_prediction import MSLAnomalyPredictionDataset
from src.trainings.patch_tst_trainer import PatchTSTTrainer
from src.utils.config.config_reader import ConfigReader
from src.utils.config.ini_config_reader import INIConfigReader

torch.set_float32_matmul_precision("medium")


def get_args():
    parser = argparse.ArgumentParser(description='Program that accepts a string argument')
    parser.add_argument('--checkpoint', type=str, help='Input string to process', default=None)
    parser.add_argument('--experiment', type=str, help='Experiment name', default="patchtst_training")
    args = parser.parse_args()
    return args


def train_model(dataset_config_name: str, model_config_name: str, train_config_name: str, logs_path: str,
                trainer_class: Type[PatchTSTTrainer], checkpoint_file_name: str = None, configs_dir=None):
    """
    Trains a model using the specified configuration files and trainer class.

    :param dataset_config_name: Name of the dataset configuration file relative to the configs directory.
    :param model_config_name: Name of the model configuration file relative to the configs directory.
    :param train_config_name: Name of the training configuration file relative to the configs directory.
    :param logs_path: Directory path where training logs and outputs will be saved respect to the base project
        directory.
    :param trainer_class: Class that inherits from PatchTSTTrainer and implements the training logic.
    :param checkpoint_file_name: File name of the checkpoint respect to the model_checkpoint.save_directory in the
        training.ini configuration file.
    """
    base_dir = Path(__file__).parent.parent

    logs_dir = base_dir / Path(logs_path)
    os.makedirs(logs_dir, exist_ok=True)
    if not configs_dir:
        configs_dir = base_dir / Path('configs')

    dataset_config_name = INIConfigReader(configs_dir / dataset_config_name, base_path=base_dir)
    patch_tst_config = INIConfigReader(configs_dir / model_config_name, base_path=logs_dir)
    training_config = INIConfigReader(configs_dir / train_config_name, base_path=logs_dir)
    return train_with_configs(dataset_config_name, patch_tst_config, training_config, checkpoint_file_name,
                              trainer_class)


def train_with_configs(dataset_config_name: ConfigReader, patch_tst_config: ConfigReader, training_config: ConfigReader,
                       checkpoint_file_name: str, trainer_class):
    args = get_args()
    valid_split = dataset_config_name.get_param('dataset.valid_split', v_type=float)

    train_dataset = MSLAnomalyPredictionDataset(
        ds_type="train",
        window_size=dataset_config_name.get_param("train.window_size", v_type=int),
        horizon_size=dataset_config_name.get_param("train.horizon_size", v_type=int),
        stride=dataset_config_name.get_param("train.stride", v_type=int),
    )

    test_dataset = MSLAnomalyPredictionDataset(
        ds_type="test",
        window_size=dataset_config_name.get_param("test.window_size", v_type=int),
        horizon_size=dataset_config_name.get_param("test.horizon_size", v_type=int),
        stride=dataset_config_name.get_param("test.stride", v_type=int),
    )
    dataset_size = len(train_dataset)
    valid_size = int(valid_split * dataset_size)
    test_size = dataset_size - (valid_size)
    """print("Normalizing all datasets...")
        train_dataset.normalize()
        test_dataset.normalize()"""

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,
                                                                 [test_size, valid_size])
    """train_dataset_balanced = deepcopy(train_dataset.dataset)  # Ensure you don't modify original dataset
    train_dataset_balanced.data = [train_dataset.dataset[i] for i in train_dataset.indices]
    train_dataset = train_dataset_balanced
    print("Anomalies ratio: ", train_dataset.anomalies_ratio)
    print("Balancing...")
    train_dataset.balance(0.35)
    print("Dataset balanced! New ratio: ", train_dataset.anomalies_ratio)"""
    if checkpoint_file_name is not None:
        checkpoint = f"{checkpoint_file_name}.ckpt" if not checkpoint_file_name.endswith(
            ".ckpt") else checkpoint_file_name
    else:
        checkpoint = None

    trainer = trainer_class(
        model_config=patch_tst_config,
        training_config=training_config,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        experiment_name=args.experiment,
        checkpoint_file=checkpoint,
    )
    trainer.train()
    torch.use_deterministic_algorithms(False)  # I get an error if True (default)
    return trainer.evaluate(test_dataset=test_dataset)

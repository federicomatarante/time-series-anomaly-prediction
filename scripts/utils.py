import argparse
import os
from pathlib import Path
from typing import Type

import torch

from src.dataset.esa import ESADataset
from src.trainings.patch_tst_trainer import PatchTSTTrainer
from src.utils.config.ini_config_reader import INIConfigReader

torch.set_float32_matmul_precision("medium")


def get_args():
    parser = argparse.ArgumentParser(description='Program that accepts a string argument')
    parser.add_argument('--checkpoint', type=str, help='Input string to process', default=None)
    parser.add_argument('--experiment', type=str, help='Experiment name', default="patchtst_training")
    args = parser.parse_args()
    return args


def train_model(dataset_config_name: str, model_config_name: str, train_config_name: str, logs_path: str,
                trainer_class: Type[PatchTSTTrainer], checkpoint_file_name: str = None):
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
    args = get_args()
    base_dir = Path(__file__).parent.parent

    logs_dir = base_dir / Path(logs_path)
    os.makedirs(logs_dir, exist_ok=True)
    configs_dir = base_dir / Path('configs')

    dataset_config_name = INIConfigReader(configs_dir / dataset_config_name, base_path=base_dir)
    patch_tst_config = INIConfigReader(configs_dir / model_config_name, base_path=logs_dir)
    training_config = INIConfigReader(configs_dir / train_config_name, base_path=logs_dir)

    dataset_args = {
        'folder': dataset_config_name.get_param('dataset.folder', v_type=Path),
        'mission': dataset_config_name.get_param('dataset.mission', v_type=str),
        'period': dataset_config_name.get_param('dataset.period', v_type=str),
        'ds_type': dataset_config_name.get_param('dataset.type', v_type=str, domain={'train', 'val', 'test'}),
        'window_size': patch_tst_config.get_param('seq.len', v_type=int),
        'horizon_size': patch_tst_config.get_param('pred.len', v_type=int),
        'stride': dataset_config_name.get_param('windows.stride', v_type=int),
    }

    # split_ratio = dataset_config.get_param('dataset.train_split', v_type=float)
    valid_split = dataset_config_name.get_param('dataset.valid_split', v_type=float)

    dataset = ESADataset(**dataset_args)
    # dataset = Subset(dataset, range(2000))
    dataset_size = len(dataset)
    valid_size = int(valid_split * dataset_size)
    train_size = dataset_size - (valid_size)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_size, valid_size])
    test_dataset = ESADataset(
        folder=dataset_args['folder'],
        mission=1,
        period="84_months",
        ds_type='test',
        window_size=dataset_args['window_size'],
        horizon_size=dataset_args['horizon_size'],
        stride=5,
    )
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
    trainer.evaluate(test_dataset=test_dataset)

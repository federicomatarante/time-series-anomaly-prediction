from pathlib import Path

import torch
from torch.utils.data import random_split
import tqdm

from src.dataset.esa import ESADataset
from src.trainings.patch_tst_trainer import PatchTSTTrainer
from src.utils.config.ini_config_reader import INIConfigReader
from torch.utils.data import DataLoader, Subset


def main():
    configs_path = Path('./configs')
    dataset_config = INIConfigReader(configs_path / 'esa.ini')
    patch_tst_config = INIConfigReader(configs_path / 'patchtst.ini')
    training_config = INIConfigReader(configs_path / 'training.ini')
    dataset_args = {
        'folder': dataset_config.get_param('dataset.folder', v_type=Path),
        'mission': dataset_config.get_param('dataset.mission', v_type=str),
        'period': dataset_config.get_param('dataset.period', v_type=str),
        'ds_type': dataset_config.get_param('dataset.type', v_type=str),
        'window_size': dataset_config.get_param('windows.window_size', v_type=int),
        'horizon_size': dataset_config.get_param('windows.horizon_size', v_type=int),
        'stride': dataset_config.get_param('windows.stride', v_type=int),
    }
    dataset = ESADataset(**dataset_args)
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)

    train_dataset = Subset(dataset, range(train_size))
    valid_dataset = Subset(dataset, range(train_size, len(dataset)))

    
    train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
            # collate_fn=_collate,
    )

    valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
            # collate_fn=_collate,
    )

    # for X, y in tqdm.tqdm(train_loader):
    #     continue

    trainer = PatchTSTTrainer(
        model_config=patch_tst_config,
        training_config=training_config,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        experiment_name='patchtst_training',
        checkpoint_file=None,
    )
    trainer.train()


if __name__ == '__main__':
    # TODO path of ini files from directory
    main()
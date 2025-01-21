import os
from pathlib import Path

from torch.utils.data import Subset

from src.dataset.esa import ESADataset
from src.trainings.patch_tst_trainer import PatchTSTTrainer
from src.utils.config.ini_config_reader import INIConfigReader


def main():
    base_dir = Path(__file__).parent.parent

    logs_dir = base_dir / Path('logs/patch_tst_training')
    os.makedirs(logs_dir, exist_ok=True)
    configs_dir = base_dir / Path('configs')

    dataset_config = INIConfigReader(configs_dir / 'esa.ini', base_path=base_dir)
    patch_tst_config = INIConfigReader(configs_dir / 'patchtst.ini', base_path=logs_dir)
    training_config = INIConfigReader(configs_dir / 'training.ini', base_path=logs_dir)

    dataset_args = {
        'folder': dataset_config.get_param('dataset.folder', v_type=Path),
        'mission': dataset_config.get_param('dataset.mission', v_type=str),
        'period': dataset_config.get_param('dataset.period', v_type=str),
        'ds_type': dataset_config.get_param('dataset.type', v_type=str, domain={'train', 'val', 'test'}),
        'window_size': dataset_config.get_param('windows.prediction_size', v_type=int),
        'horizon_size': dataset_config.get_param('windows.horizon_size', v_type=int),
        'stride': dataset_config.get_param('windows.stride', v_type=int),
    }
    split_ratio = dataset_config.get_param('dataset.train_split', v_type=float)

    dataset = ESADataset(**dataset_args)
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)

    train_dataset = Subset(dataset, range(train_size))
    valid_dataset = Subset(dataset, range(train_size, len(dataset)))

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
    main()

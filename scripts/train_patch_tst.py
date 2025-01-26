import os
from pathlib import Path

from torch.utils.data import Subset

from src.dataset.esa import ESADataset
from src.trainings.patch_tst_trainer import PatchTSTTrainer
from src.utils.config.ini_config_reader import INIConfigReader
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Program that accepts a string argument')
    parser.add_argument('--checkpoint', type=str, help='Input string to process', default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
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
        'window_size': patch_tst_config.get_param('seq.len', v_type=int),
        'horizon_size': patch_tst_config.get_param('pred.len', v_type=int),
        'stride': dataset_config.get_param('windows.stride', v_type=int),
    }
    print(dataset_args)
    split_ratio = dataset_config.get_param('dataset.train_split', v_type=float)

    dataset = ESADataset(**dataset_args)
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)

    train_dataset = Subset(dataset, range(train_size))
    valid_dataset = Subset(dataset, range(train_size, len(dataset)))
    # train_dataset = Subset(dataset, range(512))
    # valid_dataset = Subset(dataset, range(512, 1024))

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = f"{args.checkpoint}.ckpt"

    trainer = PatchTSTTrainer(
        model_config=patch_tst_config,
        training_config=training_config,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        experiment_name='patchtst_training',
        checkpoint_file=checkpoint,
    )
    trainer.train()


if __name__ == '__main__':
    main()

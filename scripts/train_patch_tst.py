from scripts.utils import train_model
from src.trainings.base_patchtst_trainer import BasePatchTSTTrainer
import sys

if __name__ == '__main__':
    ckpt_name = None if len(sys.argv) < 2 else sys.argv[1]
    train_model(
        dataset_config_name='esa.ini',
        model_config_name='patchtst.ini',
        train_config_name='training.ini',
        logs_path='logs/patch_tst_training',
        trainer_class=BasePatchTSTTrainer,
        checkpoint_file_name=ckpt_name,
    )

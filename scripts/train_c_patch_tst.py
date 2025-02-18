from scripts.utils import train_model
from src.trainings.c_patch_tst_trainer import CPatchTSTTrainer

if __name__ == '__main__':
    train_model(
        dataset_config_name='dataset.ini',
        model_config_name='camsa_patchtst.ini',
        train_config_name='camsa_patchtst_training.ini',
        logs_path='logs/c_patch_tst_training',
        trainer_class=CPatchTSTTrainer
    )

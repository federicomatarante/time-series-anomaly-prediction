from scripts.utils import train_model
from src.trainings.c_patch_tst_trainer import CPatchTSTTrainer

if __name__ == '__main__':
    train_model(
        dataset_config_name='esa.ini',
        model_config_name='cpatchtst.ini',
        train_config_name='ctraining.ini',
        logs_path='logs/c_patch_tst_training',
        trainer_class=CPatchTSTTrainer
    )

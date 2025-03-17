from scripts.utils import train_model
from src.trainings.baseline_trianer import BaselineTrainer

if __name__ == '__main__':
    train_model(
        dataset_config_name='dataset.ini',
        model_config_name='baseline.ini',
        train_config_name='baseline_training.ini',
        logs_path='logs/baseline_trainings',
        trainer_class=BaselineTrainer,
    )

from scripts.utils import train_model
from src.trainings.camsa_anomaly_prediction_trainer import CAMSAAnomalyPredictionTrainer

if __name__ == '__main__':
    train_model(
        dataset_config_name='dataset.ini',
        model_config_name='cbart.ini',
        train_config_name='btraining.ini',
        logs_path='logs/camsa_bart_trainings',
        trainer_class=CAMSAAnomalyPredictionTrainer
    )

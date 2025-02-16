from scripts.utils import train_model
from src.trainings.bert_prediction_trainer import BertAnomalyPredictionTrainer

if __name__ == '__main__':
    train_model(
        dataset_config_name='dataset.ini',
        model_config_name='bart.ini',
        train_config_name='btraining.ini',
        logs_path='logs/bert_prediction',
        trainer_class=BertAnomalyPredictionTrainer
    )

from scripts.utils import train_model
from src.trainings.graph_encoder_bert_prediction_trainer import GraphEncoderBertAnomalyPredictionTrainer

if __name__ == '__main__':
    train_model(
        dataset_config_name='dataset.ini',
        model_config_name='graph_encoder_bart.ini',
        train_config_name='graph_encoder_training.ini',
        logs_path='logs/graph_encoder_bert_training',
        trainer_class=GraphEncoderBertAnomalyPredictionTrainer
    )

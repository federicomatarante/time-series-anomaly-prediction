from scripts.utils import train_model
from src.trainings.graph_anomaly_prediction_training import GraphAnomalyPredictionTrainer

if __name__ == '__main__':
    train_model(
        dataset_config_name='dataset.ini',
        model_config_name='graph.ini',
        train_config_name='graph_training.ini',
        logs_path='logs/graph_anomaly_prediction_bert_trainings',
        trainer_class=GraphAnomalyPredictionTrainer
    )

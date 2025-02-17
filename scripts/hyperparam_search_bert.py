from pathlib import Path
from scripts.utils import train_model
from src.trainings.bert_prediction_trainer import BertAnomalyPredictionTrainer
import os
import pickle

if __name__ == '__main__':
    metrics_list  = []
    config_folder = Path('./hyperparams-search')
    configs = os.listdir(config_folder)
    bert_configs = list( filter (lambda x: x.startswith('bart'), configs))
    for x in bert_configs:
        res = dict()
        res['metrics'] = train_model(
            dataset_config_name='dataset.ini',
            model_config_name=x,
            train_config_name='btraining.ini',
            logs_path='logs/bert_prediction',
            configs_dir=config_folder,
            trainer_class=BertAnomalyPredictionTrainer
        )
        res['config'] = {
            'model': x,
            'dataset': 'dataset.ini',
            'training': 'btraining.ini'
        }
        
    
    pickle.dump(metrics_list, open('metrics_list.pkl', 'wb'))

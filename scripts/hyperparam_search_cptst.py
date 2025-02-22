from pathlib import Path
from scripts.utils import train_model
from src.trainings.bert_prediction_trainer import BertAnomalyPredictionTrainer
import os
import pickle

from src.trainings.base_patchtst_trainer import BasePatchTSTTrainer

if __name__ == '__main__':
    metrics_list  = []
    config_folder = Path('./hyperparams-search-ptst')
    configs = os.listdir(config_folder)
    configs = list( filter (lambda x: x.startswith('patchtst'), configs))
    for x in configs:
        res = dict()
        res['metrics'] = train_model(
            dataset_config_name='dataset.ini',
            model_config_name=x,
            train_config_name='training.ini',
            logs_path='logs/patchtst_training',
            configs_dir=config_folder,
            trainer_class=BasePatchTSTTrainer
        )
        res['config'] = {
            'model': x,
            'dataset': 'dataset.ini',
            'training': 'training.ini'
        }
        metrics_list.append(res)
        
    # fl = open('metrics_list.pkl', 'wb')
    # pickle.dump(metrics_list, fl)
    # fl.close()

    fl = open('ptst_comparison.csv', 'w')
    for x in metrics_list:
        fl.write(f"{x['config']['model']},{x['metrics']['test_loss']},{x['metrics']['test_existence']},{x['metrics']['test_density']},{x['metrics']['test_leadtime']},{x['metrics']['test_dice']}\n")
    fl.close()
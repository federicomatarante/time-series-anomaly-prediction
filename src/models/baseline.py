from typing import Tuple

from torch import nn

from src.models.anomaly_prediction_module import AnomalyPredictionModule
from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader


class AnomalyPredictionBaseline(AnomalyPredictionModule):
    """
    Basic Implementation of the AnomalyPredictionModule.
    Encoder: Identity function (no encoding)
    Classifier: Fully Connected Network

    :param model_config: Configuration file with model's hyperparameters.
    :param training_config: Configuration file with training's hyperparameters.
    """
    def __init__(self, model_config: ConfigReader, training_config: ConfigReader):
        # Encoder
        self.model_config = model_config
        # Classifier parameters
        self.layers_size = model_config.get_collection('model.layers_size', v_type=int,
                                                       collection_type=tuple)
        self.hidden_act = model_config.get_param('model.hidden_act', v_type=str)
        self.output_act = model_config.get_param('model.output_act', v_type=str)
        self.dropout = model_config.get_param('model.dropout', v_type=float)

        # Data characteristics
        self.window_size = model_config.get_param('seq.len', v_type=int)
        self.pred_len = model_config.get_param('pred.len', v_type=int)
        self.channels = model_config.get_param('seq.channels', v_type=int)
        super().__init__(training_config, self.channels, self.pred_len, self.window_size)

    def _setup_encoder(self) -> nn.Module:
        """
        Identity encoder that passes input through unchanged.
        """
        return nn.Identity()

    def _setup_classifier(self) -> nn.Module:
        """
        Fully Connected Network classifier that processes the raw input sequence.
        Takes flattened input of size (window_size * channels) and applies
        multiple linear layers with dropout and activation functions.
        """
        # Initialize activations
        hidden_activation = get_activation_fn(self.hidden_act)
        output_activation = get_activation_fn(self.output_act)
        dropout = nn.Dropout(p=self.dropout)
        layers = []
        input_size = self.window_size * self.channels
        for size in self.layers_size:
            layers.extend([
                nn.Linear(input_size, size),
                hidden_activation,
                dropout
            ])
            input_size = size

        layers.extend([
            nn.Linear(input_size, self.pred_len * self.channels),
            output_activation
        ])

        classifier = nn.Sequential(*layers)
        return classifier

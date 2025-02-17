from torch import nn

from src.models.anomaly_prediction_module import AnomalyPredictionModule
from src.models.modules.graph_encoder import GraphCorrelationEncoder
from src.models.utils import init_mlp_classifier_weights
from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader


class GraphAnomalyPrediction(AnomalyPredictionModule):
    """
    Base Implementation of the AnomalyPredictionModule.
    Encoder: Base PatchTST
    Classifier: Fully Connected Layer
    :param model_config: Configuration file with model's hyperparameters.
    :param training_config: Configuration file with training's hyperparameters.
    """

    def __init__(self, model_config: ConfigReader, training_config: ConfigReader):
        # Encoder
        self.model_config_reader = model_config
        # Classifier parameters
        self.c_layers_sizes = model_config.get_collection('classifier.layers_sizes', v_type=int,
                                                          collection_type=tuple)
        self.c_hidden_act = model_config.get_param('classifier.hidden_activation', v_type=str)
        self.dropout = model_config.get_param('training.dropout', v_type=float)

        # Data characteristics
        self.channels = model_config.get_param('data.channels', v_type=int)
        self.pred_len = model_config.get_param('data.output_length', v_type=int)
        self.seq_len = model_config.get_param('data.input_length', v_type=int)

        # Graph information
        self.graph_hidden_layers = model_config.get_collection("graph.hidden_layers", v_type=int, collection_type=list)

        super().__init__(training_config, self.channels, self.pred_len, self.seq_len)

    def _setup_encoder(self) -> nn.Module:
        """
        PatchTST standard encoder.
        """
        graph_encoder = GraphCorrelationEncoder(
            num_nodes=self.channels,
            dropout=self.dropout,
            input_features=self.seq_len,
            embedding_size=self.pred_len,
            hidden_layers_sizes=self.graph_hidden_layers
        )
        return graph_encoder

    def _setup_classifier(self) -> nn.Module:
        """
        Fully Connected Network classifier.
        """
        # Initialize activations
        hidden_activation = get_activation_fn(self.c_hidden_act)
        output_activation = nn.Sigmoid()
        dropout = nn.Dropout(p=self.dropout)
        layers = []
        encoder_output_size = self.pred_len * self.channels
        for size in self.c_layers_sizes:
            layers.extend([
                nn.Linear(encoder_output_size, size),
                hidden_activation,
                dropout
            ])
            encoder_output_size = size

        layers.extend([
            nn.Linear(encoder_output_size, self.pred_len * self.channels),
            output_activation
        ])

        classifier = nn.Sequential(*layers)
        init_mlp_classifier_weights(classifier)
        return classifier

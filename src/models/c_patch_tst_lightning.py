from torch import nn, Tensor

from src.models.anomaly_prediction_module import AnomalyPredictionModule
from src.models.modules.camsa_module import CAMSAModule
from src.models.modules.camsa_patch_tst import CAMSAPatchTST
from src.models.modules.graph_encoder import GraphCorrelationEncoder
from src.models.utils import init_transformer_encoder_weights, init_mlp_classifier_weights, init_gcn_weights
from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader


class CPatchTSTLightning(AnomalyPredictionModule):

    """
    Implementation of the AnomalyPredictionModule.
    A Correlation-aware Patch Time Series Transformer model for time series prediction.
    Encoder:
     This model combines a graph encoder for capturing channel correlations with a
     CAMSA-PatchTST transformer architecture for sequence modeling. It processes multivariate
     time series data to generate future predictions.

     Model Steps:
         1. Graph Encoding:
            - Input time series is processed by the graph encoder
            - Extracts correlation features between different channels/variables
            - Output: correlation features tensor [batch_size, channels, correlation_features]

         2. CAMSA-PatchTST Processing:
            - Time series is divided into patches and embedded
            - Transformer processes the sequence by combining the channels correlations with the CAMSA mechanism.
            - Generates predictions for future timesteps

    Classifier: Fully Connected Model
    :param model_config: Configuration file with model's hyperparameters.
    :param training_config: Configuration file with training's hyperparameters.
     """

    def __init__(self, model_config: ConfigReader, training_config: ConfigReader):
        # Encoder
        self.model_config = model_config
        # Classifier parameters
        self.c_layers_sizes = model_config.get_collection('classifier.layers_sizes', v_type=int,
                                                          collection_type=tuple)
        self.c_hidden_act = model_config.get_param('classifier.activation.hidden', v_type=str)
        self.c_output_act = model_config.get_param('classifier.activation.output', v_type=str)
        self.c_dropout = model_config.get_param('dropout.classifier', v_type=float)

        # Data characteristics
        self.channels = model_config.get_param('data.enc_in', v_type=int)
        self.pred_len = model_config.get_param('pred.len', v_type=int)
        self.seq_len = model_config.get_param('seq.len', v_type=int)
        super().__init__(training_config, self.channels, self.pred_len, self.seq_len)

    def _setup_encoder(self) -> nn.Module:
        """
        PatchTST encoder with Graph Correlation Encoder and CAMSA attention..
        """
        graph_encoder = GraphCorrelationEncoder(self.model_config)
        init_gcn_weights(graph_encoder)
        camsa_patch_tst = CAMSAPatchTST(self.model_config)
        init_transformer_encoder_weights(camsa_patch_tst)
        return CAMSAModule(graph_encoder, camsa_patch_tst)

    def _setup_classifier(self) -> nn.Module:
        """
        Fully Connected Network classifier.
        """
        # Initialize activations
        hidden_activation = get_activation_fn(self.c_hidden_act)
        output_activation = get_activation_fn(self.c_output_act)
        dropout = nn.Dropout(p=self.c_dropout)
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

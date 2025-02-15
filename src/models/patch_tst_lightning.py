from torch import nn

from src.models.anomaly_prediction_module import AnomalyPredictionModule
from src.models.utils import Permute, init_transformer_encoder_weights, init_mlp_classifier_weights
from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader

from src.patchtst.models.PatchTST import Model as PatchTST





class PatchTSTLightning(AnomalyPredictionModule):
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
        PatchTST standard encoder.
        """
        encoder = PatchTST(self.model_config_reader)
        init_transformer_encoder_weights(encoder)
        permute = Permute(0, 2, 1)
        return nn.Sequential(
            permute, encoder, permute
        )

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

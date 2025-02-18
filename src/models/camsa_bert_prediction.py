from torch import nn

from src.models.anomaly_prediction_module import AnomalyPredictionModule
from src.models.modules.bert_anomaly_transformer.anomaly_prediction_transformer import get_anomaly_prediction_bert
from src.models.modules.camsa_bert_anomaly_transformer.camsa_bert import get_camsa_anomaly_prediction_bert
from src.models.modules.graph_encoder import GraphCorrelationEncoder
from src.utils.config.config_reader import ConfigReader



class CAMSAAnomalyPredictionBertLightning(AnomalyPredictionModule):
    """
    Bert module with CAMSA attention mechanism. It's composed by a bert module and a GNN module.
    The GNN module is the following:
        - A Graph is created in which each node is a channel, so the input is (channels,seq_len)
        - An learnable adjacency matrix is used
        - A series of graph convolutional layers are used
        - A final projector creates an embedding matrix C (channels, c_embed_size) which contains information about channels correlation
    The following key features of the bert are:
        - Input (channels, seq_len) is divided into patches and reshaped in (num_patches, channels * patch_size )
        - The patches are projected in a new embedding space (num_patches, embed_dim)
        - The patches are processed in a transformer encoder
        - In the transformer encoder it's used cross attention between the patches and the channels embeddings C by the graph.
        - The output is processed by a MLP and classified with a classification layer
    :param model_config: Configuration file with model's hyperparameters.
    :param training_config: Configuration file with training's hyperparameters.
    """

    def __init__(self, model_config: ConfigReader, training_config: ConfigReader):
        # Encoder
        self.model_config_reader = model_config
        # Classifier parameters
        self.seq_len = model_config.get_param('data.input_length', v_type=int)
        self.channels = model_config.get_param('data.channels', v_type=int)
        self.pred_len = model_config.get_param('data.output_length', v_type=int)
        self.patch_size = model_config.get_param('data.patch_size', v_type=int)

        # Next, get the core model dimension parameters
        # These control the transformer's embedding space and positional encoding
        self.d_embed = model_config.get_param('embedding.d_embed', v_type=int)
        self.positional_encoding = model_config.get_param('embedding.positional_encoding', v_type=str, nullable=True)
        self.relative_position_embedding = model_config.get_param('embedding.relative_position_embedding',
                                                                  v_type=bool)

        # Get transformer block specific parameters
        self.causal_mask = model_config.get_param('transformer_block.causal_mask', v_type=bool)
        self.transformer_n_layer = model_config.get_param('transformer_block.transformer_n_layer', v_type=int)
        self.transformer_n_head = model_config.get_param('transformer_block.transformer_n_head', v_type=int)
        self.hidden_dim_rate = model_config.get_param('transformer_block.hidden_dim_rate', v_type=float)

        # Finally, get training-related parameters
        self.dropout = model_config.get_param('training.dropout', v_type=float)

        # Graph Information
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
            embedding_size=self.d_embed,
            hidden_layers_sizes=self.graph_hidden_layers
        )
        model = get_camsa_anomaly_prediction_bert(
            input_channels=self.channels,
            output_channels=self.channels,
            patch_size=self.patch_size,
            d_embed=self.d_embed,
            hidden_dim_rate=self.hidden_dim_rate,
            input_length=self.seq_len,
            output_length=self.pred_len,
            positional_encoding=self.positional_encoding,
            relative_position_embedding=self.relative_position_embedding,
            transformer_n_layer=self.transformer_n_layer,
            transformer_n_head=self.transformer_n_head,
            dropout=self.dropout,
            causal_mask=self.causal_mask,
            graph_encoder=graph_encoder
        )
        return model

    def _setup_classifier(self) -> nn.Module:
        """
        Fully Connected Network classifier.
        """
        # Initialize activations
        return nn.Identity()

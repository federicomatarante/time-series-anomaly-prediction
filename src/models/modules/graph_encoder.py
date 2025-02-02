from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader
from src.utils.config.ini_config_reader import INIConfigReader


class GraphCorrelationEncoder(nn.Module):
    """Graph Neural Network Encoder with learnable adjacency matrix and projection layer.

       The model consists of:
        - A learnable adjacency matrix that defines the graph structure
        - Multiple GCN layers that process channels features
        - A final projection layer that creates the embedding for each channel
    """

    def __init__(self, model_config_reader: ConfigReader, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Parameters"
        self.threshold = model_config_reader.get_param("graph.adjacency_matrix.threshold", v_type=float)
        adj_dropout = model_config_reader.get_param("graph.adjacency_matrix.dropout", v_type=float)
        self.num_nodes = model_config_reader.get_param("data.enc_in", v_type=int)
        input_features = model_config_reader.get_param("seq.len", v_type=int)
        hidden_layers_sizes = model_config_reader.get_collection("graph.encoder.hidden_sizes", v_type=int,
                                                                 collection_type=tuple)
        hidden_act_name = model_config_reader.get_param('graph.encoder.activation_fn', v_type=str,
                                                        domain={'relu', 'gelu', 'softmax', 'sigmoid', 'tanh',
                                                                'leakyrelu'})
        hidden_dropout = model_config_reader.get_param('graph.encoder.dropout', v_type=float)

        projector_act_name = model_config_reader.get_param('graph.projector.activation_fn', v_type=str,
                                                           domain={'relu', 'gelu', 'softmax', 'sigmoid', 'tanh',
                                                                   'leakyrelu'})
        self.embedding_size = model_config_reader.get_param("graph.projector.embedding_size", v_type=int)
        projector_dropout = model_config_reader.get_param("graph.projector.dropout", v_type=float)

        # Layers
        self.adj = torch.nn.Parameter(torch.rand(self.num_nodes, self.num_nodes))
        self.adj_dropout = nn.Dropout(adj_dropout)
        prev_size = input_features
        self.hidden_layers = nn.ModuleList()
        for size in hidden_layers_sizes:
            self.hidden_layers.append(
                GCNConv(prev_size, size)
            )
            prev_size = size
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.hidden_act = get_activation_fn(hidden_act_name)
        self.projection_layer = nn.Linear(self.num_nodes * prev_size, self.num_nodes * self.embedding_size)
        self.projector_dropout = nn.Dropout(projector_dropout)
        self.projector_act = get_activation_fn(projector_act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Graph Encoder.
        :param x: Input node features matrix. Shape: [batch_size, channels, seq_len].
        :return: Channels embeddings after projection. Shape: [batch_size, channels, embedding_size],
        """
        adj = torch.sigmoid(self.adj)
        adj = self.adj_dropout(adj)

        edge_index = (adj > self.threshold).nonzero().t()

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, edge_index)
            x = self.hidden_dropout(x)
            x = self.hidden_act(x)
        x = x.flatten(-2)
        x = self.projection_layer(x)
        x = self.projector_dropout(x)
        x = self.projector_act(x)
        x = x.view(-1, self.num_nodes, self.embedding_size)
        return x

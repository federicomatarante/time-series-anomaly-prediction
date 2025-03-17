from typing import Any

import torch
from torch import nn

from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader


class WeightedGCNLayer(nn.Module):
    """
    A Graph Convolutional Layer that works with weighted adjacency matrices.
    This layer performs graph convolution operations using continuous edge weights
    rather than binary connections.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Kaiming initialization and zeros for bias."""
        nn.init.kaiming_uniform_(self.weight, mode='fan_out')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the weighted GCN layer.

        Args:
            x: Node features tensor of shape [batch_size, num_nodes, in_features]
            adj: Weighted adjacency matrix of shape [num_nodes, num_nodes]

        Returns:
            Updated node features of shape [batch_size, num_nodes, out_features]
        """
        # Normalize adjacency matrix using degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        # Handle isolated nodes (nodes with degree 0)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Symmetric normalization
        norm_adj = torch.diag(deg_inv_sqrt) @ adj @ torch.diag(deg_inv_sqrt)

        # Linear transformation of node features
        support = torch.matmul(x, self.weight)

        # Aggregate neighborhood information using weighted adjacency
        output = torch.matmul(norm_adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output


class GraphCorrelationEncoder(nn.Module):
    """Graph Neural Network Encoder with learnable adjacency matrix and projection layer.

       The model consists of:
        - A learnable adjacency matrix that defines the graph structure
        - Multiple GCN layers that process channels features
        - A final projection layer that creates the embedding for each channel
    """

    def __init__(self, num_nodes, dropout, input_features, embedding_size, hidden_layers_sizes):
        super().__init__()
        # Parameters"
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        hidden_layers_sizes = hidden_layers_sizes

        # Layers

        # Initializing adjacency matrix closer to identity
        self.adj = torch.nn.Parameter(torch.eye(self.num_nodes) + 0.01 * torch.rand(self.num_nodes, self.num_nodes))

        self.adj_dropout = nn.Dropout(dropout)
        prev_size = input_features
        self.hidden_layers = nn.ModuleList()
        for size in hidden_layers_sizes:
            self.hidden_layers.append(
                WeightedGCNLayer(prev_size, size)
            )
            prev_size = size
        self.hidden_dropout = nn.Dropout(dropout)
        self.hidden_act = nn.GELU()
        self.projection_layer = nn.Linear(self.num_nodes * prev_size, self.num_nodes * embedding_size)
        self.projector_dropout = nn.Dropout(dropout)
        self.projector_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Graph Encoder.

        Args:
            x: Input node features matrix of shape [batch_size, num_nodes, input_features]

        Returns:
            Node embeddings of shape [batch_size, num_nodes, embedding_size]
        """
        # Generate weighted adjacency matrix
        adj = torch.sigmoid(self.adj)  # Constrain weights to [0,1]

        # Apply GCN layers with weighted message passing
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, adj)
            x = self.hidden_dropout(x)
            x = self.hidden_act(x)

        # Project to final embedding space
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # Flatten the features
        x = self.projection_layer(x)
        x = self.projector_dropout(x)
        x = self.projector_act(x)
        # Reshape to final dimensions
        x = x.view(batch_size, self.num_nodes, -1)
        return x

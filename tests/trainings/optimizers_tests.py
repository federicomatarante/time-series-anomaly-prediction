import unittest
from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop, Adadelta, Adagrad, RAdam

from src.trainings.utils.optimizer_factory import OptimizerFactory
from src.utils.config.ini_config_reader import INIConfigReader


class SimpleModel(nn.Module):
    """Simple model for testing optimizers."""

    def __init__(self, input_size: int = 10, output_size: int = 1) -> None:
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TestOptimizerFactory(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.config_reader = INIConfigReader(
            r'C:\Users\feder\PycharmProjects\time-series-anomaly-prediction\configs\training.ini'
        )
        self.factory = OptimizerFactory(self.config_reader)  # Using the existing self.config_reader

    def test_create_adam_optimizer(self) -> None:
        """Test creation of Adam optimizer."""
        optimizer = self.factory.get_optimizer('adam', self.model.parameters())
        self.assertIsInstance(optimizer, Adam)
        self.assertEqual(optimizer.defaults['lr'], self.config_reader.get_param('optimizer.Adam.lr', v_type=float))
        self.assertEqual(optimizer.defaults['eps'], self.config_reader.get_param('optimizer.Adam.eps', v_type=float))
        self.assertEqual(optimizer.defaults['weight_decay'],
                         self.config_reader.get_param('optimizer.Adam.weight_decay', v_type=float))
        self.assertEqual(optimizer.defaults['amsgrad'],
                         self.config_reader.get_param('optimizer.Adam.amsgrad', v_type=bool))

    def test_create_adamw_optimizer(self) -> None:
        """Test creation of AdamW optimizer."""
        optimizer = self.factory.get_optimizer('adamw', self.model.parameters())
        self.assertIsInstance(optimizer, AdamW)
        self.assertEqual(optimizer.defaults['lr'], self.config_reader.get_param('optimizer.AdamW.lr', v_type=float))
        self.assertEqual(optimizer.defaults['eps'], self.config_reader.get_param('optimizer.AdamW.eps', v_type=float))
        self.assertEqual(optimizer.defaults['weight_decay'],
                         self.config_reader.get_param('optimizer.AdamW.weight_decay', v_type=float))
        self.assertEqual(optimizer.defaults['amsgrad'],
                         self.config_reader.get_param('optimizer.AdamW.amsgrad', v_type=bool))

    def test_create_sgd_optimizer(self) -> None:
        """Test creation of SGD optimizer."""
        optimizer = self.factory.get_optimizer('sgd', self.model.parameters())
        self.assertIsInstance(optimizer, SGD)
        self.assertEqual(optimizer.defaults['lr'], self.config_reader.get_param('optimizer.SGD.lr', v_type=float))
        self.assertEqual(optimizer.defaults['momentum'],
                         self.config_reader.get_param('optimizer.SGD.momentum', v_type=float))
        self.assertEqual(optimizer.defaults['dampening'],
                         self.config_reader.get_param('optimizer.SGD.dampening', v_type=float))
        self.assertEqual(optimizer.defaults['weight_decay'],
                         self.config_reader.get_param('optimizer.SGD.weight_decay', v_type=float))
        self.assertEqual(optimizer.defaults['nesterov'],
                         self.config_reader.get_param('optimizer.SGD.nesterov', v_type=bool))

    def test_create_rmsprop_optimizer(self) -> None:
        """Test creation of RMSprop optimizer."""
        optimizer = self.factory.get_optimizer('rmsprop', self.model.parameters())
        self.assertIsInstance(optimizer, RMSprop)
        self.assertEqual(optimizer.defaults['lr'], self.config_reader.get_param('optimizer.RMSprop.lr', v_type=float))
        self.assertEqual(optimizer.defaults['alpha'], self.config_reader.get_param('optimizer.RMSprop.alpha', v_type=float))
        self.assertEqual(optimizer.defaults['eps'], self.config_reader.get_param('optimizer.RMSprop.eps', v_type=float))
        self.assertEqual(optimizer.defaults['weight_decay'],
                         self.config_reader.get_param('optimizer.RMSprop.weight_decay', v_type=float))
        self.assertEqual(optimizer.defaults['momentum'],
                         self.config_reader.get_param('optimizer.RMSprop.momentum', v_type=float))
        self.assertEqual(optimizer.defaults['centered'],
                         self.config_reader.get_param('optimizer.RMSprop.centered', v_type=bool))

    def test_create_adadelta_optimizer(self) -> None:
        """Test creation of Adadelta optimizer."""
        optimizer = self.factory.get_optimizer('adadelta', self.model.parameters())
        self.assertIsInstance(optimizer, Adadelta)
        self.assertEqual(optimizer.defaults['lr'], self.config_reader.get_param('optimizer.Adadelta.lr', v_type=float))
        self.assertEqual(optimizer.defaults['rho'], self.config_reader.get_param('optimizer.Adadelta.rho', v_type=float))
        self.assertEqual(optimizer.defaults['eps'], self.config_reader.get_param('optimizer.Adadelta.eps', v_type=float))
        self.assertEqual(optimizer.defaults['weight_decay'],
                         self.config_reader.get_param('optimizer.Adadelta.weight_decay', v_type=float))

    def test_create_adagrad_optimizer(self) -> None:
        """Test creation of Adagrad optimizer."""
        optimizer = self.factory.get_optimizer('adagrad', self.model.parameters())
        self.assertIsInstance(optimizer, Adagrad)
        self.assertEqual(optimizer.defaults['lr'], self.config_reader.get_param('optimizer.Adagrad.lr', v_type=float))
        self.assertEqual(optimizer.defaults['lr_decay'],
                         self.config_reader.get_param('optimizer.Adagrad.lr_decay', v_type=float))
        self.assertEqual(optimizer.defaults['weight_decay'],
                         self.config_reader.get_param('optimizer.Adagrad.weight_decay', v_type=float))
        self.assertEqual(optimizer.defaults['initial_accumulator_value'],
                         self.config_reader.get_param('optimizer.Adagrad.initial_accumulator_value', v_type=float))
        self.assertEqual(optimizer.defaults['eps'], self.config_reader.get_param('optimizer.Adagrad.eps', v_type=float))

    def test_create_radam_optimizer(self) -> None:
        """Test creation of RAdam optimizer."""
        optimizer = self.factory.get_optimizer('radam', self.model.parameters())
        self.assertIsInstance(optimizer, RAdam)
        self.assertEqual(optimizer.defaults['lr'], self.config_reader.get_param('optimizer.RAdam.lr', v_type=float))
        self.assertEqual(optimizer.defaults['eps'], self.config_reader.get_param('optimizer.RAdam.eps', v_type=float))
        self.assertEqual(optimizer.defaults['weight_decay'],
                         self.config_reader.get_param('optimizer.RAdam.weight_decay', v_type=float))

    def test_invalid_optimizer_type(self) -> None:
        """Test error handling for invalid optimizer type."""
        with self.assertRaises(ValueError) as context:
            self.factory.get_optimizer('invalid_optimizer', self.model.parameters())
        self.assertIn("Unsupported optimizer type", str(context.exception))
        self.assertIn("Supported types are", str(context.exception))

    def test_optimizer_training_step(self) -> None:
        """Test that each optimizer can perform a training step."""
        optimizers: List[str] = ['adam', 'adamw', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'radam']

        for opt_name in optimizers:
            with self.subTest(optimizer=opt_name):
                # Create optimizer
                optimizer = self.factory.get_optimizer(opt_name, self.model.parameters())

                # Create dummy data
                x = torch.randn(32, 10)
                y = torch.randn(32, 1)

                # Forward pass
                output = self.model(x)
                loss = nn.MSELoss()(output, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Check that gradients were updated
                for param in self.model.parameters():
                    self.assertIsNotNone(param.grad)

    def test_parameter_groups(self) -> None:
        """Test optimizer creation with parameter groups."""
        params = [
            {'params': self.model.layer.weight, 'lr': 0.01},
            {'params': self.model.layer.bias, 'lr': 0.001}
        ]

        optimizer = self.factory.get_optimizer('adam', params)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(optimizer.param_groups[1]['lr'], 0.001)

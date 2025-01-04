import unittest
from unittest.mock import patch
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from src.models.patch_tst_lightning import PatchTSTLightning
from src.utils.config.ini_config_reader import INIConfigReader


class TestPatchTSTLightning(unittest.TestCase):
    def setUp(self):
        # Mock config readers

        self.model_config = INIConfigReader(
            r'C:\Users\feder\PycharmProjects\time-series-anomaly-prediction\configs\patchtst.ini'
        )
        self.training_config = INIConfigReader(
            r'C:\Users\feder\PycharmProjects\time-series-anomaly-prediction\configs\training.ini'
        )

        # Initialize model
        self.model = PatchTSTLightning(self.model_config, self.training_config)

        # Create sample data
        self.batch_size = 32
        self.seq_length = 100
        self.n_features = 10
        self.window_size = 5
        self.n_samples = 64

        self.x = torch.randn(self.batch_size, self.seq_length, self.n_features)
        self.y = torch.randint(0, 2, (self.batch_size, self.window_size,)).float()

    def test_model_initialization(self):
        """Test if model initializes correctly with given configurations"""
        self.assertIsInstance(self.model, pl.LightningModule)
        self.assertEqual(self.model.learning_rate, 0.001)
        self.assertEqual(self.model.optimizer_type, 'Adam')

    def test_forward_pass(self):
        """Test if forward pass produces expected output shape"""
        output = self.model(self.x)
        expected_shape = (self.batch_size, self.window_size)
        self.assertEqual(output.shape, expected_shape)

    def test_training_step(self):
        """Test if training step runs without errors and returns loss"""
        batch = (self.x, self.y)
        loss = self.model.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Loss should be a scalar

    def test_validation_step(self):
        """Test if validation step computes metrics correctly"""
        batch = (self.x, self.y)
        loss = self.model.validation_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(hasattr(self.model.val_existence, 'compute'))

    def test_test_step(self):
        """Test if test step computes metrics correctly"""
        batch = (self.x, self.y)
        loss = self.model.test_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(hasattr(self.model.test_existence, 'compute'))

    def test_configure_optimizers(self):
        """Test if optimizer and scheduler are configured correctly"""
        optim_config = self.model.configure_optimizers()

        self.assertIn('optimizer', optim_config)
        self.assertIn('lr_scheduler', optim_config)
        self.assertIsInstance(optim_config['optimizer'], torch.optim.Adam)
        self.assertEqual(optim_config['lr_scheduler']['monitor'], 'val_loss')

    def test_metrics_thresholds(self):
        """Test if metrics are initialized with correct thresholds"""
        self.assertEqual(self.model.train_existence.threshold, 0.5)
        self.assertEqual(self.model.val_existence.threshold, 0.5)
        self.assertEqual(self.model.test_existence.threshold, 0.5)

    @patch('torch.save')
    def test_save_hyperparameters(self, mock_save):
        """Test if hyperparameters are saved correctly"""
        self.assertTrue(hasattr(self.model, 'hparams'))
        self.assertIn('learning_rate', self.model.hparams)
        self.assertIn('optimizer_type', self.model.hparams)

    def test_loss_computation(self):
        """Test if loss function computes without errors"""
        y_pred = torch.randn(self.seq_length, 1)
        y_true = torch.randint(0, 2, (self.seq_length, 1)).float()

        loss = self.model.loss_fn(y_pred, y_true)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_with_trainer(self):
        """Test model with PyTorch Lightning Trainer"""
        # Create sample dataset
        x = torch.randn(self.n_samples, self.seq_length, self.n_features)
        y = torch.randint(0, 2, (self.n_samples, self.window_size)).float()
        dataset = TensorDataset(x, y)

        # Create dataloaders
        train_loader = DataLoader(dataset, batch_size=self.batch_size)
        val_loader = DataLoader(dataset, batch_size=self.batch_size)

        # Initialize trainer with minimal settings
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='auto',
            logger=False,
            enable_checkpointing=False
        )

        # Test fit and validate
        trainer.fit(self.model, train_loader, val_loader)
        result = trainer.validate(self.model, val_loader)

        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
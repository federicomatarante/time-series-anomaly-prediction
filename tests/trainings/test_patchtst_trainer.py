import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path

from src.trainings.patch_tst_trainer import PatchTSTTrainer
from src.utils.config.config_reader import ConfigReader
from src.utils.config.ini_config_reader import INIConfigReader


class MockDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(32, 100, 10), torch.randn(32, 5)


class TestPatchTSTTrainer(unittest.TestCase):
    def setUp(self):
        # Create mock configurations
        self.model_config = INIConfigReader(
            r'C:\Users\feder\PycharmProjects\time-series-anomaly-prediction\configs\patchtst.ini'
        )
        self.training_config = INIConfigReader(
            r'C:\Users\feder\PycharmProjects\time-series-anomaly-prediction\configs\training.ini'
        )

        # Create mock datasets
        self.train_dataset = MockDataset(100)
        self.val_dataset = MockDataset(20)

    def test_initialization(self):
        """Test trainer initialization with default parameters"""
        trainer = PatchTSTTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset
        )

        self.assertEqual(trainer.batch_size, self.training_config.get_param('training.batch_size', v_type=int))
        self.assertEqual(trainer.num_workers, self.training_config.get_param('training.num_workers', v_type=int))
        self.assertEqual(trainer.max_epochs, self.training_config.get_param('training.max_epochs', v_type=int))
        self.assertEqual(trainer.hardware_accelerator,
                         self.training_config.get_param('hardware.accelerator', v_type=str))

    def test_dataloader_setup(self):
        """Test dataloader configuration"""
        trainer = PatchTSTTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset
        )

        train_loader, val_loader = trainer._setup_dataloaders()

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertEqual(train_loader.batch_size, trainer.batch_size)
        self.assertEqual(val_loader.batch_size, trainer.batch_size)

    def test_callback_setup(self):
        """Test callback configuration"""
        trainer = PatchTSTTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset
        )

        callbacks = trainer._setup_callbacks()

        self.assertEqual(len(callbacks), 2)  # ModelCheckpoint and EarlyStopping
        self.assertIsInstance(callbacks[0], pl.callbacks.ModelCheckpoint)
        self.assertIsInstance(callbacks[1], pl.callbacks.EarlyStopping)

    @patch('pytorch_lightning.Trainer')
    def test_evaluate(self, mock_pl_trainer):
        """Test model evaluation"""
        trainer = PatchTSTTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset
        )

        # Configure mock trainer
        mock_trainer_instance = mock_pl_trainer.return_value
        mock_trainer_instance.test.return_value = [{
            'test_loss': 0.1,
            'test_existence': 0.95,
            'test_density': 0.85,
            'test_leadtime': 0.75,
            'test_dice': 0.9
        }]

        # Create test dataset
        test_dataset = MockDataset(50)

        # Run evaluation
        metrics = trainer.evaluate(test_dataset=test_dataset)

        # Verify results
        self.assertIn('test_loss', metrics)
        self.assertIn('test_existence', metrics)
        self.assertIn('test_density', metrics)
        self.assertIn('test_leadtime', metrics)
        self.assertIn('test_dice', metrics)

    def test_checkpoint_loading(self):
        """Test loading from checkpoint"""
        # Create temporary checkpoint path
        checkpoint_path = "test_checkpoint.ckpt"

        with patch('pathlib.Path.exists', return_value=True), \
                patch('src.trainings.patch_tst_lightning.PatchTSTLightning.load_from_checkpoint') as mock_load:
            trainer = PatchTSTTrainer(
                model_config=self.model_config,
                training_config=self.training_config,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                checkpoint_file=checkpoint_path
            )

            mock_load.assert_called_once()

    def test_invalid_checkpoint(self):
        """Test handling of invalid checkpoint path"""
        with self.assertRaises(ValueError):
            trainer = PatchTSTTrainer(
                model_config=self.model_config,
                training_config=self.training_config,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                checkpoint_file="nonexistent.ckpt"
            )

    def test_evaluate_without_dataset(self):
        """Test evaluate method raises error when no dataset provided"""
        trainer = PatchTSTTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset
        )

        with self.assertRaises(ValueError):
            trainer.evaluate()

import os
import random
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer

from src.models.c_patch_tst_lightning import CPatchTSTLightning
from src.models.patch_tst_lightning import PatchTSTLightning
from src.utils.config.config_reader import ConfigReader


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, PyTorch, and CUDA.

    :param seed: seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: ensure CuDNN uses deterministic algorithms
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def collate_batch(batch):
    """
    Collate function for DataLoader that stacks inputs and targets.

    Args:
        batch: List of tuples (input, target)
    Returns:
        Tuple of (stacked_inputs, stacked_targets)
    """
    return (
        torch.stack([x[0] for x in batch]),
        torch.stack([x[1] for x in batch])
    )


class PatchTSTTrainer(ABC):
    """
    PatchTST model trainer that handles training, evaluation, and testing procedures.

    The trainer provides functionality for:
    - Model training with validation
    - Model evaluation on test data
    - Checkpoint management
    - Training resumption
    - Hardware acceleration
    - Logging and monitoring

    Required configuration parameters:

    Training Config:
        training:
            - batch_size: int - Batch size for training and evaluation
            - num_workers: int - Number of workers for data loading
            - max_epochs: int - Maximum number of training epochs
            - gradient_clip_value: float (optional) - Value for gradient clipping

        model_checkpoint:
            - save_directory: str - Directory to save model checkpoints
            - filename: str - Format for checkpoint filenames
            - monitor: str - Metric to monitor for checkpointing
            - mode: str - 'min' or 'max' for metric monitoring
            - save_top_k: int - Number of best models to save
            - save_last: bool - Whether to save the last model

        early_stopping:
            - monitor: str - Metric to monitor for early stopping
            - patience: int - Number of epochs to wait before stopping
            - mode: str - 'min' or 'max' for metric monitoring
            - min_delta: float - Minimum change to qualify as improvement
            - verbose: bool - Whether to print early stopping info

        hardware:
            - num_devices: int - Number of devices to use
            - accelerator: str - Type of accelerator ('cpu', 'gpu', etc.)

        logging:
            - save_directory: str - Directory to save training logs

    :param model_config: Configuration reader for model parameters
    :param training_config: Configuration reader for training parameters
    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param experiment_name: Name of the experiment for logging
    :param checkpoint_file: Optional path to checkpoint file for resuming training

    Example:
         # Initialize configuration
         model_config = ConfigReader("model_config.yaml")
         training_config = ConfigReader("training_config.yaml")
        
         # Create datasets
         train_dataset = YourDataset(...)
         val_dataset = YourDataset(...)
        
         # Initialize trainer
         trainer = PatchTSTTrainer(
             model_config=model_config,
             training_config=training_config,
             train_dataset=train_dataset,
             val_dataset=val_dataset,
             experiment_name="anomaly_detection",
             checkpoint_file=None
         )
        
         # Train the model
         results = trainer.train()
         print(f"Best model path: {results['best_model_path']}")
        
         # Evaluate on test data
         test_dataset = YourDataset(...)
         metrics = trainer.evaluate(test_dataset=test_dataset)
         print(f"Test loss: {metrics['test_loss']}")
    """

    def __init__(
            self,
            model_config: ConfigReader,
            training_config: ConfigReader,
            train_dataset: Dataset,
            val_dataset: Dataset,
            experiment_name: str = "patchtst_training",
            checkpoint_file: Optional[str] = None
    ):
        """
        Initialize the PatchTST trainer.

        Args:
            model_config: Configuration reader for model parameters
            training_config: Configuration reader for training parameters
            train_dataset: Training dataset
            val_dataset: Validation dataset
            experiment_name: Name of the experiment for logging
        """

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.experiment_name = experiment_name

        # Seed
        seed = training_config.get_param('training.seed', v_type=int, nullable=True)
        if seed is not None:
            set_seed(seed)

        # Setup training parameters
        self.batch_size = training_config.get_param('training.batch_size', v_type=int)
        self.num_workers = training_config.get_param('training.num_workers', v_type=int)
        self.max_epochs = training_config.get_param('training.max_epochs', v_type=int)
        self.gradient_clip_value = training_config.get_param('training.gradient_clip_value', v_type=float,
                                                             nullable=True)

        # Setup checkpoints
        self.checkpoint_dir = training_config.get_param('model_checkpoint.save_directory', v_type=Path)
        self.checkpoint_file_name = training_config.get_param('model_checkpoint.filename', v_type=str)
        self.checkpoint_monitor = training_config.get_param('model_checkpoint.monitor', v_type=str)
        self.checkpoint_mode = training_config.get_param('model_checkpoint.mode', v_type=str)
        self.checkpoint_save_top_k = training_config.get_param('model_checkpoint.save_top_k', v_type=int)
        self.checkpoint_save_last = training_config.get_param('model_checkpoint.save_last', v_type=bool)

        # Setup Early Stopping
        self.early_stopping_monitor = training_config.get_param('early_stopping.monitor', v_type=str)
        self.early_stopping_patience = training_config.get_param('early_stopping.patience', v_type=int)
        self.early_stopping_mode = training_config.get_param('early_stopping.mode', v_type=str)
        self.early_stopping_min_delta = training_config.get_param('early_stopping.min_delta', v_type=float)
        self.early_stopping_verbose = training_config.get_param('early_stopping.verbose', v_type=bool)

        # Setup hardware
        self.hardware_num_devices = training_config.get_param('hardware.num_devices', v_type=int)
        self.hardware_accelerator = training_config.get_param('hardware.accelerator', v_type=str)

        # Setup directories
        self.log_dir = training_config.get_param('logging.save_directory', v_type=Path)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        # Setup model
        self.checkpoint_path, self.model = self.setup_model(checkpoint_file)

        # Setup callbacks
        callbacks = self._setup_callbacks()

        # Setup trainer
        self._setup_trainer(callbacks)

    @abstractmethod
    def setup_model(self, checkpoint_file: Optional[str]) -> Tuple[Path, nn.Module]:
        """
        Model to inherit to specify the class-behavior. Must return the checkpoint path and the model to use for trianing.
        :param checkpoint_file: the name of the checkpoint file to use. Can be null.
        :return: the checkpoint path and the model to use
        """
        pass

    # TODO fix dataset!

    def _setup_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Set up training and validation dataloaders.

        :return: Tuple containing (train_dataloader, val_dataloader)

        Example:
             train_loader, val_loader = trainer._setup_dataloaders()
             next(iter(train_loader))  # Get first batch
        """

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            #collate_fn=collate_batch,

        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            #collate_fn=collate_batch,
        )

        return train_loader, val_loader

    def _setup_callbacks(self) -> list:
        """
        Set up training callbacks for checkpointing and early stopping.

        :return: List of configured callbacks

        Example:
             callbacks = trainer._setup_callbacks()
             checkpoint_callback = callbacks[0]
             early_stopping = callbacks[1]
        """
        callbacks = []

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=self.checkpoint_file_name,
            monitor=self.checkpoint_monitor,
            mode=self.checkpoint_mode,
            save_top_k=self.checkpoint_save_top_k,
            save_last=self.checkpoint_save_last,

        )
        callbacks.append(checkpoint_callback)

        # Early stopping callback
        early_stop_callback = EarlyStopping(
            monitor=self.early_stopping_monitor,
            patience=self.early_stopping_patience,
            mode=self.early_stopping_mode,
            min_delta=self.early_stopping_min_delta,
            verbose=self.early_stopping_verbose
        )
        callbacks.append(early_stop_callback)

        return callbacks

    def _setup_trainer(self, callbacks: list):
        """
        Set up the PyTorch Lightning trainer with specified configuration.

        :param callbacks: List of callbacks for training
        :return: Configured PyTorch Lightning trainer

        Example:
             callbacks = trainer._setup_callbacks()
             lightning_trainer = trainer._setup_trainer(callbacks)
             lightning_trainer.fit(...)
        """

        # Setup logger
        logger = TensorBoardLogger(
            save_dir=self.log_dir,
            name=self.experiment_name,
        )

        # Get hardware acceleration config
        accelerator = self.hardware_accelerator
        devices = self.hardware_num_devices

        # Initialize trainer
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            deterministic=False,
            gradient_clip_val=self.gradient_clip_value,
        )

    def train(self) -> Dict[str, Any]:
        """
        Train the model with the specified configuration.

        :return: Dictionary containing training results:
            - best_model_path: Path to the best checkpoint
            - best_model_score: Score of the best model
            - trained_epochs: Number of epochs trained
            - training_log_dir: Directory containing training logs

        Example:
             results = trainer.train()
             print(f"Best model score: {results['best_model_score']}")
             print(f"Trained for {results['trained_epochs']} epochs")
        """
        # Check compatibility
        self.model.check_compatibility(self.train_dataset)

        # Setup dataloaders
        train_loader, val_loader = self._setup_dataloaders()

        # Train the model
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=self.checkpoint_path
        )

        # Return the best model path and training results
        results = {
            'best_model_path': self.trainer.checkpoint_callback.best_model_path,
            'best_model_score': self.trainer.checkpoint_callback.best_model_score,
            'trained_epochs': self.trainer.current_epoch,
            'training_log_dir': self.trainer.logger.log_dir
        }

        return results

    def evaluate(
            self,
            test_dataset: Optional[torch.utils.data.Dataset] = None,
            test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        :param test_dataset: Dataset for evaluation
        :param test_dataloader: Pre-configured dataloader for evaluation
        :return: Dictionary containing evaluation metrics:
            - test_loss: Overall test loss
            - test_existence: Existence of anomaly metric
            - test_density: Density of anomalies metric
            - test_leadtime: Lead time metric
            - test_dice: Dice score metric
        :raises ValueError: If neither test_dataset nor test_dataloader is provided

        Example:
             # Evaluate using dataset
             metrics = trainer.evaluate(test_dataset=test_dataset)
             print(f"Test loss: {metrics['test_loss']}")
             
             # Or evaluate using dataloader
             test_loader = DataLoader(test_dataset, batch_size=32)
             metrics = trainer.evaluate(test_dataloader=test_loader)
             print(f"Dice score: {metrics['test_dice']}")
        """
        if test_dataloader is None:
            if test_dataset is None:
                raise ValueError("Either test_dataset or test_dataloader must be provided")

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True
            )

        # Run test
        test_results = self.trainer.test(self.model, dataloaders=test_dataloader)[0]

        # Format metrics
        metrics = {
            'test_loss': test_results['test_loss'],
            'test_existence': test_results['test_existence'],
            'test_density': test_results['test_density'],
            'test_leadtime': test_results['test_leadtime'],
            'test_dice': test_results['test_dice']
        }

        return metrics

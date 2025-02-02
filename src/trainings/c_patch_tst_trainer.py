from pathlib import Path
from typing import Optional, Tuple

from torch import nn
from torch.utils.data import Dataset

from src.models.c_patch_tst_lightning import CPatchTSTLightning
from src.trainings.patch_tst_trainer import PatchTSTTrainer
from src.utils.config.config_reader import ConfigReader


class CPatchTSTTrainer(PatchTSTTrainer):

    def __init__(self, model_config: ConfigReader, training_config: ConfigReader, train_dataset: Dataset,
                 val_dataset: Dataset, experiment_name: str = "c_patchtst_training",
                 checkpoint_file: Optional[str] = None):
        self.model_config = model_config
        self.training_config = training_config
        super().__init__(model_config, training_config, train_dataset, val_dataset, experiment_name, checkpoint_file)

    def setup_model(self, checkpoint_file: Optional[str]) -> Tuple[Path, nn.Module]:
        # Setup model and checkpoint file
        if checkpoint_file:
            checkpoint_path = Path(self.checkpoint_dir) / Path(checkpoint_file)
            if not self.checkpoint_path.exists():
                raise ValueError(f"Checkpoint file not found: {self.checkpoint_path}")
            model = CPatchTSTLightning.load_from_checkpoint(
                self.checkpoint_path,
                model_config_reader=self.model_config,
                training_config_reader=self.training_config
            )
        else:
            # Initialize model
            checkpoint_path = None
            model = CPatchTSTLightning(
                model_config=self.model_config,
                training_config=self.training_config,
            )
        return checkpoint_path, model

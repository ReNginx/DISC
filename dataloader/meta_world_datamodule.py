"""
Lightning DataModule for Meta-World dataset with HDF5 file loading.
"""

import copy
from pathlib import Path
from typing import Optional, Dict, Any
import lightning as pl

from torch.utils.data import DataLoader, random_split

import torch

from .meta_world_dataloader import MetaWorldDataset, MetaWorldDataConfig


class MetaWorldDataModule(pl.LightningDataModule):
    """Lightning DataModule for Meta-World dataset."""

    def __init__(
        self,
        task_suite_name: str = "meta_world",
        data_root_path: str = "./data/meta-world",
        image_size: tuple = (224, 224),
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.8,
        seed: int = 42,
        horizon: int = 20,
        train_split_file: Optional[str] = None,
        test_split_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize MetaWorldDataModule.

        Args:
            task_suite_name: Meta-World task suite name
            data_root_path: Path to Meta-World data
            image_size: Image size (height, width)
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            train_val_split: Train/validation split ratio
            seed: Random seed for splits
            horizon: Action sequence horizon length
            train_split_file: Path to train split txt file (optional)
            test_split_file: Path to test split txt file (optional)
        """
        super().__init__()

        # Pop eval_config from kwargs if present (dataset-specific eval configuration)
        self.eval_config = kwargs.pop('eval_config', None)

        # Store parameters
        self.task_suite_name = task_suite_name
        self.data_root_path = Path(data_root_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed
        self.horizon = horizon
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file

        # Create dataset configuration
        self.config = MetaWorldDataConfig(
            task_suite_name=task_suite_name,
            data_root_path=str(data_root_path),
            image_size=image_size,
            horizon=horizon,
            split_file=None,  # Will be set per dataset in setup()
            **kwargs,
        )

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""

        # Check if we have separate train/val split files
        if self.train_split_file and self.test_split_file:
            # Use provided split files for train and validation
            if stage == "fit":
                train_config = copy.deepcopy(self.config)
                train_config.split = "train"
                train_config.split_file = self.train_split_file
                self.train_dataset = MetaWorldDataset(train_config)
                print(f"Using split files - Train dataset size: {len(self.train_dataset)}")

            val_config = copy.deepcopy(self.config)
            val_config.split = "val"  # Use val split name for clarity
            val_config.split_file = self.test_split_file
            self.val_dataset = MetaWorldDataset(val_config)
            print(f"Using split files - Validation dataset size: {len(self.val_dataset)}")
        else:
            # No split files provided, use random splitting
            train_config = copy.deepcopy(self.config)
            train_config.split = "train"
            train_config.split_file = None

            full_train_dataset = MetaWorldDataset(train_config)

            # Split into train and validation
            train_size = int(self.train_val_split * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, [train_size, val_size], generator=generator
            )

            print(f"Random split - Train dataset size: {len(self.train_dataset)}")
            print(f"Random split - Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            # Load test split
            test_config = copy.deepcopy(self.config)
            test_config.split = "test"
            test_config.split_file = self.test_split_file

            self.test_dataset = MetaWorldDataset(test_config)
            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

    def get_eval_config(self) -> Optional[Dict[str, Any]]:
        """Get evaluation configuration if available."""
        return self.eval_config

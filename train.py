#!/usr/bin/env python3
"""
Main training / evaluation entrypoint for HyLaP on LIBERO and Meta-World.

Uses Hydra for configuration management and PyTorch Lightning for training.

Usage:
    python train.py --config-name=hylap                      # Train HyLaP on LIBERO-90
    python train.py --config-name=hylap data=libero_spatial  # Train on a different LIBERO suite
    python train.py --config-name=hylap data=metaworld       # Train on Meta-World
    python train.py --config-name=libero_90_ttt              # Test-time training on LIBERO
    python train.py --config-name=metaworld_ttt              # Test-time training on Meta-World
    python train.py model.optimizer.lr=5e-4 data.batch_size=64
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as pl
from lightning import seed_everything
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trainer.generic_trainer import GenericTrainer

torch.set_float32_matmul_precision("medium")

def create_trainer(cfg: DictConfig) -> pl.Trainer:
    """Create PyTorch Lightning trainer from config."""
    
    # Extract trainer config
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    target = trainer_cfg.pop("_target_")
    
    # Handle callbacks separately
    callbacks = []
    if "callbacks" in trainer_cfg:
        callback_configs = trainer_cfg.pop("callbacks")
        for callback_config in callback_configs:
            callback_target = callback_config.pop("_target_")
            callback_class = hydra.utils.get_class(callback_target)
            callbacks.append(callback_class(**callback_config))
    
    # Create logger
    logger = None
    if "logger" in cfg:
        logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
        logger_target = logger_cfg.pop("_target_")
        logger_class = hydra.utils.get_class(logger_target)
        logger = logger_class(**logger_cfg)
    
    # Create trainer
    trainer_class = hydra.utils.get_class(target)
    trainer = trainer_class(
        callbacks=callbacks,
        logger=logger,
        **trainer_cfg
    )
    
    return trainer


def create_model(cfg: DictConfig) -> GenericTrainer:
    """Create model from config."""
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    target = model_cfg.pop("_target_")

    model_class = hydra.utils.get_class(target)
    model = model_class(**model_cfg)

    return model


def create_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Create datamodule from config."""
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    target = data_cfg.pop("_target_")

    # Resolve paths
    if "data_root_path" in data_cfg:
        data_cfg["data_root_path"] = str(Path(data_cfg["data_root_path"]).expanduser().resolve())

    datamodule_class = hydra.utils.get_class(target)
    datamodule = datamodule_class(**data_cfg)

    return datamodule


@hydra.main(version_base=None, config_path="config", config_name="hylap")
def main(cfg: DictConfig) -> Optional[float]:
    """Main training function."""

    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)
        print(f"Seed set to {cfg.seed}")

    # Create components
    print("Creating model...")
    model = create_model(cfg)

    print("Creating datamodule...")
    datamodule = create_datamodule(cfg)

    print("Creating trainer...")
    trainer = create_trainer(cfg)

    # Log hyperparameters
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    stage = cfg.get("stage", "fit")

    # Setup data
    print("Setting up data...")
    datamodule.prepare_data()

    if stage == "fit":
        # Start training
        print("Starting training...")
        trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path", None))
    else:
        print("Starting evaluation...")
        model.load_state_dict(torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))["state_dict"], strict=True)
        trainer.validate(model, datamodule, ckpt_path=None)

if __name__ == "__main__":
    main()

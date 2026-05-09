"""
Generic PyTorch Lightning trainer base class that can be configured via YAML.

This module provides a flexible base trainer class that handles common functionality
like data loading, optimizer configuration, encoder management, and video recording.
Specific training strategies should inherit from this class and implement their
own forward, training, and validation logic.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import lightning as pl
import re
import json
from pathlib import Path

from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import importlib
import numpy as np

from tqdm import tqdm

# 添加libero路径到sys.path
libero_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "modified_libero")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

# LIBERO imports for environment testing
from libero.libero.utils import get_libero_path

from model.image_encoders import create_image_encoder
from model.language_encoders import create_language_encoder
from transformers import BertTokenizerFast


class GenericTrainer(pl.LightningModule):
    """
    Generic PyTorch Lightning trainer base class configurable via YAML.

    This base trainer handles common functionality:
    - Image encoders (ResNet, ConvNet, etc.)
    - Language encoders (GRU, LSTM, etc.)
    - Observation encoding and conditioning
    - Optimizer and scheduler configuration
    - Video recording during evaluation
    - LIBERO environment testing during validation
    - Test-time training/adaptation support

    Subclasses should implement specific training strategies and the following abstract methods:
    - _create_policy_network(): Create the policy network
    - forward(): Forward pass through the model
    - training_step(): Training step logic
    - validation_step(): Validation step logic
    - predict_step(): Prediction step for inference
    - prepare_fast_adapt(): Prepare model for test-time adaptation
    - restore_adaptation(): Restore model state after adaptation
    - get_adaptation_parameters(): Get parameters to adapt for a task
    - get_adaptation_optimizer(): Get optimizer for adaptation
    """

    def __init__(
        self,
        # Model components (configured via YAML)
        image_encoder: Optional[Dict[str, Any]] = None,
        language_encoder: Optional[Dict[str, Any]] = None,
        state_encoder: Optional[Dict[str, Any]] = None,
        # Task configuration
        action_dim: int = 7,
        state_dim: int = 8,  # Default state dimension for LIBERO
        # Training configuration
        optimizer: Dict[str, Any] = None,
        lr_scheduler: Optional[Dict[str, Any]] = None,
        loss_weights: Dict[str, float] = None,
        # Conditioning
        use_language_conditioning: bool = True,
        use_state_conditioning: bool = True,
        language_tokenizer: Optional[Dict[str, Any]] = None,
        # Data configuration
        data_config: Optional[Dict[str, Any]] = None,
        # Evaluation configuration
        eval_config: Optional[Dict[str, Any]] = None,
        # Test-time training configuration
        test_time_training_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize generic trainer base class.

        Args:
            image_encoder: Image encoder configuration
            language_encoder: Language encoder configuration
            state_encoder: State encoder configuration
            action_dim: Action space dimension
            state_dim: State vector dimension
            optimizer: Optimizer configuration
            lr_scheduler: Learning rate scheduler configuration
            loss_weights: Loss component weights
            use_language_conditioning: Whether to use language conditioning
            use_state_conditioning: Whether to use state conditioning
            language_tokenizer: Tokenizer configuration
            data_config: Data configuration containing image_size and other data parameters
            eval_config: Evaluation configuration
            test_time_training_config: Test-time training configuration
        """
        super().__init__()
        self.save_hyperparameters()

        # Store configuration
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_language_conditioning = use_language_conditioning
        self.use_state_conditioning = use_state_conditioning
        self.data_config = data_config or {}

        # Set default configurations
        if optimizer is None:
            optimizer = {"_target_": "torch.optim.AdamW", "lr": 1e-4, "weight_decay": 1e-4}

        if loss_weights is None:
            loss_weights = {"action_loss": 1.0}

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.loss_weights = loss_weights

        # Create encoders
        self.image_encoder = create_image_encoder(image_encoder)
        self.language_encoder = create_language_encoder(language_encoder)
        self.tokenizer = self.language_encoder.tokenizer

        self.state_encoder = self._create_state_encoder(state_encoder)

        # Setup evaluator based on evaluation config
        self._setup_evaluator(eval_config)

        # Test-time training setup
        self._setup_test_time_training(test_time_training_config)

        print(f"GenericTrainer initialized:")
        print(f"  Image encoder: {image_encoder}")
        print(f"  Language encoder: {language_encoder}")
        print(f"  State encoder: {state_encoder}")
        print(f"  Action dim: {action_dim}")
        print(f"  State dim: {state_dim}")
        print(f"  Evaluation enabled: {self.eval_enabled}")
        print(f"  Test-time training enabled: {self.test_time_training_enabled}")

    def on_fit_start(self):
        # Log distributed setup info
        world_size, global_rank = self._get_distributed_info()
        if world_size > 1:
            print(f"  Multi-GPU setup: rank {global_rank}/{world_size}")
        else:
            print(f"  Single GPU setup")

    def on_validation_start(self):
        self.on_fit_start()

    def _get_num_views(self) -> int:
        # Check if cameras are explicitly listed in data_config
        cameras = self.data_config.get("cameras", [])
        if isinstance(cameras, list) and len(cameras) > 0:
            return len(cameras)

        task_suite_name = str(self.data_config.get("task_suite_name", ""))
        if "libero" in task_suite_name.lower():
            return 2
        return 1

    def _get_distributed_info(self) -> Tuple[int, int]:
        """Get world size and global rank for distributed training."""
        try:
            world_size = self.trainer.world_size if self.trainer else 1
            global_rank = self.trainer.global_rank if self.trainer else 0
        except AttributeError:
            world_size = 1
            global_rank = 0
        return world_size, global_rank

    def _setup_evaluator(self, config: Optional[Dict[str, Any]]):
        """Setup evaluator based on evaluation configuration."""
        # Setup evaluation config
        from evaluator.base_evaluator import setup_eval_config

        self.eval_config = setup_eval_config(config)
        self.eval_enabled = self.eval_config["enabled"]

        # Create appropriate evaluator based on config
        if self.eval_enabled:
            # Lazy import to avoid circular imports
            from evaluator import LiberoEvaluator, MetaWorldEvaluator

            # Determine evaluator type based on task suite name or other indicators
            task_suite_name = self.eval_config.get("task_suite_name", "")

            if "libero" in task_suite_name.lower():
                self.evaluator = LiberoEvaluator(self, self.eval_config)
                print(f"Created LiberoEvaluator")
            elif "meta_world" in task_suite_name.lower():
                self.evaluator = MetaWorldEvaluator(self, self.eval_config)
                print(f"Created MetaWorldEvaluator")
            else:
                raise ValueError(f"Unknown task suite name: {task_suite_name}")
        else:
            self.evaluator = None
            print(f"Evaluation disabled")

    def _setup_test_time_training(self, config: Optional[Dict[str, Any]]):
        """Setup test-time training configuration."""
        self.test_time_training_config = config

        self.test_time_training_enabled = self.test_time_training_config.get("enabled", False)

        if self.test_time_training_enabled:
            print(f"Test-time training enabled")
        else:
            print(f"Test-time training disabled")

    def _create_state_encoder(self, config: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
        """
        Create state encoder from configuration.

        Args:
            config: State encoder configuration

        Returns:
            State encoder module or None if no state conditioning
        """
        if not self.use_state_conditioning:
            return None

        # Default simple MLP state encoder if config is None
        if config is None:
            config = {"_target_": "torch.nn.Linear", "in_features": self.state_dim, "out_features": 256}

        target = config.pop("_target_")

        if target == "torch.nn.Linear":
            return nn.Linear(**config)
        elif target == "torch.nn.Dropout":
            p = config.pop("p")
            print(f"\033[91mCreating state encoder with Dropout rate {p}\033[0m")
            return nn.Sequential(nn.Dropout(p), nn.Linear(**config))
        else:
            # Dynamic import for custom state encoders
            module_path, class_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            encoder_class = getattr(module, class_name)
            return encoder_class(**config)

    @abstractmethod
    def _create_policy_network(self, config: Dict[str, Any]) -> nn.Module:
        """
        Create policy network from configuration.

        This method must be implemented by subclasses to create
        their specific policy networks.

        Args:
            config: Policy network configuration

        Returns:
            Policy network module
        """
        pass

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        This method must be implemented by subclasses to define
        their specific forward logic.

        Args:
            batch: Input batch containing observations and actions

        Returns:
            Model outputs
        """
        pass

    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step.

        This method must be implemented by subclasses to define
        their specific training logic.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Loss dictionary
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        This method must be implemented by subclasses to define
        their specific validation logic.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Validation metrics dictionary
        """
        pass

    @abstractmethod
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Prediction step for inference.

        This method must be implemented by subclasses to define
        their specific inference logic.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Predicted actions
        """
        pass

    def prepare_fast_adapt(self, config: Dict[str, Any]) -> Any:
        """
        Prepare the model for fast adaptation to a specific task.

        This method should save any necessary state before adaptation
        and return information needed for restoration.

        Args:
            config: Configuration dictionary containing adaptation parameters

        Returns:
            State information needed for restoration (implementation-specific)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support test-time training. "
            "Please implement prepare_fast_adapt() method or disable test-time training."
        )

    def restore_adaptation(self, restoration_info: Any) -> None:
        """
        Restore the model state after fast adaptation.
        
        This method should restore the model to its pre-adaptation state
        using the information returned by prepare_fast_adapt.

        Args:
            restoration_info: State information from prepare_fast_adapt
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support test-time training. "
            "Please implement restore_adaptation() method or disable test-time training."
        )

    def get_adaptation_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Get the optimizer for test-time adaptation.

        This allows subclasses to specify the optimizer configuration
        for adaptation (learning rate, optimizer type, etc.).

        Args:
            config: Configuration dictionary containing optimizer parameters

        Returns:
            Optimizer instance for adaptation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support test-time training. "
            "Please implement get_adaptation_optimizer() method or disable test-time training."
        )

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        optimizer_config = self.optimizer_config.copy()
        optimizer_target = optimizer_config.pop("_target_")

        if optimizer_target == "torch.optim.AdamW":
            optimizer = AdamW(self.parameters(), **optimizer_config)
        elif optimizer_target == "torch.optim.Adam":
            optimizer = Adam(self.parameters(), **optimizer_config)
        elif optimizer_target == "torch.optim.SGD":
            optimizer = SGD(self.parameters(), **optimizer_config)
        else:
            # Dynamic import
            module_path, class_name = optimizer_target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            optimizer_class = getattr(module, class_name)
            optimizer = optimizer_class(self.parameters(), **optimizer_config)

        # Create scheduler if specified
        if self.lr_scheduler_config is not None:
            scheduler_config = self.lr_scheduler_config.copy()
            scheduler_target = scheduler_config.pop("_target_")

            if scheduler_target == "torch.optim.lr_scheduler.CosineAnnealingLR":
                scheduler = CosineAnnealingLR(optimizer, **scheduler_config)
            elif scheduler_target == "torch.optim.lr_scheduler.StepLR":
                scheduler = StepLR(optimizer, **scheduler_config)
            elif scheduler_target == "torch.optim.lr_scheduler.ExponentialLR":
                scheduler = ExponentialLR(optimizer, **scheduler_config)
            else:
                # Dynamic import
                module_path, class_name = scheduler_target.rsplit(".", 1)
                module = importlib.import_module(module_path)
                scheduler_class = getattr(module, class_name)
                scheduler = scheduler_class(optimizer, **scheduler_config)

            return [optimizer], [scheduler]

        return optimizer

    def _perform_test_time_training(self, task_name: str, suite_name: str) -> bool:
        """
        Perform test-time training/adaptation for a specific task.

        This method uses the abstract methods prepare_fast_adapt() and
        get_adaptation_optimizer() to allow subclasses to control the adaptation process.

        Args:
            task_name: Name of the task to adapt to
            suite_name: Name of the task suite

        Returns:
            True if adaptation was successful, False otherwise
        """
        from torch.utils.data import DataLoader

        print(f"Starting test-time training for task: {task_name}")

        config = self.test_time_training_config

        dataset = self.evaluator.create_test_time_dataset(
            task_name=task_name,
            suite_name=suite_name,
            data_root_path=config["data_root_path"],
            horizon=config["horizon"],
            max_demos=config.get("max_demos"),
        )

        print(f"Created dataset with {len(dataset)} transitions for adaptation")

        # Create dataloader
        batch_size = config.get("adaptation_batch_size", config.get("batch_size", 8))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,  # Keep simple for test-time training
            pin_memory=True,
        )
        # Store original training state
        original_training_state = self.training

        # Set model to training mode for adaptation
        self.eval()

        adaptation_optimizer = self.get_adaptation_optimizer(self.test_time_training_config)

        print(f"Starting adaptation with {config['num_adaptation_steps']} steps")

        step_count = 0
        total_loss = 0.0

        # Adaptation loop
        while step_count < config["num_adaptation_steps"]:
            for batch_idx, batch in enumerate(dataloader):
                if step_count >= config["num_adaptation_steps"]:
                    break

                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device).detach().clone()

                # Zero gradients
                adaptation_optimizer.zero_grad()

                # Forward pass and compute loss using the trainer's training_step
                loss_dict = self.training_step(batch, batch_idx, is_fast_adapt=True)

                # Extract loss
                if isinstance(loss_dict, dict) and "loss" in loss_dict:
                    loss = loss_dict["loss"]
                elif isinstance(loss_dict, torch.Tensor):
                    loss = loss_dict
                else:
                    print(f"Unexpected loss format: {type(loss_dict)}")
                    continue

                # Backward pass
                loss.backward()
                adaptation_optimizer.step()

                total_loss += loss.item()
                step_count += 1

                if step_count % 10 == 0:
                    avg_loss = total_loss / step_count
                    print(f"  Adaptation step {step_count}/{config['num_adaptation_steps']}, avg_loss: {avg_loss:.4f}")

        avg_loss = total_loss / max(step_count, 1)
        print(f"Test-time training completed. Average loss: {avg_loss:.4f}")

        return True

    def on_validation_epoch_end(self):
        """Run environment evaluation at the end of validation epochs using evaluator."""
        if not self.eval_enabled or self.evaluator is None:
            # Log default metrics when evaluation is disabled
            self.log("overall_success_rate", 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return

        if self.trainer.sanity_checking:
            print("Sanity check: skipping evaluation")
            return

        # Run evaluation using the evaluator (JSON saving handled internally)
        overall_success_rate = self.evaluator.evaluate_all_tasks(
            current_epoch=self.current_epoch, logger_log_dir=self.logger.log_dir
        )

        # Log the overall success rate for checkpoint saving and progress tracking
        self.log("overall_success_rate", overall_success_rate, on_step=False, on_epoch=True, prog_bar=True)

"""
Trainer module for PyTorch Lightning-based model training.

Includes the GenericTrainer base class and the HyLaPTrainer implementation.
"""

from .generic_trainer import GenericTrainer
from .hylap_trainer import HyLaPTrainer

__all__ = [
    "GenericTrainer",
    "HyLaPTrainer",
]

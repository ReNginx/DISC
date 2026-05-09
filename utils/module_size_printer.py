"""
Reusable module size analysis utilities for PyTorch models.

This module provides generic functions for analyzing and printing the parameter count
and memory usage of PyTorch models and their components.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def count_parameters(module: nn.Module) -> int:
    """Count the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_module_size_mb(module: nn.Module) -> float:
    """Get the approximate memory size of a module in MB."""
    total_params = 0
    for param in module.parameters():
        if param.requires_grad:
            total_params += param.numel()
    
    # Assuming float32 (4 bytes per parameter)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def print_model_summary(
    model: nn.Module,
    module_info: List[Tuple[str, int, float]],
    sort_by: str = "size",
    model_name: str = "Model"
) -> None:
    """
    Print a formatted summary of model module sizes.
    
    Args:
        model: PyTorch model instance
        module_info: List of tuples (module_name, parameter_count, size_mb)
        sort_by: Either "size" (MB) or "params" (parameter count)
        model_name: Name to display in the header
    """
    print("=" * 80)
    print(f"{model_name} Module Size Analysis")
    print("=" * 80)
    
    # Get total model info
    total_params = count_parameters(model)
    total_size = get_module_size_mb(model)
    
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Total Model Size: {total_size:.2f} MB")
    print("-" * 80)
    
    # Sort by specified criteria
    if sort_by == "size":
        module_info.sort(key=lambda x: x[2])  # Sort by size_mb
        sort_label = "Size (MB)"
    else:
        module_info.sort(key=lambda x: x[1])  # Sort by parameter count
        sort_label = "Parameters"
    
    print(f"Modules sorted by {sort_label} (smallest to largest):")
    print("-" * 80)
    print(f"{'Module Name':<40} {'Parameters':<15} {'Size (MB)':<12} {'% of Total':<10}")
    print("-" * 80)
    
    for name, param_count, size_mb in module_info:
        percentage = (param_count / total_params) * 100 if total_params > 0 else 0
        print(f"{name:<40} {param_count:<15,} {size_mb:<12.4f} {percentage:<10.2f}%")
    
    print("=" * 80)
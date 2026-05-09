"""
Base evaluator class for environment-specific evaluation.

This module provides the abstract base class for all evaluator implementations.
Each environment (LIBERO, MetaWorld, etc.) should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import os
import json
import numpy as np
import imageio


def get_default_eval_config() -> Dict[str, Any]:
    """Get default evaluation configuration."""
    return {
        "enabled": True,
        "task_suite_name": "libero_spatial",
        "num_episodes_per_task": 3,
        "num_parallel": 5,  # Number of parallel environments (default to num_episodes_per_task)
        "max_steps": 300,
        "camera_name": "agentview",
        "eval_frequency": 5,  # Run every N epochs
        "max_tasks": 2,  # Limit number of tasks for faster validation
        "save_video": True,
        "video_dir": "videos",
        "metrics_dir": "metrics",
        "split_txt": None,  # Path to text file containing task IDs to evaluate
    }


def setup_eval_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Setup evaluation configuration with defaults."""
    default_config = get_default_eval_config()

    if config is None:
        return default_config

    # Merge with defaults
    merged_config = default_config.copy()
    merged_config.update(config)

    # Ensure num_parallel is at most num_episodes_per_task
    if "num_parallel" not in merged_config:
        merged_config["num_parallel"] = min(merged_config["num_episodes_per_task"], default_config["num_parallel"])
    merged_config["num_parallel"] = min(merged_config["num_parallel"], merged_config["num_episodes_per_task"])

    return merged_config


class BaseEvaluator(ABC):
    """
    Abstract base class for environment evaluation.
    
    This class defines the interface that all evaluator classes must implement.
    Each evaluator class is responsible for:
    - Setting up the evaluation environment (single GPU only)
    - Running evaluation episodes
    - Computing metrics
    - Saving results to JSON files
    - Returning a universal accuracy metric for the trainer
    """

    def __init__(self, trainer, eval_config: Dict[str, Any]):
        """
        Initialize the evaluator class.
        
        Args:
            trainer: The trainer instance to use for prediction
            eval_config: Configuration dictionary for evaluation
        """
        self.trainer = trainer
        self.eval_config = eval_config

        # Setup evaluation configuration
        self._setup_eval_config()

    @property
    def device(self):
        return self.trainer.device

    @abstractmethod
    def _setup_eval_config(self):
        """Setup evaluation configuration with defaults."""
        pass

    @abstractmethod
    def evaluate_all_tasks(self, current_epoch: int, logger_log_dir: str) -> float:
        """
        Evaluate all tasks and return universal accuracy.
        
        Args:
            current_epoch: Current training epoch
            logger_log_dir: Directory for saving logs and metrics
            
        Returns:
            Universal accuracy metric (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def evaluate_single_task(self, task_id: int, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single task.
        
        Args:
            task_id: ID of the task to evaluate
            task_info: Task information dictionary
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    def _save_metrics_to_json(self, metrics: Dict[str, Any], save_path: str):
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary to save
            save_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")

    def _check_task_metrics_exist(self, task_id: int, task_name: str, current_epoch: int, metric_save_dir: str) -> bool:
        """
        Check if metrics already exist for a specific task.
        
        Args:
            task_id: ID of the task
            task_name: Name of the task
            current_epoch: Current training epoch
            metric_save_dir: Directory where metrics are saved
            
        Returns:
            True if metrics file exists for this task, False otherwise
        """
        metrics_file = os.path.join(metric_save_dir, f"task_{task_id}_{task_name}_epoch_{current_epoch}.json")
        return os.path.exists(metrics_file)

    def _save_single_task_metrics(self, task_metrics: Dict[str, Any], current_epoch: int, metric_save_dir: str) -> None:
        """
        Save metrics for a single task instantly.
        
        Args:
            task_metrics: Metrics dictionary for a single task
            current_epoch: Current training epoch
            metric_save_dir: Directory to save metrics
        """
        if not task_metrics:
            return
            
        task_id = task_metrics.get("task_id", "unknown")
        task_name = task_metrics.get("task_name", "unknown").replace(" ", "_").replace("/", "_")
        
        # Create single task metrics data
        single_task_data = {
            "epoch": current_epoch,
            "task_metrics": task_metrics,
        }
        
        os.makedirs(metric_save_dir, exist_ok=True)
        metrics_file = os.path.join(metric_save_dir, f"task_{task_id}_{task_name}_epoch_{current_epoch}.json")
        
        self._save_metrics_to_json(single_task_data, metrics_file)

    def _compute_overall_success_rate(self, all_metrics: list) -> float:
        """
        Compute overall success rate from all task metrics.
        
        Args:
            all_metrics: List of metrics dictionaries from all tasks
            
        Returns:
            Overall success rate (mean success rate across all tasks)
        """
        if not all_metrics:
            return 0.0

        success_rates = [m.get("success_rate", 0.0) for m in all_metrics]
        return float(np.mean(success_rates))

    def _compute_universal_accuracy(self, all_task_metrics: List[Dict]) -> float:
        """
        Compute universal accuracy from all task metrics.
        
        This is an alias for _compute_overall_success_rate for consistency.
        
        Args:
            all_task_metrics: List of metrics dictionaries from all tasks
            
        Returns:
            Universal accuracy (mean success rate across all tasks)
        """
        return self._compute_overall_success_rate(all_task_metrics)

    def save_episode_videos(
        self, episode_frames: List[List], success_list: List[bool], task_name: str,
        video_save_dir: str, current_epoch: int, global_rank: Optional[int] = None
    ) -> None:
        """Save episode videos to disk."""
        video_save_dir_epoch = os.path.join(video_save_dir, f"epoch_{current_epoch}", f"{task_name}")
        os.makedirs(video_save_dir_epoch, exist_ok=True)
        for i, frames in enumerate(episode_frames):
            if frames:  # Only save if we have frames
                success_str = "success" if success_list[i] else "fail"
                rank_str = f"_rank_{global_rank}" if global_rank is not None else ""
                video_path = os.path.join(video_save_dir_epoch, f"ep{i:03d}{rank_str}_{success_str}.mp4")
                imageio.mimsave(video_path, frames, fps=20)
                # print(f"Video saved to {video_path}")

    def compute_task_metrics(
        self, task_id: int, task_name: str, success_list: List[bool],
        episode_steps: List[int], num_episodes: int, global_rank: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute evaluation metrics for a task."""
        metrics = {
            "task_id": task_id,
            "task_name": task_name,
            "success_rate": np.mean(success_list),
            "avg_episode_length": np.mean(episode_steps),
            "num_episodes": num_episodes,
        }
        if global_rank is not None:
            metrics["global_rank"] = global_rank
        return metrics

    def save_evaluation_metrics(
        self, all_task_metrics: List[Dict], current_epoch: int,
        metric_save_dir: str, global_rank: Optional[int] = None
    ) -> None:
        """Save evaluation metrics to JSON file."""
        if not all_task_metrics:
            return

        overall_success_rate = np.mean([m["success_rate"] for m in all_task_metrics])
        overall_avg_length = np.mean([m["avg_episode_length"] for m in all_task_metrics])

        metrics_data = {
            "epoch": current_epoch,
            "overall_success_rate": overall_success_rate,
            "overall_avg_length": overall_avg_length,
            "task_metrics": all_task_metrics,
        }
        if global_rank is not None:
            metrics_data["global_rank"] = global_rank

        os.makedirs(metric_save_dir, exist_ok=True)
        if global_rank is not None:
            metrics_file = os.path.join(metric_save_dir, f"metrics_epoch_{current_epoch}_rank_{global_rank}.json")
        else:
            metrics_file = os.path.join(metric_save_dir, f"metrics_epoch_{current_epoch}.json")

        self._save_metrics_to_json(metrics_data, metrics_file)

"""
MetaWorld environment evaluator class.

This module provides evaluation functionality specifically for MetaWorld environments.
It handles MetaWorld-specific setup, episode execution, and metric computation.

Test-Time Training Support:
--------------------------
This evaluator supports test-time training/adaptation where the model can be fine-tuned
on task-specific data before evaluation. This is controlled by the test_time_training_config
in the trainer configuration.

To enable test-time training:
1. Set test_time_training_config.enabled = True in your trainer config
2. Configure the adaptation parameters (learning rate, steps, etc.)
3. Optionally specify which tasks to adapt to in tasks_to_adapt list

Example configuration:
```yaml
test_time_training_config:
  enabled: true
  data_root_path: "./data/meta-world"
  num_adaptation_steps: 50
  batch_size: 8
  horizon: 20
  adaptation_lr: 1e-4
  adaptation_weight_decay: 1e-4
  lora_rank: 16  # For LoRA-based adaptation
  tasks_to_adapt: []  # Empty means adapt to all tasks
```

The adaptation process:
1. Creates a per-task dataset using training data for the specific task
2. Calls trainer.prepare_fast_adapt() to prepare the model (e.g., add LoRA layers)
3. Runs adaptation training loop with task-specific data
4. Evaluates the adapted model on the task
5. Calls trainer.restore_adaptation() to restore the original model state

This allows for task-specific fine-tuning without permanently modifying the base model.

Extra Description Support:
-------------------------
This evaluator supports using extra language descriptions for tasks from a JSON file.
The JSON file should contain task-specific instructions organized by task name and split.

Example JSON structure:
```json
{
  "task_name": {
    "train": ["instruction1", "instruction2", ...],
    "eval": ["instruction1", "instruction2", ...]
  }
}
```

To enable extra descriptions:
1. Set extra_desc_json_path in the evaluation configuration
2. The evaluator will load and use these instructions during evaluation
3. Instructions are distributed evenly across episodes for each task
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
import metaworld
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path

from .base_evaluator import BaseEvaluator, setup_eval_config


class MetaWorldEvaluator(BaseEvaluator):
    """
    MetaWorld environment evaluator class.

    This class handles evaluation for MetaWorld environments, including:
    - Single GPU evaluation only (no multi-GPU distribution)
    - Metrics computation and JSON saving
    - Support for multiple MetaWorld tasks
    """

    def __init__(self, trainer, eval_config: Dict[str, Any]):
        """
        Initialize MetaWorld evaluator.

        Args:
            trainer: The trainer instance to use for prediction
            eval_config: MetaWorld evaluation configuration
        """
        super().__init__(trainer, eval_config)
        self.action_min = None
        self.action_max = None

        # Initialize MetaWorld components
        self._initialize_metaworld_components()
        self._load_action_norm_stats()

        # Setup image transform to match dataloader
        image_norm_type = "imagenet"
        if hasattr(self.trainer, "data_config") and self.trainer.data_config:
            image_norm_type = self.trainer.data_config.get("image_norm_type", "imagenet")
        
        if image_norm_type == "imagenet":
            img_norm_transform = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0)
        elif image_norm_type == "standard":
            img_norm_transform = A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0)
        else:
            img_norm_transform = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0)

        self.image_transform = A.Compose([
            img_norm_transform,
            ToTensorV2(),
        ])

    def _setup_eval_config(self):
        """Setup MetaWorld evaluation configuration with defaults."""
        # Use base evaluator setup function
        self.metaworld_eval_config = setup_eval_config(self.eval_config)
        self.eval_enabled = self.metaworld_eval_config["enabled"]
        self.action_norm_stats_path = self.metaworld_eval_config.get("action_norm_stats_path", None)
        self.action_smoothing = self.metaworld_eval_config.get("action_smoothing", False)

    def _initialize_metaworld_components(self):
        """Initialize MetaWorld benchmark components."""
        if not self.eval_enabled:
            return

        # Get available tasks from MetaWorld ML45_V3
        train_envs = metaworld.env_dict.ML45_V3["train"]
        test_envs = metaworld.env_dict.ML45_V3["test"]

        tasks_dict = {**train_envs, **test_envs}

        # Load tasks from split file if provided
        split_txt = self.metaworld_eval_config.get("split_txt")
        if split_txt and os.path.exists(split_txt):
            with open(split_txt, "r") as f:
                task_names = [line.strip() for line in f.readlines() if line.strip()]
            self.task_list = [(name, tasks_dict[name]) for name in task_names if name in tasks_dict]
        else:
            # Use all tasks from the suite
            self.task_list = list(tasks_dict.items())

        # Load extra descriptions if provided
        self.extra_descriptions = None
        if "extra_desc_json_path" in self.metaworld_eval_config and self.metaworld_eval_config["extra_desc_json_path"]:
            self.extra_descriptions = self._load_extra_descriptions(self.metaworld_eval_config["extra_desc_json_path"])

        print(f"MetaWorld evaluator initialized with {len(self.task_list)} tasks.")

    def _load_action_norm_stats(self):
        if not self.eval_enabled:
            return

        stats_path = self.action_norm_stats_path

        if stats_path is None or not os.path.exists(stats_path):
            print(f"\033[91mWarning: Action norm stats path {stats_path} not found\033[0m")
            return

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        self.action_min = stats.get("action_min", None)
        self.action_max = stats.get("action_max", None)
        print(f"Loaded action normalization stats from: {stats_path}")

    def _unnormalize_actions_from_minus_one_one(self, actions: np.ndarray) -> np.ndarray:
        if self.action_min is None or self.action_max is None:
            return actions

        action_min = np.asarray(self.action_min)
        action_max = np.asarray(self.action_max)
        denom = action_max - action_min
        denom = np.where(denom == 0, 1.0, denom)
        actions_01 = (actions + 1.0) / 2.0
        return actions_01 * denom + action_min

    def _get_image_size(self) -> int:
        """Get image size from trainer's data configuration."""
        if hasattr(self.trainer, 'data_config') and self.trainer.data_config:
            image_size = self.trainer.data_config.get('image_size', [224, 224])
            print(f"\033[91mimage_size: {image_size}\033[0m")
            if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                # Use the first dimension (assuming square images or use minimum)
                return min(image_size[0], image_size[1])
            elif isinstance(image_size, int):
                return image_size

        # Default fallback
        print("\033[91mWarning: Could not read image_size from data config, using default 224\033[0m")
        return 224

    def _load_extra_descriptions(self, extra_desc_json_path: str) -> Dict[str, Dict[str, List[str]]]:
        """Load extra descriptions from JSON file."""
        json_path = Path(extra_desc_json_path)
        if not json_path.exists():
            print(f"Warning: Extra descriptions file not found: {extra_desc_json_path}")
            return None

        try:
            with open(json_path, 'r') as f:
                descriptions = json.load(f)
            print(f"Loaded extra descriptions for {len(descriptions)} tasks from {extra_desc_json_path}")
            return descriptions
        except Exception as e:
            print(f"Warning: Failed to load extra descriptions from {extra_desc_json_path}: {e}")
            return None

    def _get_task_instruction_and_init_states_ids(self, task_name: str, num_episodes: int, use_test_split: bool = True) -> Tuple[List[str], List[int]]:
        """
        Get task instructions from extra descriptions if available.
        
        Args:
            task_name: Name of the task
            num_episodes: Number of episodes to generate instructions for
            use_test_split: Whether to use test/eval split (True) or train split (False)
            
        Returns:
            Tuple of (List of task instruction strings for each episode, List of init state IDs)
        """
        if not self.extra_descriptions:
            return [task_name] * num_episodes, np.arange(num_episodes)

        # Convert task name to format that matches JSON keys
        # MetaWorld task names are like "assembly-v3", convert to "assembly_v3" for matching
        task_underscore = task_name.replace("-v3", "")

        # Find matching task in extra descriptions
        matching_task = None
        for json_task_name in self.extra_descriptions.keys():
            # Direct match or check if JSON key contains our task
            if json_task_name == task_underscore or json_task_name.endswith(task_underscore):
                matching_task = json_task_name
                break

        if not matching_task:
            print(f"Warning: No extra description found for task '{task_name}', using original")
            return [task_name] * num_episodes, np.arange(num_episodes)

        # Select instruction based on split
        split_key = "eval" if use_test_split else "train"

        if split_key in self.extra_descriptions[matching_task]:
            instructions = self.extra_descriptions[matching_task][split_key]
            if instructions:
                # Use numpy to create deterministic indices for instruction selection
                # This ensures even distribution across all available instructions
                num_instructions = len(instructions)
                indices = np.arange(num_episodes) % num_instructions
                indices = np.sort(indices)
                init_states_ids = np.zeros(num_episodes, dtype=int)
                for i in range(indices[-1]+1):
                    mask = indices == i
                    init_states_ids[mask] = np.arange(mask.sum())

                print(f"  Using {num_instructions} unique instructions for {num_episodes} episodes")
                return [instructions[idx] for idx in indices], init_states_ids

        # Fallback to original task description
        print(f"Warning: No {split_key} instructions found for task '{matching_task}', using original")
        return [task_name] * num_episodes, np.arange(num_episodes)

    def create_test_time_dataset(self, task_name: str, suite_name: str, data_root_path: str, horizon: int = 20, max_demos: Optional[int] = None):
        """
        Create a per-task dataset for test-time training.
        
        Args:
            task_name: Name of the task to create dataset for
            suite_name: Name of the task suite (e.g., 'meta_world')
            data_root_path: Root path to MetaWorld data
            horizon: Action sequence horizon
            max_demos: Maximum number of demos to load
            
        Returns:
            MetaWorldPerTaskDataset instance
        """
        from dataloader.meta_world_dataloader import MetaWorldPerTaskDataset, MetaWorldPerTaskDataConfig

        assert suite_name == "meta_world", "Only MetaWorld is supported for test-time training"

        # Get image size from trainer's data configuration
        image_size = self._get_image_size()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        config = MetaWorldPerTaskDataConfig(
            task_suite_name=suite_name,
            data_root_path=data_root_path,
            split="train",  # Use training data for test-time adaptation
            task_name=task_name,
            horizon=horizon,
            max_demos=max_demos,
            debug=False,
            image_size=image_size,  # Add image_size configuration
            state_dim=getattr(self.trainer, 'state_dim', 8),  # Get from trainer with fallback
            action_dim=getattr(self.trainer, 'action_dim', 4)  # Get from trainer with fallback
        )

        try:
            dataset = MetaWorldPerTaskDataset(config)
            print(f"Created test-time dataset for task '{task_name}' with {len(dataset)} transitions")
            return dataset
        except Exception as e:
            print(f"Failed to create dataset for task '{task_name}': {e}")
            return None

    def _resize_image_batch(self, images: np.ndarray, target_resolution: int) -> np.ndarray:
        """
        Resize a batch of images to target resolution while maintaining aspect ratio.

        Args:
            images: Input images as numpy array (N, H, W, C)
            target_resolution: Target resolution (will be target_resolution x target_resolution)

        Returns:
            np.ndarray: Resized images of shape (N, target_resolution, target_resolution, C)
        """
        if images is None or images.size == 0:
            return np.zeros((len(images), target_resolution, target_resolution, 3), dtype=np.uint8)

        batch_size = images.shape[0]
        resized_batch = []

        for i in range(batch_size):
            image = images[i]
            h, w = image.shape[:2]

            # If already the correct size, add as is
            if h == target_resolution and w == target_resolution:
                resized_batch.append(image)
                continue

            # Resize using OpenCV with high quality interpolation
            resized = cv2.resize(image, (target_resolution, target_resolution), interpolation=cv2.INTER_LANCZOS4)

            # Ensure the image is in the correct format (H, W, C) and dtype
            if len(resized.shape) == 2:  # Grayscale
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 4:  # RGBA
                resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)

            resized_batch.append(resized.astype(np.uint8))

        return np.array(resized_batch)

    def _create_vectorized_env(self, env_name: str, num_envs: int, seed: int = 42):
        """
        Create vectorized MetaWorld environments using gym.make_vec.

        Args:
            env_name: Name of the environment to create
            num_envs: Number of parallel environments
            seed: Random seed for environments

        Returns:
            Vectorized environment instance
        """
        envs = gym.make_vec(
            "Meta-World/custom-mt-envs",
            vector_strategy="sync",
            envs_list=[env_name] * num_envs,
            seed=seed,
            render_mode="rgb_array",
            camera_name="custom",
        )
        return envs

    def _extract_xyz_gripper(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract xyz position and gripper state from MetaWorld observation.

        Args:
            obs: MetaWorld observation array

        Returns:
            np.ndarray: 4D array containing [x, y, z, gripper]
        """
        # Same logic as in collect_data.py
        if len(obs) >= 4:
            xyz = obs[:3]  # xyz position
            gripper = obs[3:4]  # gripper state (usually 1D)
            return np.concatenate([xyz, gripper])
        else:
            # Fallback: pad with zeros if observation is too short
            return np.pad(obs, (0, max(0, 4 - len(obs))))[:4]

    def evaluate_all_tasks(self, current_epoch: int, logger_log_dir: str) -> float:
        """
        Evaluate all MetaWorld tasks and return universal accuracy.

        Args:
            current_epoch: Current training epoch
            logger_log_dir: Directory for saving logs and metrics

        Returns:
            Universal accuracy metric (mean success rate across all tasks)
        """
        if not self.eval_enabled:
            return 0.0

        print(f"Running MetaWorld evaluation at epoch {current_epoch}")
        print(f"Evaluating {len(self.task_list)} tasks")

        all_task_metrics = []

        # Create video save directory
        video_save_dir = None
        if self.metaworld_eval_config.get("save_video", False):
            video_save_dir = os.path.join(logger_log_dir, self.metaworld_eval_config["video_dir"])
            os.makedirs(video_save_dir, exist_ok=True)

        # Create metrics save directory early for checking existing metrics
        metrics_save_dir = os.path.join(logger_log_dir, self.metaworld_eval_config["metrics_dir"])

        # Evaluate each task
        for task_idx, (task_name, task_cls) in enumerate(tqdm(self.task_list, desc="Evaluating MetaWorld tasks")):
            print(f"\n[{task_idx + 1}/{len(self.task_list)}] Evaluating task: {task_name}")

            # Check if metrics already exist for this task
            if self._check_task_metrics_exist(task_idx, task_name, current_epoch, metrics_save_dir):
                print(f"Task {task_name} - Metrics already exist, skipping evaluation")

                # Load existing metrics and add to all_task_metrics
                task_name_safe = task_name.replace(" ", "_").replace("/", "_")
                existing_metrics_file = os.path.join(metrics_save_dir, f"task_{task_idx}_{task_name_safe}_epoch_{current_epoch}.json")
                try:
                    with open(existing_metrics_file, 'r') as f:
                        existing_data = json.load(f)
                        if 'task_metrics' in existing_data:
                            all_task_metrics.append(existing_data['task_metrics'])
                            print(f"Task {task_name} - Loaded existing metrics - Success rate: {existing_data['task_metrics']['success_rate']:.3f}")
                except Exception as e:
                    print(f"Warning: Failed to load existing metrics for task {task_name}: {e}")
                continue

            task_info = {
                "task_name": task_name,
                "task_class": task_cls,
                "video_save_dir": video_save_dir,
                "current_epoch": current_epoch,
            }

            task_metrics = self.evaluate_single_task(task_idx, task_info)
            all_task_metrics.append(task_metrics)

            # Save metrics instantly for this task
            self._save_single_task_metrics(task_metrics, current_epoch, metrics_save_dir)

            print(f"Task {task_name} - Success rate: {task_metrics['success_rate']:.3f}")

        # Compute overall metrics and save
        if all_task_metrics:
            overall_success_rate = self._compute_universal_accuracy(all_task_metrics)

            metrics_data = {
                "epoch": current_epoch,
                "overall_success_rate": overall_success_rate,
                "num_tasks": len(all_task_metrics),
                "task_metrics": all_task_metrics,
                "config": self.metaworld_eval_config,
            }

            # Save overall metrics to JSON
            os.makedirs(metrics_save_dir, exist_ok=True)
            metrics_file = os.path.join(metrics_save_dir, f"metaworld_metrics_epoch_{current_epoch}.json")
            self._save_metrics_to_json(metrics_data, metrics_file)

            print(f"\nMetaWorld evaluation completed!")
            print(f"Overall success rate: {overall_success_rate:.3f}")
            print(f"Metrics saved to: {metrics_file}")

            return overall_success_rate

    def evaluate_single_task(self, task_id: int, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single MetaWorld task.

        Args:
            task_id: ID of the task to evaluate
            task_info: Task information dictionary

        Returns:
            Dictionary containing evaluation metrics
        """
        task_name = task_info["task_name"]
        task_class = task_info["task_class"]
        video_save_dir = task_info.get("video_save_dir")
        current_epoch = task_info.get("current_epoch", 0)

        # Configuration
        num_episodes = self.metaworld_eval_config["num_episodes_per_task"]
        max_steps = self.metaworld_eval_config["max_steps"]
        resolution = self._get_image_size()
        save_video = self.metaworld_eval_config.get("save_video", False)

        print(f"  Evaluating MetaWorld task {task_id}: {task_name} ({num_episodes} episodes, fully parallel)")

        # Generate language instructions for each episode if extra descriptions are available
        if self.extra_descriptions:
            print(f"    Using extra descriptions for task evaluation")
            episode_instructions, init_state_seeds = self._get_task_instruction_and_init_states_ids(task_name, num_episodes, use_test_split=True)
        else:
            # Use original task description for all episodes
            task_description = task_name.replace("-", " ").replace("v3", "").strip()
            episode_instructions = [task_description] * num_episodes
            init_state_seeds = np.arange(num_episodes)

        # Test-time training/adaptation
        restoration_info = None
        if self.trainer.test_time_training_enabled:
            ttt_config = self.trainer.test_time_training_config.copy()

            # Check if this task should be adapted
            tasks_to_adapt = ttt_config.get("tasks_to_adapt", [])
            should_adapt = not tasks_to_adapt or task_name in tasks_to_adapt

            if should_adapt:
                print(f"  Performing test-time training for task: {task_name}")
                ttt_config["task_name"] = [task_name]
                task_description = task_name.replace("-", " ").replace("v3", "").strip()
                ttt_config["task_description"] = task_description

                # Use the trainer's test-time training method
                with torch.enable_grad(), torch.inference_mode(False):
                    restoration_info = self.trainer.prepare_fast_adapt(ttt_config)
                    adaptation_success = self.trainer._perform_test_time_training(task_name, "meta_world")

                    if not adaptation_success:
                        print(f"  Test-time training failed for task: {task_name}")
            else:
                print(f"  Skipping test-time training for task: {task_name} (not in tasks_to_adapt)")

        # Create vectorized environment for all episodes
        envs = self._create_vectorized_env(task_name, num_episodes, seed=42)

        # Reset environments
        obs_list, info_list = envs.reset()

        # Add 20-step wait before starting evaluation
        print(f"  Waiting 20 steps for environment to settle...")
        for wait_step in range(20):
            # Execute no-op actions (zeros) for all environments
            noop_actions = [np.zeros(4) for _ in range(num_episodes)]
            obs_list, _, _, _, info_list = envs.step(noop_actions)

        # Initialize tracking for all episodes
        episode_steps = [0] * num_episodes
        done_list = [False] * num_episodes
        success_list = [False] * num_episodes
        episode_frames = [[] for _ in range(num_episodes)] if save_video else None
        prev_xyz_gripper_list = [np.zeros(4) for _ in range(num_episodes)]
        action_buffer = None

        # Run all episodes in parallel
        for step in tqdm(range(max_steps), desc="Running episodes"):
            if all(done_list):
                break

            # Get images from all environments
            images = envs.render()

            # Process images and prepare batch data for unfinished episodes
            unfinished_indices = [i for i in range(num_episodes) if not done_list[i]]
            if not unfinished_indices:
                break

            images = np.array(images)

            # Vectorized image processing
            unfinished_images = images[unfinished_indices]
            # Apply flip operations to all images at once
            flipped_images = np.flip(np.flip(unfinished_images, axis=1), axis=2).copy()

            # Vectorized image resizing
            processed_images = self._resize_image_batch(flipped_images, resolution)

            # Vectorized state extraction
            unfinished_obs = [obs_list[i] for i in unfinished_indices]
            current_xyz_gripper_batch = np.array([self._extract_xyz_gripper(obs) for obs in unfinished_obs])

            # Vectorized robot state creation (8D)
            prev_xyz_gripper_batch = np.array([prev_xyz_gripper_list[i] for i in unfinished_indices])
            robot_state_8d_batch = np.concatenate([prev_xyz_gripper_batch, current_xyz_gripper_batch], axis=1)

            # Get episode-specific instructions for unfinished episodes
            unfinished_instructions = [episode_instructions[i] for i in unfinished_indices]

            # Get action sequences from trainer for unfinished episodes
            action_sequences = self._predict_actions_batch_optimized(
                processed_images, robot_state_8d_batch, unfinished_instructions
            )

            # Temporal ensembling (action chunking)
            if self.action_smoothing:
                if action_buffer is None:
                    action_buffer = action_sequences[:, None]
                else:
                    action_buffer = torch.cat([action_sequences[:, None], action_buffer], dim=1)
                
                action_buffer = action_buffer[:, :action_sequences.shape[1]]
                buf_idx = torch.arange(action_buffer.shape[1], device=action_buffer.device)
                action_slice = action_buffer[:, buf_idx, buf_idx] 
                action_weight = F.normalize(torch.exp(-buf_idx * 0.01), dim=0, p=1)[None, :, None]
                actions_unfinished_tensor = (action_slice * action_weight).sum(dim=1)
            else:
                actions_unfinished_tensor = action_sequences[:, 0]
            
            # Unnormalize actions
            actions_unfinished = actions_unfinished_tensor.cpu().numpy()
            actions_unfinished = self._unnormalize_actions_from_minus_one_one(actions_unfinished)

            # Save frames for video and update previous states (optimized)
            if episode_frames is not None:
                for idx, i in enumerate(unfinished_indices):
                    episode_frames[i].append(processed_images[idx])  # Remove unnecessary .copy()

            # Update previous states for next timestep (vectorized)
            for idx, i in enumerate(unfinished_indices):
                prev_xyz_gripper_list[i] = current_xyz_gripper_batch[idx]  # Remove unnecessary .copy()

            # Create full action list (including dummy actions for finished episodes)
            actions = []
            action_idx = 0
            for i in range(num_episodes):
                if not done_list[i]:
                    actions.append(actions_unfinished[action_idx])
                    action_idx += 1
                else:
                    actions.append(np.zeros(4))  # MetaWorld typically uses 4D actions

            # Execute actions
            obs_list, reward_list, terminated_list, truncated_list, info_list = envs.step(actions)

            # Update episode metrics
            for i in range(num_episodes):
                if not done_list[i]:
                    episode_steps[i] += 1

                    # Check for success and done conditions
                    success = info_list["success"][i] if "success" in info_list else False
                    terminated = (
                        terminated_list[i] if isinstance(terminated_list, (list, np.ndarray)) else terminated_list
                    )
                    truncated = truncated_list[i] if isinstance(truncated_list, (list, np.ndarray)) else truncated_list

                    if success or terminated or truncated:
                        done_list[i] = True
                        success_list[i] = success

            # Update action buffer to remove finished episodes
            if action_buffer is not None:
                still_unfinished_mask = [not done_list[i] for i in unfinished_indices]
                still_unfinished_mask = torch.tensor(still_unfinished_mask, device=self.device)
                action_buffer = action_buffer[still_unfinished_mask]

        # Clean up environment
        envs.close()

        # Restore model state after test-time training
        if restoration_info is not None:
            self.trainer.restore_adaptation(restoration_info)

        # Save videos if enabled
        if episode_frames and save_video and video_save_dir:
            self.save_episode_videos(episode_frames, success_list, task_name, video_save_dir, current_epoch)

        # Compute and return metrics
        return self.compute_task_metrics(task_id, task_name, success_list, episode_steps, num_episodes, 
                                       episode_instructions=episode_instructions, global_rank=None)

    def _predict_actions_batch_optimized(
        self, processed_images: np.ndarray, robot_state_8d_batch: np.ndarray, task_descriptions: List[str]
    ) -> torch.Tensor:
        """
        Optimized prediction of action sequences for a batch of observations using the trainer.

        Performance optimizations:
        - Eliminates intermediate list of dictionaries (batch_data)
        - Combines tensor operations (permute + normalize)
        - Pre-computes batch size
        - Returns torch.Tensor directly
        - Reduces memory allocations and copying

        Args:
            processed_images: Batch of processed images (N, H, W, C)
            robot_state_8d_batch: Batch of robot states (N, 8)
            task_descriptions: List of task description strings for each episode

        Returns:
            Predicted action sequences as torch.Tensor (N, horizon, action_dim)
        """
        if processed_images.shape[0] == 0:
            return torch.tensor([]).to(self.device)

        # Apply albumentations transform to match dataloader (applied per-image)
        transformed_images = []
        for i in range(processed_images.shape[0]):
            transformed = self.image_transform(image=processed_images[i])["image"]
            transformed_images.append(transformed)
        
        images_tensor = torch.stack(transformed_images).to(self.device)

        batch = {
            "images": images_tensor,
            "task_descriptions": task_descriptions,  # Use the provided list of descriptions
            "states": torch.from_numpy(robot_state_8d_batch).float().to(self.device),
        }

        # Get actions from trainer
        with torch.no_grad(), torch.inference_mode():
            self.trainer.eval()
            action_sequences = self.trainer.predict_step(batch, batch_idx=0)
        
        return action_sequences

    def compute_task_metrics(
        self, task_id: int, task_name: str, success_list: List[bool],
        episode_steps: List[int], num_episodes: int, 
        episode_instructions: Optional[List[str]] = None, global_rank: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute evaluation metrics for a task, including language instruction tracking."""
        metrics = {
            "task_id": task_id,
            "task_name": task_name,
            "success_rate": np.mean(success_list),
            "avg_episode_length": np.mean(episode_steps),
            "num_episodes": num_episodes,
        }

        # Add episode-level details if language instructions are tracked
        if episode_instructions is not None:
            episode_details = []
            for i in range(len(success_list)):
                episode_detail = {
                    "episode_idx": i,
                    "success": bool(success_list[i]),
                    "episode_steps": int(episode_steps[i]),
                    "language_instruction": episode_instructions[i] if i < len(episode_instructions) else "unknown"
                }
                episode_details.append(episode_detail)

            metrics["episode_details"] = episode_details

            # Add statistics about language instruction usage
            if self.extra_descriptions:
                unique_instructions = list(set(episode_instructions))
                metrics["num_unique_instructions"] = len(unique_instructions)
                metrics["unique_instructions"] = unique_instructions

                # Success rate per unique instruction
                instruction_stats = {}
                for instruction in unique_instructions:
                    instruction_episodes = [i for i, instr in enumerate(episode_instructions) if instr == instruction]
                    instruction_successes = [success_list[i] for i in instruction_episodes]
                    instruction_stats[instruction] = {
                        "num_episodes": len(instruction_episodes),
                        "success_rate": np.mean(instruction_successes) if instruction_successes else 0.0,
                        "episode_indices": instruction_episodes
                    }
                metrics["instruction_statistics"] = instruction_stats

        if global_rank is not None:
            metrics["global_rank"] = global_rank

        return metrics

"""
LIBERO environment evaluator class.

This module provides evaluation functionality specifically for LIBERO environments.
It handles LIBERO-specific setup, episode execution, and metric computation.

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
  data_root_path: "./data/libero-original"
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
"""

import os
import sys
import re
import json
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import random
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

# Add libero path to sys.path
libero_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "modified_libero")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

# LIBERO imports
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils import get_libero_path
from dataloader.utils import quat2axisangle

from .base_evaluator import BaseEvaluator


def create_eval_image_transform(image_size: Tuple[int, int]) -> A.Compose:
    """Create image transform for evaluation (no augmentation, just resize and normalize)."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        # A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def find_task_ids_by_names(task_suite, target_names: List[str]) -> List[int]:
    """
    Find task IDs that match the given task names.

    Args:
        task_suite: LIBERO task suite object
        target_names: List of target task names

    Returns:
        List of task IDs that match the names
    """
    matching_task_ids = []

    # Get all tasks from the suite
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name.lower().strip()

        # Check if this task's name matches any of the target names
        for target_name in target_names:
            target_lower = target_name.lower().strip()

            if task_name == target_lower:
                matching_task_ids.append(task_id)
                print(f"  Exact match: Task {task_id} - '{task_name}'")
                break

    return matching_task_ids


def initialize_libero_components(config: Dict[str, Any]) -> Tuple[Dict, Any, str, str]:
    """Initialize LIBERO benchmark and task suite components."""
    if not config["enabled"]:
        return None, None, None, None

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[config["task_suite_name"]]()
    camera_name = config["camera_name"]
    camera_name_image = camera_name + "_image"

    return benchmark_dict, task_suite, camera_name, camera_name_image


def preprocess_observation(obs: Dict[str, Any], camera_name_image: str, device: torch.device, image_transform: A.Compose) -> Dict[str, torch.Tensor]:
    """Preprocess observation for model input."""
    processed_obs = {}

    # Process main camera image
    image = obs[camera_name_image]
    # Flip image for proper orientation (match dataloader)
    image = np.flipud(image)
    image = np.fliplr(image)
    image = image.copy()

    
    # Apply transforms and convert to tensor
    image = image_transform(image=image)["image"].float()
    # Add batch dimension
    image = image.unsqueeze(0)
    processed_obs["image"] = image

    # Process gripper camera image
    gripper_image_key = "robot0_eye_in_hand_image"
    if gripper_image_key in obs:
        gripper_image = obs[gripper_image_key]
        
        # Apply transforms and convert to tensor
        gripper_image = image_transform(image=gripper_image)["image"].float()
        # Add batch dimension
        gripper_image = gripper_image.unsqueeze(0)
        processed_obs["gripper_image"] = gripper_image

    # Process robot state
    robot_state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    robot_state = torch.from_numpy(robot_state).to(device).float()
    robot_state = robot_state.unsqueeze(0)  # Add batch dimension
    processed_obs["state"] = robot_state

    return processed_obs


def setup_libero_environment(
    task_bddl_file: str, camera_name: str, num_parallel: int, img_resolution: int
) -> SubprocVectorEnv:
    """Setup LIBERO parallel environments.

    Args:
        task_bddl_file: Path to BDDL task file
        camera_name: Name of camera for observation
        num_parallel: Number of parallel environments to create

    Returns:
        Parallel vector environment
    """
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": img_resolution,
        "camera_widths": img_resolution,
        "camera_names": [camera_name, "robot0_eye_in_hand"],  # Add gripper camera
        "render_gpu_device_id": 0,
    }

    env_fns = [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_parallel)]
    return SubprocVectorEnv(env_fns)


def initialize_batch_tracking(
    num_parallel: int, save_video: bool
) -> Tuple[List[int], List[bool], List[bool], Optional[List[List]]]:
    """Initialize tracking variables for parallel batch evaluation.

    Args:
        num_parallel: Number of parallel environments
        save_video: Whether to save video frames

    Returns:
        Tuple of tracking lists for parallel environments
    """
    episode_steps = [0] * num_parallel
    done_list = [False] * num_parallel
    success_list = [False] * num_parallel
    episode_frames = [[] for _ in range(num_parallel)] if save_video else None

    return episode_steps, done_list, success_list, episode_frames


def record_frame(obs: Dict[str, Any], camera_name_image: str) -> Optional[np.ndarray]:
    """Process and return a frame for video recording."""
    if camera_name_image not in obs:
        return None

    frame = obs[camera_name_image]
    if not isinstance(frame, np.ndarray):
        return None

    # Ensure frame is in the right format (RGB, uint8)
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

    # Flip image for proper orientation
    frame = np.flipud(frame)
    frame = np.fliplr(frame)

    return frame


class LiberoEvaluator(BaseEvaluator):
    """
    LIBERO environment evaluator class.
    
    This class handles evaluation for LIBERO environments, including:
    - Single GPU evaluation only (no multi-GPU distribution)
    - Video recording and saving
    - Metrics computation and JSON saving
    - Support for multiple task suites
    """

    def __init__(self, trainer, eval_config: Dict[str, Any]):
        """
        Initialize LIBERO evaluator.
        
        Args:
            trainer: The trainer instance to use for prediction
            eval_config: LIBERO evaluation configuration
        """
        super().__init__(trainer, eval_config)

        # Initialize LIBERO components
        self._initialize_libero_components()

    def _setup_eval_config(self):
        """Setup LIBERO evaluation configuration with defaults."""
        from .base_evaluator import setup_eval_config
        self.eval_config = setup_eval_config(self.eval_config)
        self.eval_enabled = self.eval_config["enabled"]

    def _initialize_libero_components(self):
        """Initialize LIBERO benchmark and task suite components."""
        if not self.eval_enabled:
            return

        task_suite_name = self.eval_config["task_suite_name"]

        # Single suite evaluation
        self.eval_suite_names = [task_suite_name]
        (self.benchmark_dict, self.task_suite, self.camera_name, self.camera_name_image) = initialize_libero_components(
            self.eval_config
        )
        self.task_suites = {task_suite_name: self.task_suite}
        
        # Initialize image transform for evaluation
        image_size = self._get_image_size()
        self.image_transform = create_eval_image_transform((image_size, image_size))
        
        # Load extra descriptions if provided
        self.extra_descriptions = None
        if "extra_desc_json_path" in self.eval_config and self.eval_config["extra_desc_json_path"]:
            self.extra_descriptions = self._load_extra_descriptions(self.eval_config["extra_desc_json_path"])
        
        print(f"LIBERO evaluator initialized with task suite: {task_suite_name}")

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
            List of task instruction strings for each episode
        """
        if not self.extra_descriptions:
            return [task_name] * num_episodes, np.arange(num_episodes)
        
        # Convert task description to underscore format for matching
        task_underscore = task_name.replace(" ", "_")
        
        # Find matching task in extra descriptions by checking if JSON key ends with our task
        matching_task = None
        for json_task_name in self.extra_descriptions.keys():
            # Check if the JSON key ends with our task (after scene prefix)
            if json_task_name.endswith(task_underscore):
                matching_task = json_task_name
                break
        
        assert matching_task, f"No extra description found for task '{task_name}'"
        
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

                print(f"  init_states_ids: {init_states_ids}")
                return [instructions[idx] for idx in indices], init_states_ids
        
        # Fallback to original task description
        print(f"Warning: No {split_key} instructions found for task '{matching_task}', using original")
        return [task_name] * num_episodes, np.arange(num_episodes)

    def _get_image_size(self) -> int:
        """Get image size from trainer's data configuration."""
        if hasattr(self.trainer, 'data_config') and self.trainer.data_config:
            image_size = self.trainer.data_config.get('image_size', [128, 128])
            if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                # Use the first dimension (assuming square images or use minimum)
                return min(image_size[0], image_size[1])
            elif isinstance(image_size, int):
                return image_size
        
        # Default fallback
        print("Warning: Could not read image_size from data config, using default 128")
        return 128

    def _process_task_description(self, task_name: str, task_description: str, suite_name: str) -> str:
        """
        Process task description based on include_scene_name configuration.
        
        Args:
            task_name: Original task name from LIBERO benchmark
            task_description: Original task description from LIBERO benchmark  
            suite_name: Name of the task suite
            
        Returns:
            Processed task description
        """
        # Get include_scene_name setting from trainer's data config
        include_scene_name = False
        if hasattr(self.trainer, 'data_config') and self.trainer.data_config:
            include_scene_name = self.trainer.data_config.get('include_scene_name', False)
        
        if include_scene_name:
            # Extract scene name from task_name including scene number (e.g., KITCHEN_SCENE1_...)
            scene_match = re.match(r"^([A-Z_]+_SCENE\d+)_", task_name)
            if scene_match:
                scene_name = scene_match.group(1).lower().replace("_", " ")
                return f"Scene: {scene_name}. Task: {task_description}."
            else:
                # Fallback: if no SCENE pattern found, use the cleaned up task_name
                return task_name.lower().replace("_", " ")
        else:
            # Use the original task description
            return task_description

    def create_test_time_dataset(self, task_name: str, suite_name: str, data_root_path: str, horizon: int = 20, max_demos: Optional[int] = None):
        """
        Create a per-task dataset for test-time training.
        
        Args:
            task_name: Name of the task to create dataset for
            suite_name: Name of the task suite (e.g., 'libero_spatial')
            data_root_path: Root path to LIBERO data
            horizon: Action sequence horizon
            max_demos: Maximum number of demos to load
            
        Returns:
            LiberoOriginalPerTaskDataset instance
        """
        from dataloader.libero_original_dataloader import LiberoOriginalPerTaskDataset, LiberoOriginalPerTaskDataConfig
        
        # Get include_scene_name setting from trainer's data config
        include_scene_name = False
        if hasattr(self.trainer, 'data_config') and self.trainer.data_config:
            include_scene_name = self.trainer.data_config.get('include_scene_name', False)
        
        config = LiberoOriginalPerTaskDataConfig(
            task_suite_name=suite_name,
            data_root_path=data_root_path,
            split="train",  # Use training data for test-time adaptation
            task_name=task_name,
            horizon=horizon,
            max_demos=max_demos,
            debug=False,
            include_scene_name=include_scene_name
        )
        
        try:
            dataset = LiberoOriginalPerTaskDataset(config)
            return dataset
        except Exception as e:
            print(f"Failed to create dataset for task '{task_name}': {e}")
            return None

    def evaluate_all_tasks(self, current_epoch: int, logger_log_dir: str) -> float:
        """
        Evaluate all LIBERO tasks and return universal accuracy.
        
        Args:
            current_epoch: Current training epoch
            logger_log_dir: Directory for saving logs and metrics
            
        Returns:
            Universal accuracy metric (mean success rate across all tasks)
        """
        if not self.eval_enabled:
            return 0.0

        print(f"Running LIBERO evaluation at epoch {current_epoch}")

        all_task_metrics = []

        # Evaluate each task suite
        for suite_name in self.eval_suite_names:
            task_suite = self.task_suites[suite_name]
            print(f"Evaluating task suite: {suite_name}")

            # Determine which tasks to evaluate
            config = self.eval_config
            if config.get("split_txt") is not None:
                # Load tasks from split file
                split_txt_path = config["split_txt"]
                target_task_names = []
                with open(split_txt_path, "r") as f:
                    for line in f:
                        line = line.strip().replace(".hdf5", "")
                        if line and not line.startswith("#"):
                            target_task_names.append(line)

                task_ids_to_eval = find_task_ids_by_names(task_suite, target_task_names)
                print(f"  Evaluating {len(task_ids_to_eval)} tasks from split file: {task_ids_to_eval}")
            else:
                # Evaluate subset of tasks
                max_tasks = min(config["max_tasks"], task_suite.n_tasks)
                task_ids_to_eval = list(range(max_tasks))

            # Create metrics save directory early for checking existing metrics
            metrics_save_dir = os.path.join(logger_log_dir, self.eval_config["metrics_dir"])

            # Evaluate each task
            for task_id in task_ids_to_eval:
                task = task_suite.get_task(task_id)
                task_info = {
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_description": task.language,
                    "suite_name": suite_name,
                    "task_suite": task_suite,
                }

                # Check if metrics already exist for this task
                if self._check_task_metrics_exist(task_id, task.name, current_epoch, metrics_save_dir):
                    print(f"  Task {task_id} ({task.name}): Metrics already exist, skipping evaluation")
                    
                    # Load existing metrics and add to all_task_metrics
                    task_name_safe = task.name.replace(" ", "_").replace("/", "_")
                    existing_metrics_file = os.path.join(metrics_save_dir, f"task_{task_id}_{task_name_safe}_epoch_{current_epoch}.json")
                    try:
                        with open(existing_metrics_file, 'r') as f:
                            existing_data = json.load(f)
                            if 'task_metrics' in existing_data:
                                all_task_metrics.append(existing_data['task_metrics'])
                                print(f"  Task {task_id} ({task.name}): Loaded existing metrics - Success rate = {existing_data['task_metrics']['success_rate']:.2f}")
                    except Exception as e:
                        print(f"  Warning: Failed to load existing metrics for task {task_id}: {e}")
                    continue

                metrics = self.evaluate_single_task(task_id, task_info)

                if metrics:
                    all_task_metrics.append(metrics)
                    
                    # Save metrics instantly for this task
                    self._save_single_task_metrics(metrics, current_epoch, metrics_save_dir)
                    
                    print(
                        f"  Task {task_id} ({metrics['task_name']}): "
                        f"Success rate = {metrics['success_rate']:.2f}, "
                        f"Avg length = {metrics['avg_episode_length']:.1f}"
                    )

        # Save all metrics to JSON
        if all_task_metrics:
            overall_success_rate = self._compute_universal_accuracy(all_task_metrics)
            overall_avg_length = np.mean([m["avg_episode_length"] for m in all_task_metrics])

            # Save overall metrics using base class method
            self.save_evaluation_metrics(all_task_metrics, current_epoch, metrics_save_dir, global_rank=None)

            print(
                f"LIBERO evaluation completed: "
                f"Overall success rate = {overall_success_rate:.2f}, "
                f"Overall avg length = {overall_avg_length:.1f}"
            )

            return overall_success_rate
        else:
            print("No LIBERO evaluation metrics collected")
            return 0.0

    def evaluate_single_task(self, task_id: int, task_info: Dict[str, Any], video_dir: str = "videos") -> Dict[str, Any]:
        """
        Evaluate a single LIBERO task.
        
        Args:
            task_id: ID of the task to evaluate
            task_info: Task information dictionary
            
        Returns:
            Dictionary containing evaluation metrics
        """
        suite_name = task_info["suite_name"]
        task_suite = task_info["task_suite"]
        task_name = task_info["task_name"]
        task_description = task_info["task_description"]

        config = self.eval_config
        num_episodes = config["num_episodes_per_task"]
        num_parallel = min(config["num_parallel"], num_episodes)  # Single GPU, no distribution
        max_steps = config["max_steps"]
        save_video = config.get("save_video", False)

        print(f"  Evaluating LIBERO task {task_id}: {task_name} ({num_episodes} episodes, {num_parallel} parallel)")

        # Process task description based on include_scene_name configuration
        processed_task_description = self._process_task_description(task_name, task_description, suite_name)
        
        # Generate language instructions for each episode if extra descriptions are available
        if self.extra_descriptions:
            print(f"    Using extra descriptions for task evaluation")
            episode_instructions, init_states_ids = self._get_task_instruction_and_init_states_ids(task_name, num_episodes, use_test_split=True)
            # Apply scene name processing to extra descriptions if needed
            if hasattr(self.trainer, 'data_config') and self.trainer.data_config and self.trainer.data_config.get('include_scene_name', False):
                episode_instructions = [self._process_task_description(task_name, instr, suite_name) for instr in episode_instructions]
        else:
            # Use processed task description for all episodes
            episode_instructions = [processed_task_description] * num_episodes
            init_states_ids = np.arange(num_episodes)

        # Test-time training/adaptation
        if self.trainer.test_time_training_enabled:
            ttt_config = self.trainer.test_time_training_config.copy()

            # Check if this task should be adapted
            tasks_to_adapt = ttt_config.get("tasks_to_adapt", [])
            should_adapt = not tasks_to_adapt or task_name in tasks_to_adapt

            ttt_config["task_name"] = [task_name]
            ttt_config["task_description"] = processed_task_description

            # Use the trainer's test-time training method
            with torch.enable_grad(), torch.inference_mode(False):
                restoration_info = self.trainer.prepare_fast_adapt(ttt_config)
                adaptation_success = self.trainer._perform_test_time_training(task_name, suite_name)

        # Setup task environment
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        # Get initial states for reproducibility
        if "ood" not in suite_name:
            init_states = task_suite.get_task_init_states(task_id)
            init_states_ids = init_states_ids.astype(int) % len(init_states)
            init_states = [init_states[i] for i in init_states_ids]

        # Collect results from all episodes
        all_episode_steps = []
        all_success_list = []
        all_episode_frames = []
        all_episode_instructions = []  # Track which instruction was used for each episode

        # Run episodes in batches of num_parallel
        for batch_start in range(0, num_episodes, num_parallel):
            batch_end = min(batch_start + num_parallel, num_episodes)
            batch_size = batch_end - batch_start

            print(f"    Running batch {batch_start//num_parallel + 1}/{(num_episodes + num_parallel - 1)//num_parallel}: episodes {batch_start}-{batch_end-1}")

            # Get instructions for this batch
            batch_instructions = episode_instructions[batch_start:batch_end]

            # Setup environment for this batch
            # Get image size from data configuration
            image_size = self._get_image_size()
            env = setup_libero_environment(task_bddl_file, self.camera_name, batch_size, image_size)

            # Setup seeds and initial states
            batch_seeds = [7] * batch_size
            env.seed(batch_seeds)
            obs_list = env.reset()

            if "ood" not in suite_name:
                # Set initial states for this batch
                obs_list = env.set_init_state(init_states[batch_start:batch_end])

            # Initialize tracking for this batch
            episode_steps, done_list, success_list, episode_frames = initialize_batch_tracking(batch_size, save_video)

            # Stabilization period
            stabilization_steps = 10
            action = np.zeros(self.trainer.action_dim)
            action[-1] = -1
            action = [action] * batch_size
            for _ in range(stabilization_steps):
                obs_list, _, _, _ = env.step(action)

            # Record initial frames after stabilization
            if episode_frames is not None:
                for i in range(batch_size):
                    frame = record_frame(obs_list[i], self.camera_name_image)
                    if frame is not None:
                        episode_frames[i].append(frame)

            pbar = tqdm(total=max_steps, desc="Running episodes", leave=False)
            action_buffer = None
            done_list = [False] * batch_size

            # Run episodes in this batch
            while not all(done_list) and ((curr_steps := max(episode_steps)) < max_steps):
                # Generate actions for unfinished episodes
                unfinished_indices = [i for i in range(batch_size) if not done_list[i]]
                actions = []

                pbar.update(1)

                if unfinished_indices:
                    # Preprocess observations and batch them
                    processed_obs_list = [preprocess_observation(obs_list[i], self.camera_name_image, self.device, self.image_transform) for i in unfinished_indices]
                    batched_images = torch.cat([obs["image"] for obs in processed_obs_list], dim=0).to(self.device)
                    batched_gripper_images = torch.cat([obs["gripper_image"] for obs in processed_obs_list], dim=0).to(self.device)
                    batched_states = torch.cat([obs["state"] for obs in processed_obs_list], dim=0).to(self.device)
                    
                    # Use the specific instruction for each unfinished episode
                    task_descriptions = [batch_instructions[i] for i in unfinished_indices]

                    batch = {
                        "images": batched_images,
                        "gripper_images": batched_gripper_images,
                        "task_descriptions": task_descriptions,
                        "states": batched_states,
                    }

                    # Generate actions using trainer.predict_step
                    with torch.no_grad(), torch.inference_mode():
                        self.trainer.eval()
                        action_sequences = self.trainer.predict_step(batch, batch_idx=0)
                        # first_actions = action_sequences[:, 0].cpu().numpy()

                        if action_buffer is None:
                            action_buffer = action_sequences[:, None]
                        else:
                            action_buffer = torch.cat([action_sequences[:, None], action_buffer], dim=1)
                        action_buffer = action_buffer[:, :action_sequences.shape[1]]
                        buf_idx = torch.arange(action_buffer.shape[1], device=action_buffer.device)
                        action_slice = action_buffer[:, buf_idx, buf_idx] 
                        action_weight = F.normalize(torch.exp(-buf_idx * 0.01), dim=0, p=1)[None, :, None]
                        action = (action_slice * action_weight).sum(dim=1)
                        first_actions = action.cpu().numpy()

                # Assign actions (including dummy actions for finished episodes)
                action = np.zeros((batch_size, self.trainer.action_dim))
                unfinished_idx = np.array(unfinished_indices, dtype=int)
                action[unfinished_idx] = first_actions

                next_obs_list, reward_list, cur_done_list, info_list = env.step(action)

                done_list = [done_list[i] or cur_done_list[i] for i in range(batch_size)]
                
                if action_buffer is not None:
                    keep_mask = ~np.array(done_list, dtype=bool)[unfinished_idx]
                    keep_mask = torch.from_numpy(keep_mask).to(device=self.device)
                    action_buffer = action_buffer[keep_mask]

                # Update episode metrics
                for i in range(batch_size):
                    if not done_list[i]:
                        episode_steps[i] += 1
                    elif episode_steps[i] < max_steps:
                        success_list[i] = True

                obs_list = next_obs_list

                # Record frames for ongoing episodes
                if episode_frames is not None:
                    for i in range(batch_size):
                        if not done_list[i]:
                            frame = record_frame(obs_list[i], self.camera_name_image)
                            if frame is not None:
                                episode_frames[i].append(frame)

            pbar.close()
            env.close()

            # Collect results from this batch
            all_episode_steps.extend(episode_steps)
            all_success_list.extend(success_list)
            all_episode_instructions.extend(batch_instructions)
            if episode_frames is not None:
                all_episode_frames.extend(episode_frames)

        if self.trainer.test_time_training_enabled:
            self.trainer.restore_adaptation(restoration_info)

        # Save videos if enabled
        if all_episode_frames and save_video:
            video_dir = config.get("video_dir", "videos")
            video_save_dir = os.path.join(self.trainer.logger.log_dir, video_dir, suite_name)
            self.save_episode_videos(all_episode_frames, all_success_list, task_name, video_save_dir, self.trainer.current_epoch, global_rank=None)

        # Compute and return metrics (no global_rank since single GPU)
        return self.compute_task_metrics(task_id, task_name, all_success_list, all_episode_steps, num_episodes, 
                                       episode_instructions=all_episode_instructions, global_rank=None)

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

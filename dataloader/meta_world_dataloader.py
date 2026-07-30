"""
Meta-World dataloader with HDF5 file loading and same interface as LiberoDataset.
"""

import os
import glob
import re
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import threading
# Removed joblib import for serial loading


from dataloader.utils import quat2axisangle


@dataclass
class MetaWorldDataConfig:
    """Configuration for Meta-World dataset loading."""

    # Dataset parameters
    task_suite_name: str = "meta_world"
    data_root_path: str = "./data/meta-world"
    split: str = "train"
    debug: bool = False

    # Image processing
    image_size: Tuple[int, int] = (224, 224)

    # Action sequence parameters
    horizon: int = 8

    # Split file - only one split file per dataset instance
    split_file: Optional[str] = None  # Path to split txt file

    state_dim: int = 8
    action_dim: int = 4

    # action_norm_stats_path: Optional[str] = 'config/data/meta-world_splits/action_norm_stats.pkl'
    action_norm_stats_path: Optional[str] = ''

    image_norm_type: str = 'imagenet'
    use_image_aug: bool = False

@dataclass
class MetaWorldPerTaskDataConfig(MetaWorldDataConfig):
    """Configuration for Meta-World per-task dataset loading."""

    # Task-specific parameters
    task_name: str = ""  # Specific task name to load (e.g., "door-close-v3")
    max_demos: Optional[int] = None  # Maximum number of demos to load for this task


class MetaWorldDataset(Dataset):
    """Meta-World dataset with HDF5 file loading and on-demand image loading."""

    def __init__(self, config: MetaWorldDataConfig):
        self.config = config
        self.data_root_path = Path(config.data_root_path)
        self.task_suite_name = config.task_suite_name
        self.split = config.split
        self.image_size = config.image_size
        self.horizon = config.horizon
        self.action_min = None
        self.action_max = None
        self.action_norm_stats_path = config.action_norm_stats_path

        
        print(f"\033[91m{config}\033[0m")

        self._load_action_norm_stats()

        if config.image_norm_type == 'imagenet':
            img_norm_transform = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0)
        elif config.image_norm_type == 'standard':
            img_norm_transform = A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0)
        else: 
            raise ValueError(f"Invalid image normalization type: {config.image_norm_type}")

        if self.config.use_image_aug:
            self.image_transform = A.Compose([
                A.RandomResizedCrop(
                    size=(self.image_size[0], self.image_size[1]),
                    scale=(0.9, 1.0),
                    ratio=(1.0, 1.0),
                    p=1.0
                ),
                A.ColorJitter(
                    brightness_range=(0.95, 1.05),
                    contrast_range=(0.95, 1.05),
                    saturation_range=(0.95, 1.05),
                    hue_range=(-0.02, 0.02),
                    p=1.0
                ),
                img_norm_transform,
                ToTensorV2(),
            ])
        else:
            self.image_transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),
                img_norm_transform,
                ToTensorV2(),
            ])

        # Load split file if provided
        self.split_episodes = self._load_split_file(config.split_file) if config.split_file else None

        # Thread-local storage for HDF5 file handles
        self._local = threading.local()

        # Initialize dataset loading (metadata only)
        self._init_metadata_loading()

        print(f"Initialized Meta-World dataset: {self.task_suite_name}")
        print(f"Split: {self.split}, Episodes: {self.total_episodes}")
        print(f"Total transitions: {self.total_transitions}")
        print("Loading mode: On-demand with persistent file handles")

    def _extract_task_description(self, filename: str) -> str:
        """Extract clean task description from HDF5 filename."""
        # Remove .hdf5 extension and _demo suffix
        base_name = filename.replace(".hdf5", "").replace("_demo", "")

        # Replace hyphens with spaces and format nicely
        task_description = base_name.replace("-", " ").replace("v3", "").strip()

        return task_description

    def _load_split_file(self, split_file_path: str) -> Optional[List[str]]:
        """Load task names from a split file."""
        if not split_file_path or not os.path.exists(split_file_path):
            return None

        with open(split_file_path, "r") as f:
            # Load task names (without .hdf5 extension)
            task_names = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Loaded {len(task_names)} tasks from split file: {split_file_path}")
        return task_names

    def _get_file_handle(self, file_path: Path) -> h5py.File:
        """Get or create HDF5 file handle for current worker/thread."""
        if not hasattr(self._local, 'file_handles'):
            self._local.file_handles = {}
        
        file_path_str = str(file_path)
        if file_path_str not in self._local.file_handles:
            self._local.file_handles[file_path_str] = h5py.File(file_path, 'r', swmr=True)
        
        return self._local.file_handles[file_path_str]

    def _load_action_norm_stats(self):
        """Load action normalization statistics from pickle file."""
        if self.action_norm_stats_path and os.path.exists(self.action_norm_stats_path):
            with open(self.action_norm_stats_path, "rb") as f:
                stats = pickle.load(f)
                self.action_min = stats["action_min"]
                self.action_max = stats["action_max"]
            print(f"Loaded action norm stats from {self.action_norm_stats_path}")
        else:
            print(f"Warning: Action norm stats path {self.action_norm_stats_path} not found.")

    def _normalize_actions_to_minus_one_one(self, actions: np.ndarray) -> np.ndarray:
        if self.action_min is None or self.action_max is None:
            return actions

        denom = self.action_max - self.action_min
        denom = np.where(denom == 0, 1.0, denom)
        actions_01 = (actions - self.action_min) / denom
        return actions_01 * 2.0 - 1.0

    def _init_metadata_loading(self):
        """Initialize metadata loading - store file paths and episode info without loading images."""
        self.all_episodes = []
        self.transition_to_episode = []  # Maps transition index to (episode_idx, step_idx)

        transition_count = 0

        # Determine which tasks to load based on split
        target_tasks = None
        if self.split_episodes:
            target_tasks = set(self.split_episodes)
            print(f"Using split file with {len(target_tasks)} tasks for {self.split} split")

        # Get all HDF5 files in the data directory
        hdf5_files = list(self.data_root_path.glob("*_demo.hdf5"))

        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in {self.data_root_path}")

        print(f"Loading Meta-World dataset metadata...")
        print(f"Found {len(hdf5_files)} HDF5 files")

        # Process files serially
        processed_files = 0
        for hdf5_file in hdf5_files:
            # Extract task name from filename
            task_name = hdf5_file.stem.replace("_demo", "")

            # Skip task if not in target split
            if target_tasks and task_name not in target_tasks:
                continue

            # Extract task description from filename
            task_description = self._extract_task_description(hdf5_file.name)

            print(f"Processing file {processed_files + 1}/{len(hdf5_files)}: {hdf5_file.name}")

            # Process each HDF5 file serially - load only metadata
            with h5py.File(hdf5_file, "r", swmr=True) as f:
                data_group = f["data"]
                demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
                
                # In debug mode, limit to fewer demos
                if self.config.debug and len(demo_keys) > 2:
                    demo_keys = demo_keys[:2]
                
                for demo_key in demo_keys:
                    demo = data_group[demo_key]

                    # Create episode identifier for tracking
                    episode_id = f"{task_name}_{demo_key}"

                    # Extract only metadata and preload states/actions (smaller data)
                    states = demo["robot_states"][:]  # Shape: (T, state_dim)
                    actions = demo["actions"][:]  # Shape: (T, action_dim)
                    episode_length = demo["obs"]["agentview_rgb"].shape[0]  # Get length without loading images

                    # Store episode metadata and file reference
                    episode = {
                        "file_path": hdf5_file,
                        "demo_key": demo_key,
                        "robot_states": states,
                        "actions": actions,
                        "episode_length": episode_length,
                        "task_description": task_description,
                        "task_name": task_name,
                        "episode_id": episode_id,
                    }

                    episode_idx = len(self.all_episodes)
                    self.all_episodes.append(episode)

                    # Map each valid starting point for horizon-length sequences
                    for step_idx in range(max(0, episode_length - self.horizon + 1)):
                        self.transition_to_episode.append((episode_idx, step_idx))
                        transition_count += 1

            processed_files += 1

            if self.config.debug and processed_files >= 3:
                break

        total_episode_count = len(self.all_episodes)
        print(f"\nMetadata loading complete: {total_episode_count} episodes processed")

        self.total_episodes = len(self.all_episodes)
        self.total_transitions = transition_count

        print(
            f"\nTotal loaded: {self.total_episodes} episodes with {self.total_transitions} total transitions"
        )
        print(f"Action normalization stats (per-dim): min={self.action_min}, max={self.action_max}")

    def __len__(self) -> int:
        """Return total number of transitions."""
        return self.total_transitions

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single transition with action sequence of length horizon."""
        episode_idx, step_idx = self.transition_to_episode[idx]
        episode = self.all_episodes[episode_idx]

        # Load image on-demand using persistent file handle
        file_handle = self._get_file_handle(episode["file_path"])
        demo = file_handle["data"][episode["demo_key"]]
        image = demo["obs"]["agentview_rgb"][step_idx]  # Load only the specific image

        state = episode["robot_states"][step_idx]
        task_description = episode["task_description"]

        # Get action sequence starting from current step
        end_idx = min(step_idx + self.horizon, episode["episode_length"])
        action_sequence = episode["actions"][step_idx:end_idx]

        # Pad action sequence if needed (when near end of episode)
        if len(action_sequence) < self.horizon:
            # Pad with the last action to reach horizon length
            last_action = action_sequence[-1] if len(action_sequence) > 0 else np.zeros_like(episode["actions"][0])
            padding_length = self.horizon - len(action_sequence)
            padding = np.tile(last_action, (padding_length, 1))
            action_sequence = np.concatenate([action_sequence, padding], axis=0)

        action_sequence = self._normalize_actions_to_minus_one_one(action_sequence)

        image = self.image_transform(image=image)["image"].float()

        # Convert to torch tensors
        state = torch.tensor(state, dtype=torch.float32)
        action_sequence = torch.tensor(action_sequence, dtype=torch.float32)  # Shape: (horizon, action_dim)
        # if self.action_min is not None and self.action_max is not None:
        #     assert action_sequence.min() >= -1 and action_sequence.max() <= 1, f"Action sequence out of range: {action_sequence.min()} {action_sequence.max()}"

        return {
            "images": image,
            "states": state,
            "actions": action_sequence,
            "task_descriptions": task_description,
            "task_ids": torch.tensor(0, dtype=torch.long),  # Simplified
        }

    def __del__(self):
        """Clean up HDF5 file handles when dataset is destroyed."""
        if hasattr(self, '_local') and hasattr(self._local, 'file_handles'):
            for handle in self._local.file_handles.values():
                try:
                    handle.close()
                except:
                    pass  # Ignore errors during cleanup


class MetaWorldPerTaskDataset(MetaWorldDataset):
    """Meta-World per-task dataset that loads only a specific task with configurable limits."""

    def __init__(self, config: MetaWorldPerTaskDataConfig):
        if not config.task_name:
            raise ValueError("task_name must be specified for MetaWorldPerTaskDataset")

        self.task_name = config.task_name
        self.max_demos = config.max_demos
        self.action_norm_stats_path = Path(config.data_root_path) / f"action_norm_stats_{self.task_name}.pkl"

        # Initialize parent class
        super().__init__(config)

        print(f"Initialized Meta-World per-task dataset for task: '{self.task_name}'")
        if self.max_demos:
            print(f"Limited to {self.max_demos} demos")

    def _init_metadata_loading(self):
        """Initialize metadata loading - store file paths and episode info without loading images."""
        self.all_episodes = []
        self.transition_to_episode = []  # Maps transition index to (episode_idx, step_idx)

        transition_count = 0
        loaded_demos_count = 0

        # Construct HDF5 filename directly from task name
        hdf5_filename = f"{self.task_name}_demo.hdf5"
        task_description = self._extract_task_description(hdf5_filename)

        # Construct full path to HDF5 file
        hdf5_file_path = self.data_root_path / hdf5_filename

        if not hdf5_file_path.exists():
            available_files = [f.name for f in self.data_root_path.glob("*.hdf5")]
            raise ValueError(
                f"Task file '{hdf5_filename}' not found in {self.data_root_path}. Available files: {available_files}"
            )

        print(f"Loading task '{self.task_name}' metadata from {hdf5_file_path}")

        # Process the HDF5 file serially - load only metadata
        with h5py.File(hdf5_file_path, "r", swmr=True) as f:
            data_group = f["data"]
            demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]

            for demo_key in demo_keys:
                # Check if we've reached the demo limit
                if self.max_demos and loaded_demos_count >= self.max_demos:
                    print(f"Reached maximum demos limit ({self.max_demos})")
                    break

                demo = data_group[demo_key]

                print(f"Loading demo {loaded_demos_count + 1}...", end="\r")

                if self.config.debug and loaded_demos_count >= 5:
                    break

                # Create episode identifier for tracking
                episode_id = f"{self.task_name}_{demo_key}"

                # Extract only metadata and preload states/actions (smaller data)
                states = demo["robot_states"][:]  # Shape: (T, state_dim)
                actions = demo["actions"][:]  # Shape: (T, action_dim)
                episode_length = demo["obs"]["agentview_rgb"].shape[0]  # Get length without loading images

                # Store episode metadata and file reference
                episode = {
                    "file_path": hdf5_file_path,
                    "demo_key": demo_key,
                    "robot_states": states,
                    "actions": actions,
                    "episode_length": episode_length,
                    "task_description": task_description,
                    "task_name": self.task_name,
                    "episode_id": episode_id,
                }

                episode_idx = len(self.all_episodes)
                self.all_episodes.append(episode)

                # Map each valid starting point for horizon-length sequences
                for step_idx in range(max(0, episode_length - self.horizon + 1)):
                    self.transition_to_episode.append((episode_idx, step_idx))
                    transition_count += 1

                loaded_demos_count += 1

        self.total_episodes = len(self.all_episodes)
        self.total_transitions = transition_count

        if self.total_episodes == 0:
            raise ValueError(f"No episodes found for task '{self.task_name}'.")

        print(
            f"\nTotal loaded for task '{self.task_name}': {self.total_episodes} episodes with {self.total_transitions} total transitions"
        )
        print(f"Action normalization stats (per-dim): min={self.action_min}, max={self.action_max}")

    def __len__(self) -> int:
        """Return total number of transitions."""
        return self.total_transitions * 10000

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return super().__getitem__(idx % self.total_transitions)

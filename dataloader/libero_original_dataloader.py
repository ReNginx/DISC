"""
LIBERO Original dataloader with HDF5 file loading and same interface as LiberoDataset.
"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import threading
import random
import torchvision

from dataloader.utils import quat2axisangle




@dataclass
class LiberoOriginalDataConfig:
    """Configuration for LIBERO Original dataset loading."""

    # Dataset parameters
    task_suite_name: str = "libero_original"
    data_root_path: str = "./data/libero-original"
    split: str = "train"
    debug: bool = False

    # Image processing
    image_size: Tuple[int, int] = (128, 128)

    # Action sequence parameters
    horizon: int = 8
    
    # Inner batch size for sampling multiple transitions from same episode
    inner_batch_size: int = 1

    # Dataset dimensions
    state_dim: int = 8
    action_dim: int = 7

    # Split file - only one split file per dataset instance
    split_file: Optional[str] = None  # Path to split txt file
    
    # Task description options
    include_scene_name: bool = False  # Whether to include scene name in task descriptions


@dataclass
class LiberoOriginalPerTaskDataConfig(LiberoOriginalDataConfig):
    """Configuration for LIBERO Original per-task dataset loading."""
    
    # Task-specific parameters
    task_name: str = ""  # Specific task name to load (e.g., "put the black bowl on the plate")
    max_demos: Optional[int] = None  # Maximum number of demos to load for this task


class LiberoOriginalDataset(Dataset):
    """LIBERO Original dataset with HDF5 file loading."""

    def __init__(self, config: LiberoOriginalDataConfig):
        self.config = config
        self.data_root_path = Path(config.data_root_path)
        self.task_suite_name = config.task_suite_name
        self.split = config.split
        self.image_size = config.image_size
        self.horizon = config.horizon

        # Image transforms with Albumentations
        # === comment below if using image aug
        # self.image_transform = A.Compose([
        #     A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),
        #     A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        #     ToTensorV2(),
        # ])
        # === comment above if using image aug
        
        # === uncomment below if using image aug
        self.image_transform = A.Compose([
            A.OneOf([
                # Augmented branch (weight 0.8):
                A.Compose([
                    # Side 0.90–0.99  => area 0.81–0.9801, keep aspect 1.0
                    A.RandomResizedCrop(
                        size=(self.image_size[0], self.image_size[1]),
                        scale=(0.81, 1.0),
                        ratio=(1.0, 1.0),
                        p=1.0
                    ),
                    A.ColorJitter(
                        brightness_range=(0.9, 1.1),
                        contrast_range=(0.9, 1.1),
                        saturation_range=(0.9, 1.1),
                        hue_range=(-0.05, 0.05),
                        p=1.0
                    ),
                ], p=0.8),
                # Base branch (weight 0.2): resize only
                A.Resize(height=self.image_size[0], width=self.image_size[1], p=0.2),
            ], p=1.0),
            # A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        # === uncomment above if using image aug

        # Load split file if provided
        self.split_episodes = self._load_split_file(config.split_file) if config.split_file else None

        # Thread-local storage for HDF5 file handles
        self._local = threading.local()

        # Initialize dataset loading (metadata only)
        self._init_metadata_loading()

        print(f"Initialized LIBERO Original dataset: {self.task_suite_name}")
        print(f"Split: {self.split}, Episodes: {self.total_episodes}")
        print(f"Total transitions: {self.total_transitions}")
        print("Loading mode: On-demand with persistent file handles")

    def _extract_task_description(self, filename: str) -> str:
        """Extract task description from HDF5 filename, optionally including scene info."""
        # Remove .hdf5 extension and _demo suffix
        base_name = filename.replace(".hdf5", "").replace("_demo", "")

        if self.config.include_scene_name:
            # Extract scene name including scene number (e.g., KITCHEN_SCENE6)
            scene_match = re.match(r"^([A-Z_]+_SCENE\d+)_", base_name)
            if scene_match:
                scene_name = scene_match.group(1).lower().replace("_", " ")
                # Extract task part (part after SCENE\d+_)
                task_part = re.sub(r"^[A-Z_]*SCENE\d+_", "", base_name)
                task_description = task_part.replace("_", " ")
                return f"Scene: {scene_name}. Task: {task_description}."
            else:
                # Fallback: if no SCENE\d+ found, just return the whole name cleaned up
                return base_name.lower().replace("_", " ")
        else:
            # Remove scene information (capital letters and numbers pattern like KITCHEN_SCENE6_)
            # This regex matches patterns like KITCHEN_SCENE6_, LIVING_ROOM_SCENE1_, etc.
            clean_name = re.sub(r"^[A-Z_]*SCENE\d+_", "", base_name)
            # Replace remaining underscores with spaces
            task_description = clean_name.replace("_", " ")

        return task_description

    def _load_split_file(self, split_file_path: str) -> Optional[List[str]]:
        """Load HDF5 filenames from a split file."""
        if not split_file_path or not os.path.exists(split_file_path):
            return None

        with open(split_file_path, "r") as f:
            # Load HDF5 filenames (with .hdf5 extension)
            hdf5_filenames = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Loaded {len(hdf5_filenames)} HDF5 files from split file: {split_file_path}")
        return hdf5_filenames

    def _get_file_handle(self, file_path: Path) -> h5py.File:
        """Get or create HDF5 file handle for current worker/thread."""
        if not hasattr(self._local, "file_handles"):
            self._local.file_handles = {}

        file_path_str = str(file_path)
        if file_path_str not in self._local.file_handles:
            self._local.file_handles[file_path_str] = h5py.File(file_path, "r", swmr=True)

        return self._local.file_handles[file_path_str]

    def _init_metadata_loading(self):
        """Initialize metadata loading - store file paths and episode info without loading images."""
        self.all_episodes = []
        self.transition_to_episode = []  # Maps transition index to (episode_idx, step_idx)

        transition_count = 0
        total_episode_count = 0

        # Determine which episodes to load based on split
        target_hdf5_files = None
        if self.split_episodes:
            target_hdf5_files = set(self.split_episodes)
            print(f"Using split file with {len(target_hdf5_files)} HDF5 files for {self.split} split")

        # Determine which task suites to load based on task_suite_name
        target_suite_names = [self.task_suite_name]
        print("Loading LIBERO Original dataset metadata...")

        # Get all suite directories and filter by target suites
        all_suite_dirs = [d for d in self.data_root_path.iterdir() if d.is_dir()]
        suite_dirs = [d for d in all_suite_dirs if d.name in target_suite_names]

        if not suite_dirs:
            available_suites = [d.name for d in all_suite_dirs]
            raise ValueError(
                f"No matching suite directories found for {target_suite_names}. " f"Available suites: {available_suites}"
            )

        print(f"Target suites: {target_suite_names}")

        # Load data from each suite
        for suite_dir in suite_dirs:
            suite_name = suite_dir.name
            print(f"Loading suite: {suite_name}")

            # Get all HDF5 files in this suite
            hdf5_files = list(suite_dir.glob("*.hdf5"))

            for hdf5_file in hdf5_files:
                # Skip HDF5 file if not in target split
                if target_hdf5_files and hdf5_file.name.replace("_demo", "") not in target_hdf5_files:
                    continue

                # Extract task description from filename using new method
                task_description = self._extract_task_description(hdf5_file.name)

                with h5py.File(hdf5_file, "r", swmr=True) as f:
                    data_group = f["data"]
                    demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]

                    for demo_key in demo_keys:
                        # Create episode identifier for tracking
                        episode_id = f"{suite_name}_{hdf5_file.stem}_{demo_key}"

                        demo = data_group[demo_key]

                        print(f"Loading episode {total_episode_count + 1} from {suite_name}...", end="\r")

                        if self.config.debug and total_episode_count > 5:
                            break

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
                            "suite_name": suite_name,
                            "episode_id": episode_id,
                        }

                        self.all_episodes.append(episode)

                        # Map each valid starting point for horizon-length sequences
                        for step_idx in range(max(0, episode_length - self.horizon + 1)):
                            self.transition_to_episode.append((total_episode_count, step_idx))
                            transition_count += 1

                        total_episode_count += 1

                        if self.config.debug and total_episode_count > 5:
                            break

                    if self.config.debug and total_episode_count > 5:
                        break

                if self.config.debug and total_episode_count > 5:
                    break

            print(f"\nLoaded episodes from {suite_name}, with task description: {task_description}")

        self.total_episodes = len(self.all_episodes)
        self.total_transitions = transition_count

        print(f"\nTotal loaded: {self.total_episodes} episodes with {self.total_transitions} total transitions")

    def __len__(self) -> int:
        """Return total number of transitions."""
        return self.total_transitions

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single transition with action sequence of length horizon."""
        if getattr(self.config, "inner_batch_size", 1) <= 1:
            episode_idx, step_idx = self.transition_to_episode[idx % self.total_transitions]
            return self._get_transition(episode_idx, step_idx)

        # Sample multiple transitions from the same episode
        episode_idx, _ = self.transition_to_episode[idx % self.total_transitions]
        episode = self.all_episodes[episode_idx]

        # Determine valid step indices
        max_step = max(1, episode["episode_length"] - self.horizon + 1)
        step_indices = [random.randint(0, max_step - 1) for _ in range(self.config.inner_batch_size)]

        items = [self._get_transition(episode_idx, s) for s in step_indices]

        # Stack items
        batch = {}
        for key in items[0].keys():
            if isinstance(items[0][key], torch.Tensor):
                batch[key] = torch.stack([item[key] for item in items])
            else:
                # Keep non-tensor items (like task_descriptions) as single item
                batch[key] = items[0][key]

        return batch

    def _get_transition(self, episode_idx: int, step_idx: int) -> Dict[str, Any]:
        """Fetch a single transition from an episode at step_idx."""
        episode = self.all_episodes[episode_idx]

        # Load image on-demand using persistent file handle
        file_handle = self._get_file_handle(episode["file_path"])
        demo = file_handle["data"][episode["demo_key"]]
        image = demo["obs"]["agentview_rgb"][step_idx]  # Load only the specific image
        gripper_image = demo["obs"]["eye_in_hand_rgb"][step_idx]

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

        # Process image - flip updown and left-right
        if isinstance(image, np.ndarray):
            # Flip image for proper orientation (match eval.py)
            image = np.flipud(image)
            image = np.fliplr(image)
        image = image.copy()

        # Apply Albumentations transforms
        image = self.image_transform(image=image)["image"].float()
        gripper_image = self.image_transform(image=gripper_image)["image"].float()

        # Convert robot state from [gripper_states(2), ee_pos(3), quat_xyzw(4)]
        # to [ee_pos(3), quat_as_axisangle(3), gripper_qpos(2)]
        gripper_states = state[:2]  # First 2 elements
        ee_pos = state[2:5]  # Next 3 elements
        quat_xyzw = state[5:9]  # Last 4 elements

        # Convert quaternion to axis-angle
        quat_axisangle = quat2axisangle(quat_xyzw)

        # Concatenate in target format: [ee_pos(3), quat_as_axisangle(3), gripper_qpos(2)]
        converted_state = np.concatenate([ee_pos, quat_axisangle, gripper_states])

        # Convert to torch tensors
        state = torch.tensor(converted_state, dtype=torch.float32)
        action_sequence = torch.tensor(action_sequence, dtype=torch.float32)  # Shape: (horizon, action_dim)

        return {
            "images": image,
            "gripper_images": gripper_image,
            "states": state,
            "actions": action_sequence,
            "task_descriptions": task_description,
            "task_ids": torch.tensor(0, dtype=torch.long),  # Simplified
        }

    def __del__(self):
        """Clean up HDF5 file handles when dataset is destroyed."""
        if hasattr(self, "_local") and hasattr(self._local, "file_handles"):
            for handle in self._local.file_handles.values():
                try:
                    handle.close()
                except:
                    pass  # Ignore errors during cleanup


class LiberoOriginalPerTaskDataset(LiberoOriginalDataset):
    """LIBERO Original per-task dataset that loads only a specific task with configurable limits."""

    def __init__(self, config: LiberoOriginalPerTaskDataConfig):
        if not config.task_name:
            raise ValueError("task_name must be specified for LiberoOriginalPerTaskDataset")

        self.task_name = config.task_name
        self.max_demos = config.max_demos

        # Initialize parent class
        super().__init__(config)

        print(f"Initialized LIBERO Original per-task dataset for task: '{self.task_name}'")
        if self.max_demos:
            print(f"Limited to {self.max_demos} demos")

    def _init_metadata_loading(self):
        """Initialize metadata loading - store file paths and episode info without loading images."""
        self.all_episodes = []
        self.transition_to_episode = []  # Maps transition index to (episode_idx, step_idx)

        transition_count = 0
        total_episode_count = 0
        loaded_demos_count = 0

        # Construct HDF5 filename directly from task name
        hdf5_filename = f"{self.task_name}_demo.hdf5"
        task_description = self._extract_task_description(hdf5_filename)

        # Get the target suite directory
        suite_dir = self.data_root_path / self.task_suite_name
        if not suite_dir.exists() or not suite_dir.is_dir():
            available_suites = [d.name for d in self.data_root_path.iterdir() if d.is_dir()]
            raise ValueError(
                f"Suite directory '{self.task_suite_name}' not found. Available suites: {available_suites}"
            )

        # Construct full path to HDF5 file
        hdf5_file_path = suite_dir / hdf5_filename

        if not hdf5_file_path.exists():
            available_files = [f.name for f in suite_dir.glob("*.hdf5")]
            raise ValueError(
                f"Task file '{hdf5_filename}' not found in {suite_dir}. Available files: {available_files}"
            )

        print(f"Loading task '{self.task_name}' from {hdf5_file_path}")

        with h5py.File(hdf5_file_path, "r", swmr=True) as f:
            data_group = f["data"]
            demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]

            for demo_key in demo_keys:
                # Check if we've reached the demo limit
                if self.max_demos and loaded_demos_count >= self.max_demos:
                    print(f"Reached maximum demos limit ({self.max_demos})")
                    break

                # Create episode identifier for tracking
                episode_id = f"{self.task_suite_name}_{hdf5_file_path.stem}_{demo_key}"

                demo = data_group[demo_key]

                print(f"Loading demo {loaded_demos_count + 1}...", end="\r")

                if self.config.debug and total_episode_count > 5:
                    break

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
                    "task_description": task_description,  # Use task_name directly
                    "suite_name": self.task_suite_name,
                    "episode_id": episode_id,
                }

                self.all_episodes.append(episode)

                # Map each valid starting point for horizon-length sequences
                for step_idx in range(max(0, episode_length - self.horizon + 1)):
                    self.transition_to_episode.append((total_episode_count, step_idx))
                    transition_count += 1

                total_episode_count += 1
                loaded_demos_count += 1

                if self.config.debug and total_episode_count > 5:
                    break

        self.total_episodes = len(self.all_episodes)
        self.total_transitions = transition_count

        if self.total_episodes == 0:
            raise ValueError(f"No episodes found for task '{self.task_name}'.")

        print(
            f"\nTotal loaded for task '{self.task_name}': {self.total_episodes} episodes with {self.total_transitions} total transitions"
        )

    def __len__(self) -> int:
        """Return total number of transitions."""
        return self.total_transitions * 10000

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return super().__getitem__(idx)

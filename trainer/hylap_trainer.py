"""
HyLaP Trainer using shared vision and language encoders with HyPoGen2 policy.

This trainer reuses the encoders created in GenericTrainer and builds a
minimal end-to-end supervised objective to train the HyPoGen2 policy
wrapper on (obs, language) -> actions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from trainer.generic_trainer import GenericTrainer
from minlora import add_lora, get_lora_params, get_lora_state_dict, remove_lora, LoRAParametrization


class HyLaPTrainer(GenericTrainer):
    def __init__(
        self,
        policy_network: Dict[str, Any],
        # Loss
        action_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 0.0,
        # Optional repo root for external dependency
        hypogen2_repo_root: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        # Test-time training configuration
        self.test_time_training_config = kwargs.get("test_time_training_config", {})
        self.test_time_training_enabled = self.test_time_training_config.get("enabled", False)
        self.use_early_sup = kwargs.get("use_early_sup", False)

        # Determine base feature dimension from encoders
        image_dim = getattr(self.image_encoder, "feature_dim", None)
        assert image_dim is not None, "image_encoder must expose feature_dim"

        language_dim = getattr(self.language_encoder, "output_dim", None)
        assert language_dim is not None, "language_encoder must expose output_dim"

        self.num_views = self._get_num_views()
        base_input_dim = image_dim * self.num_views
        if self.use_state_conditioning and self.state_encoder is not None:
            if hasattr(self.state_encoder, "out_features"):
                base_input_dim += int(self.state_encoder.out_features)
            else:
                base_input_dim += 256

        # Build HyLaP policy (HypoGen2 wrapper)
        self.policy = self._create_policy(policy_network, language_dim, base_input_dim)
        self.action_loss_weight = action_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

    # Satisfy GenericTrainer abstract API (not used by this trainer directly)
    def _create_policy_network(self, config: Dict[str, Any]) -> nn.Module:
        meta_dim = getattr(self.language_encoder, "output_dim")
        image_dim = getattr(self.image_encoder, "feature_dim")
        base_input_dim = image_dim * self._get_num_views()
        if self.use_state_conditioning and self.state_encoder is not None:
            base_input_dim += int(getattr(self.state_encoder, "out_features", 256))
        return self._create_policy(config, meta_dim, base_input_dim)

    def _create_policy(self, config: Dict[str, Any], meta_dim: int, base_input_dim: int) -> nn.Module:
        target = config.get("_target_", "model.hylap.hypogen2.HyLaPHypoGenPolicy")
        params = {k: v for k, v in config.items() if k != "_target_"}

        # Provide required dims from encoders/dataset
        params.setdefault("meta_dim", meta_dim)
        params.setdefault("base_input_dim", base_input_dim)
        params.setdefault("action_dim", self.action_dim)

        # Import and instantiate
        if target in ["HyLaPHypoGenPolicy", "model.hylap.hypogen2.HyLaPHypoGenPolicy"]:
            from model.hylap.hypogen2 import HyLaPHypoGenPolicy

            return HyLaPHypoGenPolicy(**params)
        module_path, class_name = target.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)(**params)

    def encode_observations(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Images -> vision features
        # Support multi-view input (Libero: images, gripper_images; Real: view_1, view_2, view_3, ...)
        image_feats = []
        if "view_1" in batch:
            i = 1
            while f"view_{i}" in batch:
                image_feats.append(self.image_encoder(batch[f"view_{i}"]))
                i += 1
        else:
            if "images" in batch:
                image_feats.append(self.image_encoder(batch["images"]))
            if "gripper_images" in batch:
                image_feats.append(self.image_encoder(batch["gripper_images"]))

        # Language -> language features via encoder (can output sequence if configured)
        task_desc = batch["task_descriptions"]
        # Tokenize using encoder's tokenize method to respect padding/max_length settings
        tokenized = self.language_encoder.tokenize(task_desc)
        tokens = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        lang_feat = self.language_encoder(tokens)

        # State (optional)
        if self.use_state_conditioning and self.state_encoder is not None and "states" in batch:
            state_feat = self.state_encoder(batch["states"])  # (B, S')
        else:
            state_feat = None

        # Data augmentation: 50% chance to zero out either first view or robot states during training
        # if self.training and torch.rand(1).item() < 0.5:
        #     if torch.rand(1).item() < 0.5 and len(image_feats) > 0:
        #         # Zero out first view (main camera) embedding
        #         image_feats[0] = torch.zeros_like(image_feats[0])
        #     else:
        #         # Zero out robot states if available
        #         if state_feat is not None:
        #             state_feat = torch.zeros_like(state_feat)

        return {"image_feats": image_feats, 
                "lang": lang_feat, 
                "state": state_feat, 
                "attention_mask": attention_mask}

    def forward(self, batch: Dict[str, torch.Tensor], early_sup: bool = False) -> torch.Tensor:
        feats = self.encode_observations(batch)
        base_list = feats["image_feats"]
        if feats["state"] is not None:
            base_list.append(feats["state"])
        base_feat = torch.cat(base_list, dim=1)

        if torch.is_grad_enabled() == False:
            if hasattr(self, "lang_cache") and self.lang_cache is not None:
                if len(self.lang_cache) == len(batch["task_descriptions"]) and \
                    all([self.lang_cache[i] == batch["task_descriptions"][i] for i in range(len(self.lang_cache))]):
                    return self.policy.hyper_tr.forward_with_weights(None, base_feat).reshape(-1, self.policy.horizon, self.policy.action_dim)
            self.lang_cache = batch["task_descriptions"]
        else:
            self.lang_cache = None

        return self.policy(feats["lang"], base_feat, feats["attention_mask"], early_sup)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, is_fast_adapt: bool = False):
        pred_actions = self.forward(batch, early_sup=self.use_early_sup)
        target_actions = batch["actions"]  # (B, A) or (B, H, A)
        # if self.use_early_sup:
        #     target_actions = einops.repeat(target_actions, '... -> l ...', l=pred_actions.shape[0])
        loss = F.mse_loss(pred_actions, target_actions)
        total = loss * self.action_loss_weight

        # Add contrastive loss if weight > 0 and available
        if self.contrastive_loss_weight > 0 and hasattr(self.language_encoder, "get_contrastive_loss"):
            contrastive_loss = self.language_encoder.get_contrastive_loss()
            total += contrastive_loss * self.contrastive_loss_weight
            if not is_fast_adapt:
                self.log("train/contrastive_loss", contrastive_loss, on_step=True, prog_bar=True)

        if not is_fast_adapt:
            self.log("train/loss", total, on_step=True, prog_bar=True)
            self.log("train/action_loss", loss, on_step=True, prog_bar=True)
            # if self.global_rank == 0:
            #     self.logger.experiment.add_histogram("train/lrs", self.policy.hyper_tr.lrs)
        return {"loss": total}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self.eval()
        
        with torch.no_grad():
            pred_actions = self.forward(batch)
            target_actions = batch["actions"]  # (B, A) or (B, H, A)
            loss = F.mse_loss(pred_actions, target_actions)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/action_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Predict actions.

        If policy predicts horizon>1, return (B, H, A). If it predicts a single
        step (B, A), expand to (B, 1, A) for downstream uniformity.
        """
        self.eval()
        
        with torch.no_grad():
            actions = self.forward(batch)
            if actions.ndim == 2:
                actions = actions.unsqueeze(1)  # (B, 1, A)
            return actions

    def prepare_fast_adapt(self, config: Dict[str, Any]) -> None:
        """
        Prepare the model for fast adaptation using either LoRA on hypernet or direct target network fine-tuning.

        Args:
            config: Configuration dictionary containing adaptation parameters.
                   Must include 'adaptation_method': 'lora' or 'policynet'
        """
        # Store original state for potential restoration
        self._original_state_dict = deepcopy(self.state_dict())
        
        adaptation_method = config.get("adaptation_method", "lora")
        self._adaptation_method = adaptation_method
        
        if adaptation_method == "lora":
            self._prepare_lora_hypernet_adapt(config)
        elif adaptation_method == "policynet":
            self._prepare_target_finetune_adapt(config)
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}. "
                           "Supported methods: 'lora', 'policynet'")

    def _prepare_lora_hypernet_adapt(self, config: Dict[str, Any]) -> None:
        """
        Prepare the hypernet for fast adaptation using LoRA.

        This method applies LoRA to the hypernet parameters (encoders, decoders, opt_layer, etc.)
        instead of fine-tuning the generated target network weights. The target network itself
        is excluded from LoRA adaptation.

        Args:
            config: Configuration dictionary containing LoRA parameters
        """
        lora_config = {
            nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=config["lora_rank"]),
            },
        }

        # Backup the original policy for restoration
        self._policy_bak = deepcopy(self.policy)

        # Move policy to CPU for LoRA application
        self.policy = self.policy.to("cpu")

        # Apply LoRA to hypernetwork components, excluding the target network
        # Apply to task adapter
        if hasattr(self.policy.hyper_tr, "ftask_adapter") and isinstance(self.policy.hyper_tr.ftask_adapter, nn.Linear):
            add_lora(self.policy.hyper_tr.ftask_adapter, lora_config=lora_config)

        # Apply to encoders
        if hasattr(self.policy.hyper_tr, "encoders"):
            add_lora(self.policy.hyper_tr.encoders, lora_config=lora_config)

        # Apply to decoders
        if hasattr(self.policy.hyper_tr, "decoders"):
            add_lora(self.policy.hyper_tr.decoders, lora_config=lora_config)

        # Apply to optimization layer
        if hasattr(self.policy.hyper_tr, "opt_layer"):
            add_lora(self.policy.hyper_tr.opt_layer, lora_config=lora_config)

        # Apply to layer norms
        if hasattr(self.policy.hyper_tr, "layer_norms"):
            add_lora(self.policy.hyper_tr.layer_norms, lora_config=lora_config)

        # Apply to initial weight hypernet
        if hasattr(self.policy.hyper_tr, "init_weight_hypernet"):
            add_lora(self.policy.hyper_tr.init_weight_hypernet, lora_config=lora_config)

        # Move back to device
        self.policy = self.policy.to(self.device)
        self._lora_config = lora_config

        print(
            f"LoRA adaptation prepared with rank={config['lora_rank']} on hypernet parameters (target network excluded)"
        )

    def _prepare_target_finetune_adapt(self, config: Dict[str, Any]) -> None:
        """
        Prepare the policy network for fast adaptation by fine-tuning the whole target network.

        Unlike LoRA-based adaptation, this method prepares for full parameter fine-tuning
        of the target network while keeping encoders frozen.

        Args:
            config: Configuration dictionary containing adaptation parameters
        """
        # Store original forward method
        self._original_forward = self.forward

        task_description = config["task_description"]

        # Properly tokenize the task description
        if isinstance(task_description, str):
            task_description = [task_description]  # Convert to list for tokenizer

        with torch.no_grad():
            tokenized = self.language_encoder.tokenize(
                task_description
            )
            tokens = tokenized["input_ids"].to(self.device)
            lang_feat = self.language_encoder(tokens)

            weight_dict = self.policy.generate_weights(lang_feat)[-1]

        # Convert weight_dict to parameter_dict for optimization
        parameter_dict = nn.ParameterDict()
        for key, weight_tensor in weight_dict.items():
            # Convert to parameter to enable gradient computation
            parameter_dict[key.replace(".", "_")] = nn.Parameter(weight_tensor[0].clone().detach().requires_grad_(True))

        # Store parameter_dict for optimization
        self._parameter_dict = parameter_dict

        # Override forward method to use parameter dict directly
        def custom_forward(batch: Dict[str, torch.Tensor], early_sup: bool = False) -> torch.Tensor:
            self.eval()
            # Encode observations (image and state only, no language)
            # Support multi-view input (Libero: images, gripper_images; Real: view_1, view_2, view_3, ...)
            base_list = []
            if "view_1" in batch:
                i = 1
                while f"view_{i}" in batch:
                    base_list.append(self.image_encoder(batch[f"view_{i}"]))
                    i += 1
            else:
                if "images" in batch:
                    base_list.append(self.image_encoder(batch["images"]))
                if "gripper_images" in batch:
                    base_list.append(self.image_encoder(batch["gripper_images"]))
            
            if self.use_state_conditioning and "states" in batch:
                state_feat = self.state_encoder(batch["states"])
                base_list.append(state_feat)
            
            base_feat = torch.cat(base_list, dim=1)

            # Use the target network directly with parameter dict
            target_net = self.policy.hyper_tr.target_net

            parameter_dict = {k.replace("_", "."): v for k, v in self._parameter_dict.items()}

            # Apply the policy network using functional call with parameter dict
            actions = torch.func.functional_call(target_net, parameter_dict, base_feat)

            # Reshape actions from (B, action_dim * horizon) to (B, horizon, action_dim)
            if actions.dim() == 2:
                batch_size = actions.shape[0]
                actions = actions.view(batch_size, self.policy.horizon, self.policy.action_dim)

            return actions

        # Replace the forward method
        self.forward = custom_forward

        print(f"Target network fine-tuning adaptation prepared")

    def restore_adaptation(self, restoration_info: Any) -> None:
        """
        Restore the model to its original state before adaptation.

        Args:
            restoration_info: Not used for adaptation, kept for interface compatibility
        """
        if not hasattr(self, "_original_state_dict"):
            print("Warning: No original state found. Cannot restore adaptation.")
            return

        adaptation_method = getattr(self, "_adaptation_method", "lora")
        
        if adaptation_method == "lora":
            self._restore_lora_hypernet_adapt()
        elif adaptation_method == "policynet":
            self._restore_target_finetune_adapt()

        # Clean up common state
        if hasattr(self, "_original_state_dict"):
            delattr(self, "_original_state_dict")
        if hasattr(self, "_adaptation_method"):
            delattr(self, "_adaptation_method")

    def _restore_lora_hypernet_adapt(self) -> None:
        """
        Restore the hypernet to its original state before LoRA adaptation.
        """
        # Restore the original policy from backup
        self.policy = self._policy_bak.to(self.device)
        if hasattr(self, "_policy_bak"):
            delattr(self, "_policy_bak")

        # Load original state dict
        self.load_state_dict(self._original_state_dict)

        # Clean up LoRA config
        if hasattr(self, "_lora_config"):
            delattr(self, "_lora_config")

        print("HyLaP hypernet restored to original state")

    def _restore_target_finetune_adapt(self) -> None:
        """
        Restore the model to its original state before target network adaptation.
        """
        if hasattr(self, "_parameter_dict"):
            delattr(self, "_parameter_dict")

        # Restore original state
        self.load_state_dict(self._original_state_dict)

        # Restore original forward method
        if hasattr(self, "_original_forward"):
            self.forward = self._original_forward
            delattr(self, "_original_forward")

        print("HyLaP model restored to original state")

    def get_adaptation_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Get an optimizer for fast adaptation based on the chosen method.

        Args:
            config: Configuration dictionary containing adaptation parameters

        Returns:
            Optimizer configured for the chosen adaptation method
        """
        adaptation_method = getattr(self, "_adaptation_method", "lora")
        
        if adaptation_method == "lora":
            return self._get_lora_hypernet_optimizer(config)
        elif adaptation_method == "policynet":
            return self._get_target_finetune_optimizer(config)
        else:
            raise RuntimeError(f"Unknown adaptation method: {adaptation_method}")

    def _get_lora_hypernet_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Get an optimizer that only updates LoRA parameters for fast adaptation.

        Args:
            config: Configuration dictionary containing lr and weight_decay

        Returns:
            Optimizer configured to update only LoRA parameters
        """
        if not hasattr(self, "_lora_config"):
            raise RuntimeError("LoRA adaptation not prepared. Call prepare_fast_adapt() first.")

        # Get LoRA parameters from hypernetwork components (excluding target network)
        lora_params = []

        # Collect from task adapter
        if hasattr(self.policy.hyper_tr, "ftask_adapter") and isinstance(self.policy.hyper_tr.ftask_adapter, nn.Linear):
            lora_params.extend(get_lora_params(self.policy.hyper_tr.ftask_adapter))

        # Collect from encoders
        if hasattr(self.policy.hyper_tr, "encoders"):
            lora_params.extend(get_lora_params(self.policy.hyper_tr.encoders))

        # Collect from decoders
        if hasattr(self.policy.hyper_tr, "decoders"):
            lora_params.extend(get_lora_params(self.policy.hyper_tr.decoders))

        # Collect from optimization layer
        if hasattr(self.policy.hyper_tr, "opt_layer"):
            lora_params.extend(get_lora_params(self.policy.hyper_tr.opt_layer))

        # Collect from layer norms
        if hasattr(self.policy.hyper_tr, "layer_norms"):
            lora_params.extend(get_lora_params(self.policy.hyper_tr.layer_norms))

        # Collect from initial weight hypernet
        if hasattr(self.policy.hyper_tr, "init_weight_hypernet"):
            lora_params.extend(get_lora_params(self.policy.hyper_tr.init_weight_hypernet))

        if not lora_params:
            raise RuntimeError("No LoRA parameters found. Ensure LoRA is properly added to the hypernet components.")

        lr = config["adaptation_lr"]
        weight_decay = config["adaptation_weight_decay"]

        # Create optimizer for LoRA parameters only
        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

        total_params = sum(p.numel() for p in lora_params)
        print(f"====  Total fine-tuned parameters (LoRA): {total_params/1e6:.4f}M")

        return optimizer

    def _get_target_finetune_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Get an optimizer that updates all target network parameters for fast adaptation.

        Args:
            config: Configuration dictionary containing adaptation parameters

        Returns:
            Optimizer configured to update all target network parameters
        """
        if not hasattr(self, "_parameter_dict"):
            raise RuntimeError("Target network adaptation not prepared. Call prepare_fast_adapt() first.")

        # Get all target network parameters that require gradients
        lr = config.get("adaptation_lr", 1e-3)
        weight_decay = config.get("adaptation_weight_decay", 1e-4)

        # Create optimizer for all target network parameters
        optimizer = torch.optim.AdamW(self._parameter_dict.values(), lr=lr, weight_decay=weight_decay)

        total_params = sum(p.numel() for p in self._parameter_dict.values())
        print(f"====  Total fine-tuned parameters (target network): {total_params/1e6:.4f}M")

        return optimizer

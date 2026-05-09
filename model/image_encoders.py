import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional

def convert_batchnorm_to_layernorm(module):
    """
    Recursively convert BatchNorm2d layers to GroupNorm (acting as LayerNorm for conv).
    
    Args:
        module: PyTorch module
        
    Returns:
        Module with BatchNorm replaced
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        # Create GroupNorm with 32 groups (standard for ResNet) or 1 group (LayerNorm)
        # Using 32 groups is standard "GroupNorm" for ResNet
        num_features = module.num_features
        # Ensure num_groups divides num_features. ResNet features are usually multiple of 32 or 64.
        # If not, fall back to fewer groups or 1 group.
        num_groups = 32
        if num_features % num_groups != 0:
            num_groups = 1  # LayerNorm behavior
            
        module_output = torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm_to_layernorm(child))

    return module_output


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for visual observations.

    This encoder uses a pre-trained ResNet backbone for feature extraction
    from RGB images.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 512,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        input_channels: int = 3,
        **kwargs,
    ):
        """
        Initialize ResNet encoder.

        Args:
            backbone: ResNet architecture ('resnet18', 'resnet34', 'resnet50')
            feature_dim: Output feature dimension
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze backbone parameters
            input_channels: Number of input channels (default: 3)
        """
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.input_channels = input_channels
        convert_bn2ln = kwargs.get("convert_bn2ln", False)

        # Create backbone
        if backbone == "resnet18":
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.backbone = models.resnet18()
            backbone_dim = 512
        elif backbone == "resnet34":
            if pretrained:
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                self.backbone = models.resnet34()
            backbone_dim = 512
        elif backbone == "resnet50":
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50()
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify first layer if input channels != 3
        if input_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # Initialize the new conv layer
            # If pretrained, we can copy weights for the first 3 channels and initialize others
            if pretrained:
                with torch.no_grad():
                    # Copy weights for first 3 channels
                    self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                    # Initialize remaining channels with average of first 3
                    if input_channels > 3:
                        avg_weight = torch.mean(original_conv.weight, dim=1, keepdim=True)
                        for i in range(3, input_channels):
                            self.backbone.conv1.weight[:, i:i+1, :, :] = avg_weight

        # Replace final layer
        if feature_dim != backbone_dim:
            self.backbone.fc = nn.Linear(backbone_dim, feature_dim)
        else:
            self.backbone.fc = nn.Identity()

        # Freeze backbone if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Keep final layer trainable
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            
            # If we modified input channels, we should probably keep conv1 trainable
            # unless specifically desired otherwise. Usually for depth input, we want to finetune.
            if input_channels != 3:
                for param in self.backbone.conv1.parameters():
                    param.requires_grad = True

        if convert_bn2ln:
            # Convert all BatchNorm2d layers to GroupNorm (which acts like LayerNorm for conv layers)
            self.backbone = convert_batchnorm_to_layernorm(self.backbone)
            print("BatchNorm2d converted to LayerNorm")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            features: Encoded features (B, feature_dim)
        """
        return self.backbone(x)

    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            "backbone": self.backbone_name,
            "feature_dim": self.feature_dim,
            "input_channels": self.input_channels,
        }


class ConvEncoder(nn.Module):
    """
    Simple convolutional encoder for visual observations.

    This encoder uses a series of convolutional layers followed by
    global average pooling for feature extraction.
    """

    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 512,
        conv_dims: tuple = (32, 64, 128, 256),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        activation: str = "relu",
        use_batchnorm: bool = True,
        **kwargs,
    ):
        """
        Initialize convolutional encoder.

        Args:
            input_channels: Number of input channels
            feature_dim: Output feature dimension
            conv_dims: Channel dimensions for conv layers
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            activation: Activation function
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Choose activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "swish":
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build convolutional layers
        layers = []
        in_channels = input_channels

        for out_channels in conv_dims:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act_fn())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final projection
        self.projection = nn.Linear(conv_dims[-1], feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            features: Encoded features (B, feature_dim)
        """
        # Extract features through conv layers
        features = self.conv_layers(x)  # (B, C', H', W')

        # Global average pooling
        features = self.global_pool(features)  # (B, C', 1, 1)
        features = features.view(features.shape[0], -1)  # (B, C')

        features = self.projection(features)  # (B, feature_dim)

        return features


class ResNetTokenEncoder(nn.Module):
    """
    ResNet-based encoder that outputs multiple tokens instead of a single embedding.
    
    This encoder removes the global average pooling and outputs spatial feature maps
    as multiple tokens, preserving spatial information.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        token_dim: int = 512,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        spatial_size: int = 7,  # 7x7 = 49 tokens for 224x224 input
        input_channels: int = 3,
        **kwargs,
    ):
        """
        Initialize ResNet token encoder.

        Args:
            backbone: ResNet architecture ('resnet18', 'resnet34', 'resnet50')
            token_dim: Dimension of each token
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze backbone parameters
            spatial_size: Spatial size of feature map (e.g., 7 for 7x7 tokens)
            input_channels: Number of input channels (default: 3)
        """
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.backbone_name = backbone
        self.token_dim = token_dim
        self.feature_dim = token_dim
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size
        self.input_channels = input_channels

        # Create backbone without the final layers
        if backbone == "resnet18":
            if pretrained:
                full_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                full_model = models.resnet18()
            backbone_dim = 512
        elif backbone == "resnet34":
            if pretrained:
                full_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                full_model = models.resnet34()
            backbone_dim = 512
        elif backbone == "resnet50":
            if pretrained:
                full_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                full_model = models.resnet50()
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify first layer if input channels != 3
        if input_channels != 3:
            original_conv = full_model.conv1
            full_model.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # Initialize the new conv layer
            if pretrained:
                with torch.no_grad():
                    # Copy weights for first 3 channels
                    full_model.conv1.weight[:, :3, :, :] = original_conv.weight
                    # Initialize remaining channels
                    if input_channels > 3:
                        avg_weight = torch.mean(original_conv.weight, dim=1, keepdim=True)
                        for i in range(3, input_channels):
                             full_model.conv1.weight[:, i:i+1, :, :] = avg_weight

        # Remove the global average pooling and final linear layer
        # Keep everything up to layer4
        self.backbone = nn.Sequential(*list(full_model.children())[:-2])
        
        # Add adaptive pooling to get desired spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
        
        # Project to token dimension if needed
        if token_dim != backbone_dim:
            self.token_projection = nn.Linear(backbone_dim, token_dim)
        else:
            self.token_projection = nn.Identity()

        # Freeze backbone if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Keep projection layer trainable
            if hasattr(self.token_projection, 'parameters'):
                for param in self.token_projection.parameters():
                    param.requires_grad = True
                    
            # Keep first conv trainable if modified
            if input_channels != 3:
                # We need to access conv1 within sequential.
                # full_model.children() list was used. conv1 is the first child.
                # However, self.backbone is a Sequential. We can access by index.
                # Index 0 is conv1.
                for param in self.backbone[0].parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            tokens: Encoded tokens (B, num_tokens, token_dim)
        """
        # Extract features through backbone
        features = self.backbone(x)  # (B, backbone_dim, H', W')
        
        # Adaptive pooling to get desired spatial size
        features = self.adaptive_pool(features)  # (B, backbone_dim, spatial_size, spatial_size)
        
        # Reshape to tokens
        B, C, H, W = features.shape
        tokens = features.view(B, C, H * W).transpose(1, 2)  # (B, num_tokens, backbone_dim)
        
        # Project to token dimension
        tokens = self.token_projection(tokens)  # (B, num_tokens, token_dim)
        
        return tokens

    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            "backbone": self.backbone_name,
            "token_dim": self.token_dim,
            "spatial_size": self.spatial_size,
            "num_tokens": self.num_tokens,
            "input_channels": self.input_channels,
        }


def create_image_encoder(encoder_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create image encoders.

    Args:
        encoder_config: Configuration dictionary with '_target_' key

    Returns:
        Configured encoder instance
    """
    encoder_type = encoder_config.get("_target_", "").split(".")[-1]
    config = {k: v for k, v in encoder_config.items() if k != "_target_"}

    if encoder_type == "ResNetEncoder":
        return ResNetEncoder(**config)
    elif encoder_type == "ResNetTokenEncoder":
        return ResNetTokenEncoder(**config)
    elif encoder_type == "ConvEncoder":
        return ConvEncoder(**config)
    else:
        # Try to import and instantiate dynamically
        import importlib

        module_path, class_name = encoder_config["_target_"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        encoder_class = getattr(module, class_name)
        return encoder_class(**config)


if __name__ == "__main__":
    # Test encoders
    print("Testing image encoders...")

    # Test ResNet encoder
    resnet_encoder = ResNetEncoder(backbone="resnet18", feature_dim=512, pretrained=True)
    test_images = torch.randn(4, 3, 224, 224)
    features = resnet_encoder(test_images)
    print(f"ResNet features shape: {features.shape}")

    # Test conv encoder
    conv_encoder = ConvEncoder(feature_dim=256)
    features = conv_encoder(test_images)
    print(f"Conv features shape: {features.shape}")

    print("Image encoder tests completed!")

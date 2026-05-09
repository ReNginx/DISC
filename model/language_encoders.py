"""
Language encoders for text-based conditioning in policy learning.

This module provides various encoder architectures for processing
natural language instructions and task descriptions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List, Tuple


class GRULanguageEncoder(nn.Module):
    """
    GRU-based language encoder for task descriptions.
    
    This encoder uses embeddings followed by a GRU to process
    natural language instructions. It has built-in BertTokenizer
    for consistency with other encoders.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        output_sequence: bool = False,
        tokenizer_model: str = "google-bert/bert-base-uncased",
        **kwargs,
    ):
        """
        Initialize GRU language encoder.

        Args:
            embed_dim: Embedding dimension
            hidden_dim: GRU hidden dimension
            output_dim: Output feature dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            output_sequence: If True, return sequence features; if False, return pooled features
            tokenizer_model: BERT tokenizer model name
        """
        super().__init__()

        # Import and create tokenizer
        try:
            from transformers import BertTokenizerFast
        except ImportError:
            raise ImportError("transformers library is required for GRULanguageEncoder")
        
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_model, local_files_only=True)

        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_sequence = output_sequence

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)

        # GRU layer
        self.gru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        print(f"GRU parameters: {sum(p.numel() for p in self.gru.parameters())}")
        print(f"Embedding parameters: {sum(p.numel() for p in self.embedding.parameters())}")

        # Output projection
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(gru_output_dim, output_dim)
        print(f"Output projection parameters: {sum(p.numel() for p in self.output_proj.parameters())}")

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Input token indices (B, seq_len)
            attention_mask: Attention mask (B, seq_len) where 1=valid, 0=padding

        Returns:
            If output_sequence=False: features (B, output_dim)
            If output_sequence=True: (features (B, seq_len, output_dim), attention_mask)
        """
        # Handle single token input
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len = input_ids.shape

        # Calculate lengths from attention mask if provided
        lengths = None

        # Embedding
        embedded = self.embedding(input_ids)  # (B, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        # Pack sequences if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # GRU forward pass
        output, hidden = self.gru(embedded)

        # Unpack if necessary
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Project output
        projected_output = self.output_proj(output)  # (B, seq_len, output_dim)

        # If sequence output is requested, return sequence and mask
        if self.output_sequence:
            return projected_output

        # Use final hidden state for pooled output
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, 2*hidden_dim)
            # Project to output dimension
            features = self.output_proj(hidden)  # (B, output_dim)
        else:
            hidden = hidden[-1]  # (B, hidden_dim)
            # Project to output dimension
            features = self.output_proj(hidden)  # (B, output_dim)

        return features

    def tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (if None, use self.max_length)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length

        padding_type = "max_length" if getattr(self, "padding", True) else False

        return self.tokenizer(
            texts,
            padding=padding_type,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            "tokenizer": self.tokenizer,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "output_sequence": self.output_sequence,
        }


class LSTMLanguageEncoder(nn.Module):
    """
    LSTM-based language encoder for task descriptions.
    
    Similar to GRU encoder but uses LSTM cells.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        **kwargs
    ):
        """Initialize LSTM language encoder."""
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output projection
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(lstm_output_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # Pack sequences if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward pass
        output, (hidden, cell) = self.lstm(embedded)

        # Use final hidden state
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        # Project to output dimension
        features = self.output_proj(hidden)

        return features


def create_language_encoder(encoder_config: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
    """
    Factory function to create language encoders.
    
    Args:
        encoder_config: Configuration dictionary with '_target_' key
        
    Returns:
        Configured encoder instance
    """
    if encoder_config is None:
        return None
        
    encoder_type = encoder_config.get("_target_", "").split(".")[-1]
    config = {k: v for k, v in encoder_config.items() if k != "_target_"}

    if encoder_type == "GRULanguageEncoder":
        return GRULanguageEncoder(**config)
    elif encoder_type == "LSTMLanguageEncoder":
        return LSTMLanguageEncoder(**config)
    elif encoder_type == "T5LanguageEncoder":
        return T5LanguageEncoder(**config)
    elif encoder_type == "CLIPLanguageEncoder":
        return CLIPLanguageEncoder(**config)
    elif encoder_type == "BERTLanguageEncoder":
        return BERTLanguageEncoder(**config)
    else:
        # Try to import and instantiate dynamically
        import importlib
        module_path, class_name = encoder_config["_target_"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        encoder_class = getattr(module, class_name)
        return encoder_class(**config)


class T5LanguageEncoder(nn.Module):
    """
    T5-based language encoder.
    
    Uses T5-small model for encoding language instructions.
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        output_dim: int = 256,
        output_sequence: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0,
        max_length: int = 32,  # 添加 max_length 参数
        padding: bool = True, # 添加 padding 参数
        **kwargs,  # 添加 **kwargs 来处理其他未知参数
    ):
        """
        Initialize T5 language encoder.
        
        Args:
            model_name: T5 model name (default: t5-small)
            output_dim: Output feature dimension
            output_sequence: If True, return sequence features; if False, return pooled features
            freeze_backbone: Whether to freeze T5 backbone
            dropout: Dropout probability
            max_length: Maximum sequence length for tokenization (default: 77)
            padding: Whether to use padding (default: True)
        """
        super().__init__()

        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError:
            raise ImportError("transformers library is required for T5LanguageEncoder")

        self.output_sequence = output_sequence
        self.output_dim = output_dim
        self.max_length = max_length  # 存储 max_length 参数
        self.padding = padding # 存储 padding 参数

        # Load T5 encoder and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
        self.encoder = T5EncoderModel.from_pretrained(model_name, local_files_only=True)
        self.freeze_backbone = freeze_backbone

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get T5 hidden dimension
        self.t5_hidden_dim = self.encoder.config.d_model

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection (no activation as requested)
        self.output_proj = nn.Linear(self.t5_hidden_dim, output_dim)

        self._last_loss = torch.tensor(0.0)

    def get_contrastive_loss(self) -> torch.Tensor:
        """Retrieve the precomputed contrastive loss."""
        return self._last_loss

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            
        Returns:
            If output_sequence=False: features [B, D]
            If output_sequence=True: (language_seq [B, L, D], mask [B, L])
        """
        # 直接forward，依赖初始化时设置的requires_grad=False
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        projected = self.output_proj(hidden_states)
        
        # Pool sequence features (mean pooling with attention mask) for contrastive loss
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(projected)
            sum_embeddings = torch.sum(projected * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled = projected.mean(dim=1)

        # Compute contrastive loss if in training mode and batch size > 1
        if self.training and input_ids.shape[0] > 1:
            # Normalize features for cosine similarity
            feat = torch.nn.functional.normalize(pooled, p=2, dim=1)
            sim = torch.matmul(feat, feat.T) / 0.07 # temperature=0.07
            
            # Create labels based on input_ids (same language = same class)
            # (B, 1, L) == (1, B, L) -> (B, B, L) -> (B, B)
            labels = (input_ids.unsqueeze(1) == input_ids.unsqueeze(0)).all(dim=-1).float()
            
            # InfoNCE loss with multiple positives
            mask = torch.eye(labels.shape[0], device=labels.device)
            pos_mask = labels - mask # remove self-similarity
            exp_sim = torch.exp(sim) * (1 - mask)
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
            
            # Mean log-probability over positives
            pos_counts = pos_mask.sum(dim=1)
            valid_indices = pos_counts > 0
            if valid_indices.any():
                self._last_loss = -(log_prob * pos_mask).sum(dim=1)[valid_indices] / pos_counts[valid_indices]
                self._last_loss = self._last_loss.mean()
            else:
                self._last_loss = torch.tensor(0.0, device=input_ids.device)
        else:
            self._last_loss = torch.tensor(0.0, device=input_ids.device)

        if self.output_sequence:
            # Return sequence features
            return projected
        else:
            return pooled

    def tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (if None, use self.max_length)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length

        padding_type = "max_length" if getattr(self, "padding", True) else False
        truncation = True if padding_type == 'max_length' else False

        return self.tokenizer(
            texts,
            padding=padding_type,
            max_length=max_length,
            truncation=truncation,
            return_tensors="pt"
        )


class CLIPLanguageEncoder(nn.Module):
    """
    CLIP-based language encoder.
    
    Uses CLIP text encoder for encoding language instructions.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        output_dim: int = 256,
        output_sequence: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0,
        max_length: int = 77,
        **kwargs,
    ):
        """
        Initialize CLIP language encoder.
        
        Args:
            model_name: CLIP model name (default: openai/clip-vit-base-patch16)
            output_dim: Output feature dimension
            output_sequence: If True, return sequence features; if False, return pooled features
            freeze_backbone: Whether to freeze CLIP backbone
            dropout: Dropout probability
            max_length: Maximum sequence length for tokenization (default: 77)
        """
        super().__init__()

        try:
            from transformers import AutoTokenizer, CLIPTextModel
        except ImportError:
            raise ImportError("transformers library is required for CLIPLanguageEncoder")

        self.output_sequence = output_sequence
        self.output_dim = output_dim
        self.max_length = max_length

        # Load CLIP text encoder and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get CLIP hidden dimension
        self.clip_hidden_dim = self.encoder.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(self.clip_hidden_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            
        Returns:
            If output_sequence=False: features [B, D]
            If output_sequence=True: (language_seq [B, L, D], mask [B, L])
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        projected = self.output_proj(hidden_states)
        
        if self.output_sequence:
            # Return sequence features
            return projected
        else:
            # Pool sequence features (mean pooling with attention mask)
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(projected)
                sum_embeddings = torch.sum(projected * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                # Simple mean pooling
                pooled = projected.mean(dim=1)

            return pooled

    def tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (if None, use self.max_length)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length

        padding_type = "max_length" if getattr(self, "padding", True) else False

        return self.tokenizer(
            texts,
            padding=padding_type,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )


class BERTLanguageEncoder(nn.Module):
    """
    BERT-based language encoder.
    
    Uses BERT model for encoding language instructions.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 256,
        output_sequence: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0,
        max_length: int = 77,
        **kwargs,
    ):
        """
        Initialize BERT language encoder.
        
        Args:
            model_name: BERT model name (default: bert-base-uncased)
            output_dim: Output feature dimension
            output_sequence: If True, return sequence features; if False, return pooled features
            freeze_backbone: Whether to freeze BERT backbone
            dropout: Dropout probability
            max_length: Maximum sequence length for tokenization (default: 77)
        """
        super().__init__()

        try:
            from transformers import BertTokenizer, BertModel
        except ImportError:
            raise ImportError("transformers library is required for BERTLanguageEncoder")

        self.output_sequence = output_sequence
        self.output_dim = output_dim
        self.max_length = max_length

        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get BERT hidden dimension
        self.bert_hidden_dim = self.encoder.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(self.bert_hidden_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            
        Returns:
            If output_sequence=False: features [B, D]
            If output_sequence=True: (language_seq [B, L, D], mask [B, L])
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        projected = self.output_proj(hidden_states)
        
        if self.output_sequence:
            # Return sequence features
            return projected
        else:
            # Pool sequence features (mean pooling with attention mask)
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(projected)
                sum_embeddings = torch.sum(projected * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                # Simple mean pooling
                pooled = projected.mean(dim=1)

            return pooled

    def tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (if None, use self.max_length)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length

        padding_type = "max_length" if getattr(self, "padding", True) else False

        return self.tokenizer(
            texts,
            padding=padding_type,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )


def test_language_encoder(encoder_name: str, encoder: nn.Module, input_ids: torch.Tensor, 
                         attention_mask: Optional[torch.Tensor] = None, lengths: Optional[torch.Tensor] = None):
    """
    Test a language encoder and print results.
    
    Args:
        encoder_name: Name of the encoder for display
        encoder: The encoder instance to test
        input_ids: Input token IDs
        attention_mask: Attention mask (for transformer-based encoders)
        lengths: Sequence lengths (for RNN-based encoders)
    """
    print(f"\n--- Testing {encoder_name} ---")
    
    try:
        # Test with output_sequence=False (if supported)
        if hasattr(encoder, 'output_sequence'):
            encoder.output_sequence = False
        
        if attention_mask is not None:
            # Transformer-based encoder
            features = encoder(input_ids, attention_mask)
        else:
            # RNN-based encoder
            features = encoder(input_ids, lengths)
        
        if isinstance(features, tuple):
            features = features[0]
        print(f"  Pooled features shape: {features.shape}")
        
        # Test with output_sequence=True (if supported)
        if hasattr(encoder, 'output_sequence'):
            encoder.output_sequence = True
            
            if attention_mask is not None:
                result = encoder(input_ids, attention_mask)
            else:
                result = encoder(input_ids, lengths)
            
            if isinstance(result, tuple):
                seq_features, mask = result
                print(f"  Sequence features shape: {seq_features.shape}")
                print(f"  Mask shape: {mask.shape}")
            else:
                print(f"  Sequence features shape: {result.shape}")
        
    except Exception as e:
        print(f"  Error testing {encoder_name}: {e}")


if __name__ == "__main__":
    # Test language encoders
    print("Testing language encoders...")

    # Test texts
    test_texts = ["pick up the red block", "open the drawer", "place the cup on the table", "close the door"]
    max_seq_length = 32
    
    print(f"Test texts: {test_texts}")
    print(f"Max sequence length: {max_seq_length}")
    
    try:
        # === Test GRU Encoder ===
        # Test GRU encoder (now with built-in tokenizer)
        gru_encoder = GRULanguageEncoder(output_dim=256)
        
        # Use GRU encoder's built-in tokenizer
        gru_tokens = gru_encoder.tokenize(test_texts, max_length=max_seq_length)
        gru_input_ids = gru_tokens["input_ids"]
        gru_attention_mask = gru_tokens["attention_mask"]
        
        test_language_encoder("GRU Encoder", gru_encoder, gru_input_ids, gru_attention_mask)
        
        # === Test T5 Encoder ===
        try:
            # Create T5 encoder
            print("\nDownloading T5 model (this may take a while on first run)...")
            t5_encoder = T5LanguageEncoder(output_dim=256)
            
            # Tokenize with T5 tokenizer
            t5_tokens = t5_encoder.tokenize(test_texts, max_length=max_seq_length)
            t5_input_ids = t5_tokens["input_ids"]
            t5_attention_mask = t5_tokens["attention_mask"]
            
            # Test T5 encoder
            test_language_encoder("T5 Encoder", t5_encoder, t5_input_ids, t5_attention_mask)
        except Exception as e:
            print(f"\nSkipping T5 encoder test due to error: {e}")
            print("This is likely due to network issues or missing model files.")
        
        # === Test CLIP Encoder ===
        try:
            # Create CLIP encoder
            print("\nDownloading CLIP model (this may take a while on first run)...")
            clip_encoder = CLIPLanguageEncoder(output_dim=256)
            
            # Tokenize with CLIP tokenizer
            clip_tokens = clip_encoder.tokenize(test_texts, max_length=max_seq_length)
            clip_input_ids = clip_tokens["input_ids"]
            clip_attention_mask = clip_tokens["attention_mask"]
            
            # Test CLIP encoder
            test_language_encoder("CLIP Encoder", clip_encoder, clip_input_ids, clip_attention_mask)
        except Exception as e:
            print(f"\nSkipping CLIP encoder test due to error: {e}")
            print("This is likely due to network issues or missing model files.")
        
        # === Test BERT Encoder ===
        try:
            # Create BERT encoder
            print("\nDownloading BERT model (this may take a while on first run)...")
            bert_encoder = BERTLanguageEncoder(output_dim=256)
            
            # Tokenize with BERT tokenizer
            bert_tokens = bert_encoder.tokenize(test_texts, max_length=max_seq_length)
            bert_input_ids = bert_tokens["input_ids"]
            bert_attention_mask = bert_tokens["attention_mask"]
            
            # Test BERT encoder
            test_language_encoder("BERT Encoder", bert_encoder, bert_input_ids, bert_attention_mask)
        except Exception as e:
            print(f"\nSkipping BERT encoder test due to error: {e}")
            print("This is likely due to network issues or missing model files.")
        
        print("\n" + "="*50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

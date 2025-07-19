"""
Autoregressive wrapper for Gemini models
Provides autoregressive generation capabilities for transformer models
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Callable, Any, Dict, List, Tuple
import math


class AutoregressiveWrapper(nn.Module):
    """
    Autoregressive wrapper for transformer models.

    This wrapper enables autoregressive generation for any transformer model
    by handling the sequential token generation process.

    Args:
        model: The underlying transformer model
        ignore_index: Token index to ignore in loss calculation (default: -100)
        pad_value: Padding token value (default: 0)
        mask_prob: Probability of masking tokens during training (default: 0.0)
        label_smoothing: Label smoothing factor (default: 0.0)
        max_seq_len: Maximum sequence length (default: None)
    """

    def __init__(
        self,
        model: nn.Module,
        ignore_index: int = -100,
        pad_value: int = 0,
        mask_prob: float = 0.0,
        label_smoothing: float = 0.0,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.ignore_index = ignore_index
        self.pad_value = pad_value
        self.mask_prob = mask_prob
        self.label_smoothing = label_smoothing
        self.max_seq_len = max_seq_len

        # Get model dimensions
        if hasattr(model, 'num_tokens'):
            self.num_tokens = model.num_tokens
        elif hasattr(model, 'to_logits') and hasattr(model.to_logits, 'out_features'):
            self.num_tokens = model.to_logits.out_features
        else:
            self.num_tokens = None

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        return_loss: bool = False,
        **kwargs
    ) -> Tensor:
        """
        Forward pass for autoregressive generation.

        Args:
            x: Input token sequence [batch, seq_len]
            context: Optional context tensor for conditioning
            return_loss: Whether to return loss for training
            **kwargs: Additional arguments passed to the model

        Returns:
            Model output logits or loss
        """
        if return_loss:
            return self._forward_with_loss(x, context=context, **kwargs)
        else:
            return self._forward_inference(x, context=context, **kwargs)

    def _forward_with_loss(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass during training with loss calculation."""
        # Split input and target
        inp, target = x[:, :-1], x[:, 1:]

        # Apply random masking if specified
        if self.training and self.mask_prob > 0:
            inp = self._apply_random_masking(inp)

        # Forward through model
        logits = self.model(inp, context=context, **kwargs)

        # Calculate loss
        loss = self._calculate_loss(logits, target)

        return loss

    def _forward_inference(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass during inference."""
        return self.model(x, context=context, **kwargs)

    def _apply_random_masking(self, x: Tensor) -> Tensor:
        """Apply random masking to input tokens."""
        mask = torch.rand_like(x, dtype=torch.float) < self.mask_prob
        x = x.masked_fill(mask, self.pad_value)
        return x

    def _calculate_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Calculate cross-entropy loss with optional label smoothing."""
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        if self.label_smoothing > 0:
            loss = self._label_smoothed_cross_entropy(logits, targets)
        else:
            loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=self.ignore_index
            )

        return loss

    def _label_smoothed_cross_entropy(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Calculate label-smoothed cross-entropy loss."""
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        vocab_size = logits.size(-1)
        smoothed_targets = torch.zeros_like(log_probs)

        # Fill with uniform distribution
        smoothed_targets.fill_(self.label_smoothing / (vocab_size - 1))

        # Set correct class probability
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)

        # Ignore padding tokens
        mask = targets != self.ignore_index
        smoothed_targets = smoothed_targets[mask]
        log_probs = log_probs[mask]

        loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()

        return loss

    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        seq_len: int,
        context: Optional[Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        filter_logits_fn: Optional[Callable] = None,
        eos_token: Optional[int] = None,
        **kwargs
    ) -> Tensor:
        """
        Generate text autoregressively.

        Args:
            start_tokens: Starting token sequence [batch, start_len]
            seq_len: Maximum sequence length to generate
            context: Optional context for conditioning
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            filter_logits_fn: Custom logits filtering function
            eos_token: End-of-sequence token to stop generation
            **kwargs: Additional arguments

        Returns:
            Generated token sequence [batch, total_len]
        """
        self.eval()

        device = start_tokens.device
        batch_size, start_len = start_tokens.shape

        # Initialize output sequence
        output = start_tokens.clone()

        for _ in range(seq_len):
            # Get current sequence (limit to max_seq_len if specified)
            if self.max_seq_len is not None:
                current_seq = output[:, -self.max_seq_len:]
            else:
                current_seq = output

            # Forward pass
            logits = self.model(current_seq, context=context, **kwargs)

            # Get logits for last position
            last_logits = logits[:, -1, :]

            # Apply custom filtering if provided
            if filter_logits_fn is not None:
                last_logits = filter_logits_fn(last_logits, output)

            # Apply temperature
            if temperature != 1.0:
                last_logits = last_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                last_logits = self._top_k_filter(last_logits, top_k)

            # Apply top-p filtering
            if top_p is not None:
                last_logits = self._top_p_filter(last_logits, top_p)

            # Sample next token
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to output
            output = torch.cat([output, next_token], dim=-1)

            # Check for EOS token
            if eos_token is not None and (next_token == eos_token).all():
                break

        return output

    def _top_k_filter(self, logits: Tensor, k: int) -> Tensor:
        """Apply top-k filtering to logits."""
        top_k_logits, _ = torch.topk(logits, k, dim=-1)
        min_top_k = top_k_logits[:, -1:].expand_as(logits)
        return torch.where(logits < min_top_k, torch.full_like(logits, -float('inf')), logits)

    def _top_p_filter(self, logits: Tensor, p: float) -> Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p

        # Shift indices to keep first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Create mask for original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )

        logits = logits.masked_fill(indices_to_remove, -float('inf'))
        return logits

    @torch.no_grad()
    def sample(
        self,
        start_tokens: Tensor,
        seq_len: int,
        context: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Simple sampling interface (alias for generate with default parameters).

        Args:
            start_tokens: Starting tokens
            seq_len: Length to generate
            context: Optional context
            **kwargs: Additional arguments

        Returns:
            Generated sequence
        """
        return self.generate(
            start_tokens=start_tokens,
            seq_len=seq_len,
            context=context,
            **kwargs
        )

    def get_model(self) -> nn.Module:
        """Get the underlying model."""
        return self.model

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Get named model parameters."""
        return self.model.named_parameters()

    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dict."""
        return self.model.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        super().eval()
        self.model.eval()
        return self

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.model.to(device)
        return self

    def cuda(self):
        """Move model to CUDA."""
        super().cuda()
        self.model.cuda()
        return self

    def cpu(self):
        """Move model to CPU."""
        super().cpu()
        self.model.cpu()
        return self
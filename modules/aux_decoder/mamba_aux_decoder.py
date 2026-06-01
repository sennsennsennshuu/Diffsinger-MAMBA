"""
SSM-based auxiliary decoder for DiffSinger's shallow diffusion.
Replaces the ConvNeXt-based auxiliary decoder with bidirectional Mamba blocks.

This decoder produces a low-quality mel-spectrogram prediction that serves
as the starting point for the diffusion process (shallow diffusion).

Architecture:
    Input → Conv1d(H→H) → [BiMambaBlock × N] → LayerNorm → Conv1d(H→n_mels)

Reference: https://github.com/hustvl/Vim (bidirectional Mamba pattern)
"""

import torch
import torch.nn as nn
from modules.commons.ssm_layers import BiMambaBlock


class MambaAuxDecoder(nn.Module):
    """
    SSM-based auxiliary decoder producing mel-spectrogram from encoded features.
    
    Args:
        in_dims: input feature dimension (hidden_size from encoder)
        out_dims: output mel bins (e.g., 128)
        num_channels: internal channel dimension
        num_layers: number of bidirectional Mamba blocks
        d_state: SSM state size (Mamba3 default: 128)
        d_conv: local conv kernel size
        expand: channel expansion factor
        kernel_size: input/output conv kernel size
        dropout_rate: dropout rate for Mamba blocks
    """
    def __init__(self, in_dims, out_dims, *,
                 num_channels=512, num_layers=6, kernel_size=7,
                 d_state=128, d_conv=4, expand=2, dropout_rate=0.1):
        super().__init__()
        
        # Input projection
        self.in_conv = nn.Conv1d(
            in_dims, num_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        
        # Bidirectional Mamba blocks
        self.blocks = nn.ModuleList([
            BiMambaBlock(
                c=num_channels,
                num_heads=1,  # not used by Mamba
                dropout=dropout_rate,
                attention_dropout=0.0,
                relu_dropout=dropout_rate,
                kernel_size=7,
                act='silu',
                rotary_embed=None
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_conv = nn.Conv1d(
            num_channels, out_dims, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        # Zero-initialize output for stable training start
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, infer=False):
        """
        Args:
            x: (B, T, H) encoded features from encoder
            infer: boolean flag (unused, kept for interface compatibility)
        
        Returns:
            (B, T, out_dims) mel-spectrogram prediction
        """
        # Convert to channel-first for convolution
        x = x.transpose(1, 2)  # (B, H, T)
        x = self.in_conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        
        # Bidirectional Mamba processing
        for block in self.blocks:
            x = block(x)  # padding_mask=None since all frames are valid
        
        # Output projection
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.out_conv(x)
        x = x.transpose(1, 2)  # (B, T, out_dims)
        
        return x

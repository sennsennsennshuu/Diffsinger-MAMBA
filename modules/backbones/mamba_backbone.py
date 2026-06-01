"""
SSM Diffusion Backbone for DiffSinger.
Re-exports MambaBackbone and MambaResidualBlock from the SSM layers module.

This backbone replaces both LYNXNet and WaveNet with a unified SSM-based
diffusion backbone using Mamba2 (larger state space) for long-range modeling.

Usage:
    from modules.backbones.mamba_backbone import MambaBackbone
"""

from modules.commons.ssm_layers import MambaBackbone, MambaResidualBlock

__all__ = ['MambaBackbone', 'MambaResidualBlock']

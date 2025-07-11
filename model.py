"""
model.py

This module defines the neural network architecture used for downscaling groundwater storage data.
It includes custom modules such as SEBlock, ResidualBlock, GatingBlock, and the main ResNetDownscalingModel class.
The model is designed with multiple heads for output and incorporates residual connections, squeeze-and-excitation blocks,
and gating mechanisms to enhance feature representation.

Author: [Shuitao Guo]
Date: [2025-07-11]
"""

import torch
import torch.nn as nn



class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block to adaptively recalibrate channel-wise feature responses.
    
    This block performs global average pooling followed by two convolutional layers
    and a sigmoid activation to generate channel attention weights.
    """
    ...

class ResidualBlock(nn.Module):
    """
    Residual block implementing skip connections for improved gradient flow.
    
    Consists of two convolutional layers with ReLU activations and adds input to the output.
    """
    ...

class GatingBlock(nn.Module):
    """
    Gating block that modulates feature maps using a learned gate mechanism.
    
    It computes a sigmoid gate from the globally averaged input features and scales the input accordingly.
    """
    ...

class ResNetDownscalingModel(nn.Module):
    """
    ResNet-based model for downscaling groundwater storage data.
    
    Combines multiple feature extraction blocks, SE and gating modules, and multi-head outputs.
    Designed to fuse different input features and produce high-resolution predictions.
    """
    ...


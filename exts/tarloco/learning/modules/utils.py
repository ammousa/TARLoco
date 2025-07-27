#  Copyright 2025 University of Manchester, Amr Mousa
#  SPDX-License-Identifier: CC-BY-SA-4.0

from typing import List, Optional


import torch.nn as nn


def mlp_factory(activation, input_dims, out_dims, hidden_dims, last_act=False):
    """
    Creates a multi-layer perceptron (MLP) model using PyTorch's `nn.Sequential`.

    Args:
        activation (torch.nn.Module): The activation function to use between layers.
        input_dims (int): The number of input features for the first layer.
        out_dims (int): The number of output features for the final layer. If None, no output layer is added.
        hidden_dims (list[int]): A list of integers specifying the number of units in each hidden layer.
        last_act (bool, optional): If True, applies the activation function after the final layer. Defaults to False.

    Returns:
        torch.nn.Sequential: A sequential container representing the MLP model.
    """
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)
    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    if last_act:
        layers.append(activation)

    return nn.Sequential(*layers)


def tcn_factory(
    input_channels: int,
    output_dims: int,
    num_hist: int,
    hidden_channels: Optional[List[int]] = None,
    kernel_sizes: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
    activation: nn.Module = nn.ReLU(),
    last_act: bool = True,
) -> nn.Sequential:
    """
    Constructs a Temporal Convolutional Network (TCN) encoder using Conv1D layers, followed by a Linear layer.
    The TCN processes temporal data with configurable hidden layers, kernel sizes, and strides, applying
    activation functions after each Conv1D layer.

    If `hidden_channels`, `kernel_sizes`, and `strides` are not provided, a default 2-layer configuration is used.

    Parameters:
        input_channels (int): Number of input channels for the first Conv1D layer (e.g., feature dimensions per time step).
        output_dims (int): Dimension of the final output (e.g., latent space dimensions).
        num_hist (int): Temporal dimension (sequence length) of the input data.
        hidden_channels (Optional[List[int]]): List specifying the number of channels for each hidden Conv1D layer.
                                               Defaults to [32, 16] if not provided.
        kernel_sizes (Optional[List[int]]): List specifying the kernel sizes for each Conv1D layer.
                                            Defaults to [6, 4] if not provided.
        strides (Optional[List[int]]): List specifying the stride values for each Conv1D layer.
                                       Defaults to [3, 2] if not provided.
        activation (nn.Module): Activation function to apply after each Conv1D layer and optionally after the Linear layer.
                                Defaults to nn.ReLU().
        last_act (bool): Whether to apply the activation function after the final Linear layer. Defaults to True.

    Returns:
        nn.Sequential: A Sequential model representing the TCN encoder, consisting of Conv1D layers, activations,
                       a Flatten layer, and a Linear layer.

    Notes:
        - The temporal dimension (`num_hist`) is dynamically adjusted after each Conv1D layer based on the kernel size
          and stride, ensuring compatibility with the final Linear layer.
        - Ensure that `hidden_channels`, `kernel_sizes`, and `strides` have the same length if provided.
    """
    layers = []
    current_channels = input_channels
    t_len = num_hist  # Effective temporal dimension tracking

    # Use default 2-layer architecture if no custom config is provided
    if hidden_channels is None:
        hidden_channels = [32, 16]
    if kernel_sizes is None:
        kernel_sizes = [6, 4]
    if strides is None:
        strides = [3, 2]

    # Ensure all lists have the same length
    assert (
        len(hidden_channels) == len(kernel_sizes) == len(strides)
    ), "[ERROR]: In the TCN: hidden_channels, kernel_sizes, and strides must have the same length"

    # Build each Conv -> Activation block
    for ch_out, k_size, stride in zip(hidden_channels, kernel_sizes, strides):
        conv = nn.Conv1d(in_channels=current_channels, out_channels=ch_out, kernel_size=k_size, stride=stride)
        layers.append(conv)
        layers.append(activation)

        # Update the temporal dimension
        t_len = (t_len - k_size) // stride + 1
        current_channels = ch_out

    # Flatten before Linear layer
    layers.append(nn.Flatten())

    # Construct the final Linear layer
    final_in_features = current_channels * t_len
    layers.append(nn.Linear(final_in_features, output_dims))

    # Optionally add the last activation
    if last_act:
        layers.append(activation)

    return nn.Sequential(*layers)

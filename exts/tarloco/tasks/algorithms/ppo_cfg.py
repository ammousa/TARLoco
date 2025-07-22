# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the isaaclab repository:
# https://github.com/isaac-sim/IsaacLab/
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from __future__ import annotations

from typing import List, Literal

from isaaclab.utils import configclass
from omegaconf import MISSING


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = MISSING
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    clip_action: float = MISSING
    """ The action clipping value. Default is 100."""

    squash_mode: str = MISSING
    """The squash mode for the policy actions."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoSlrActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    num_hist: int = 1
    """The number of history steps to consider."""

    latent_dims: int = MISSING
    """The latent dimensions for the actor-critic networks."""

    mlp_encoder_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    trans_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the trans dynamics network."""


@configclass
class RslRlRnnPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the LSTM-PPO actor-critic networks."""

    class_name: str = MISSING
    """The policy class name. Default is ActorCritic."""

    arch_type: str = MISSING
    """The architecture type (e.g., 'direct', 'residual')."""

    rnn_type: str = MISSING
    """The type of RNN to use (e.g., 'lstm', 'gru')."""

    rnn_hidden_dims: list = MISSING
    """The dimensions of hidden units in each RNN layer."""

    rnn_out_features: int = 0
    """The number of output units for the LSTM augmentation. Default is 0."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = MISSING
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    lr_min: float = MISSING
    """The learning rate for the policy."""

    lr_max: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    optimizer: str = MISSING
    """The optimizer to use."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = MISSING
    """The device for the rl-agent. Default is cuda."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration (considered teacher policy in distill env)."""

    student_policy: TransferStudentCfg = MISSING
    """The student policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    wandb_entity: str = "amrmousa-m"
    """The wandb project name. Default is "isaaclab"."""

    note: str = ""
    """The note for the task to be sent to wandb if used."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    note: str = ""
    """The note for the task to be sent to wandb if used."""


# =====================
#       TARLoco
# =====================


@configclass
class RslRlRnnPpoTarActorCriticCfg(RslRlRnnPpoActorCriticCfg):
    """Configuration for the RNN-PPO actor-critic networks."""

    num_hist: int = 1
    """The number of history steps to consider."""

    latent_dims: int = MISSING
    """The latent dimensions for the actor-critic networks."""

    trans_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the trans network."""

    aux_loss_coef: list[float] = MISSING
    """The coefficient for the auxiliary loss."""


@configclass
class RslRlPpoTarAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    aux_loss_coef: list[float] = MISSING
    """The coefficient for the auxiliary loss."""

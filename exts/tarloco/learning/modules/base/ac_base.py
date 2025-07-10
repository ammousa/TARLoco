# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the rsl_rl repository:
# https://github.com/leggedrobotics/rsl_rl
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation
from torch.distributions import Normal

from exts.tarloco.utils import unpad_trajectories


class ActorCriticMlp(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",  # 'tanh' or 'clip'
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.clip_action = clip_action
        self.squash_mode = squash_mode
        self.num_policy_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        self.actor = AcNet(
            True,
            num_actions,
            num_actor_obs,
            actor_hidden_dims,
            activation,
            0.6,
        )
        self.critic = AcNet(
            False,
            1,  # Critic output is a single value
            num_critic_obs,
            critic_hidden_dims,
            activation,
            0.6,
        )

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # Disable args validation for speedup
        Normal.set_default_validate_args = False  # type: ignore
        self.nan_detected = False

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"[INFO]: Actor MLP: {self.actor}")
        print(f"[INFO]: Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass  # Implement if necessary

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def extract(self, x):
        """Maintain the SLR interface"""
        return x

    def update_distribution(self, mean):
        if not torch.isnan(mean).any() and not self.nan_detected:
            try:
                self.distribution = Normal(mean, mean * 0.0 + torch.clamp(self.std, min=1e-6))
            except Exception as e:
                print(f"[ERROR]: {e}")
                print(f"[ERROR]: mean: {mean}")
                print(f"[ERROR]: std: {self.std}")
                self.nan_detected = True

    def get_actions_log_prob(self, actions):
        if self.squash_mode == "tanh":
            # Invert the tanh transform:
            # if action = tanh(raw_action) * clip_action,
            # we first scale back: unscaled_action = action / self.clip_action
            unscaled_action = torch.clamp(actions / self.clip_action, -1.0 + 1e-6, 1.0 - 1e-6)
            # Then arc-tanh (better to use formula to avoid numerical instability if using torch.atanh):
            raw_action = 0.5 * torch.log((1 + unscaled_action) / (1 - unscaled_action + 1e-6) + 1e-6)
            # log_prob under the original Normal
            log_prob_raw = self.distribution.log_prob(raw_action).sum(dim=-1)
            # Subtract the log |derivative of tanh|:
            correction = torch.log(1 - unscaled_action.pow(2) + 1e-6).sum(dim=-1)
            return log_prob_raw - correction
        else:
            return self.distribution.log_prob(actions).sum(dim=-1)

    def encode(self):
        raise NotImplementedError

    def encode_critic(self):
        raise NotImplementedError

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        mean = self.actor(observations)
        self.update_distribution(mean)
        if self.squash_mode == "tanh":
            return torch.tanh(self.distribution.sample()) * self.clip_action
        else:
            return torch.clamp(self.distribution.sample(), -self.clip_action, self.clip_action)

    def act_inference(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        actions_mean = self.actor(observations)
        if self.squash_mode == "tanh":
            return torch.tanh(actions_mean) * self.clip_action
        else:
            return torch.clamp(actions_mean, -self.clip_action, self.clip_action)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        value = self.critic(critic_observations)
        return value


class ActorCriticRnn(ActorCriticMlp):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",  # 'tanh' or 'clip'
        arch_type="augmented",
        rnn_type="lstm",
        rnn_hidden_dims=[256],
        rnn_out_features=0,
        **kwargs,
    ):

        # build config
        self.arch_type = arch_type
        self.rnn_type = rnn_type
        self.rnn_hidden_dims = rnn_hidden_dims
        self.rnn_out_features = rnn_out_features
        self.rnn_num_layers = len(self.rnn_hidden_dims)
        self.rnn_hidden_size = self.rnn_hidden_dims[0]

        # Architectures
        allowed_arch = ["augmented", "residual", "direct", "integrated"]
        self.arch = arch_type
        if arch_type not in allowed_arch:
            raise ValueError(f"[ERROR] Invalid architecture: {arch_type}. Allowed options are: {allowed_arch}")
        if arch_type == "residual":
            rnn_out_features = num_actor_obs
        assert rnn_out_features >= 0, "[ERROR] rnn_out_features must be a non-negative integer"

        self.num_policy_obs = num_actor_obs + (rnn_out_features if self.arch == "augmented" else 0)
        self.num_critic_obs = num_critic_obs + (rnn_out_features if self.arch == "augmented" else 0)

        super().__init__(
            num_actor_obs=self.num_policy_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            **kwargs,
        )

        self.num_latents = rnn_out_features
        self.memory = Memory(
            input_size=num_actor_obs,
            type=rnn_type,
            arch=arch_type,
            num_layers=self.rnn_num_layers,
            hidden_size=self.rnn_hidden_size,
            out_features=rnn_out_features,
        )
        self.reset()

    def post_init(self):
        super().post_init()
        print(
            f"[INFO]: Using ActorCriticRnn with {self.rnn_type} {self.arch} architecture, {self.rnn_num_layers} layers"
            f" and size {self.rnn_hidden_size}"
        )
        print(f"[INFO]: Encoder RNN: {self.memory}")

    def reset(self, dones=None):
        if dones is None:
            self.set_hidden_states(None, None)
        else:
            hidden_states = self.get_hidden_states()
            self.set_hidden_states(
                self.memory.reset(dones, hidden_states[0]), self.memory.reset(dones, hidden_states[1])
            )

    def get_hidden_states(self):
        return self.hidden_state_a, self.hidden_state_c

    def set_hidden_states(self, hidden_state_a, hidden_state_c):
        self.hidden_state_a = hidden_state_a
        self.hidden_state_c = hidden_state_c

    def _get_augmented_obs(self, observations, features, masks=None):
        # Align shapes based on the original shape of features
        if len(features.shape) == 2 and len(observations.shape) == 3:
            observations = observations.squeeze(0)
        elif len(features.shape) == 3 and len(observations.shape) == 2:
            observations = observations.unsqueeze(0)
        if masks is not None:
            observations = unpad_trajectories(observations, masks)
        augmented_obs = torch.cat((observations, features), dim=-1)
        # Adjust the return shape to match the original features shape
        if len(features.shape) == 2:
            augmented_obs = augmented_obs.squeeze(0)
        return augmented_obs

    def encode(self, observations, **kwargs):
        hidden_states = kwargs.get("hidden_states", None)
        masks = kwargs.get("masks", None)
        if hidden_states is not None and masks is not None and hidden_states[0].shape[1] != observations.shape[1]:
            observations = unpad_trajectories(observations, masks)
        z, hidden_states = self.memory(observations, masks, hidden_states)
        return z, hidden_states

    def encode_critic(self, observations, **kwargs):
        return self.encode(observations, **kwargs)

    def act(self, observations, **kwargs):
        masks = kwargs.get("masks", None)
        z, hidden_states = self.encode(observations, **kwargs)
        input_a = self._get_augmented_obs(observations, z, masks) if self.arch == "augmented" else z
        mean = self.actor(input_a.squeeze(0))
        self.update_distribution(mean)
        if self.squash_mode == "tanh":
            return torch.tanh(self.distribution.sample()) * self.clip_action, hidden_states
        elif self.squash_mode == "clip":
            return torch.clamp(self.distribution.sample(), -self.clip_action, self.clip_action), hidden_states

    def act_inference(self, observations, **kwargs):
        z, _ = self.encode(observations).squeeze(0)
        input_a = self._get_augmented_obs(observations, z) if self.arch == "augmented" else z
        actions_mean = self.actor(input_a)
        if self.squash_mode == "tanh":
            return torch.tanh(actions_mean) * self.clip_action
        elif self.squash_mode == "clip":
            return torch.clamp(actions_mean, -self.clip_action, self.clip_action)

    def evaluate(self, critic_observations, **kwargs):
        masks = kwargs.get("masks", None)
        z_c, hidden_states = self.encode_critic(critic_observations, **kwargs)
        features_c = (
            self._get_augmented_obs(critic_observations, z_c.detach(), masks)
            if self.arch == "augmented"
            else z_c.detach()
        )
        value = self.critic(features_c.squeeze(0))
        return value, hidden_states


class ActorCriticRnnDblEnc(ActorCriticRnn):
    """
    Child class reintroducing separate RNNs for actor and critic.
    """

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",  # 'tanh' or 'clip'
        arch_type="augmented",
        rnn_type="lstm",
        rnn_hidden_dims=[256],
        rnn_out_features=0,
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            arch_type=arch_type,
            rnn_type=rnn_type,
            rnn_hidden_dims=rnn_hidden_dims,
            rnn_out_features=rnn_out_features,
            **kwargs,
        )
        # Override single memory with two separate ones
        del self.memory
        self.memory_a = Memory(
            input_size=num_actor_obs,
            type=rnn_type,
            arch=arch_type,
            num_layers=len(rnn_hidden_dims),
            hidden_size=rnn_hidden_dims[0],
            out_features=rnn_out_features,
        )
        self.memory_c = Memory(
            input_size=num_critic_obs,
            type=rnn_type,
            arch=arch_type,
            num_layers=len(rnn_hidden_dims),
            hidden_size=rnn_hidden_dims[0],
            out_features=rnn_out_features,
        )

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"[INFO]: Actor MLP: {self.actor}")
        print(f"[INFO]: Critic MLP: {self.critic}")
        print(
            f"[INFO]: Using ActorCriticRnn with {self.rnn_type} {self.arch} architecture, {self.rnn_num_layers} layers"
            f" and size {self.rnn_hidden_size}"
        )
        print(f"[INFO]: Actor Encoder RNN: {self.memory_a}")
        print(f"[INFO] Created separate critic encoder RNN: {self.memory_c}")

    def reset(self, dones=None):
        if dones is None:
            self.set_hidden_states(None, None)
        else:
            hidden_states = self.get_hidden_states()
            self.set_hidden_states(
                self.memory_a.reset(dones, hidden_states[0]), self.memory_c.reset(dones, hidden_states[1])
            )

    def encode(self, observations, **kwargs):
        hidden_states = kwargs.get("hidden_states", None)
        masks = kwargs.get("masks", None)
        if hidden_states is not None and masks is not None:
            if hidden_states[0].shape[1] != observations.shape[1]:
                observations = unpad_trajectories(observations, masks)
        z, hidden_states = self.memory_a(observations, masks, hidden_states)
        return z, hidden_states

    def encode_critic(self, observations, **kwargs):
        hidden_states = kwargs.get("hidden_states", None)
        masks = kwargs.get("masks", None)
        if hidden_states is not None and masks is not None:
            if hidden_states[0].shape[1] != observations.shape[1]:
                observations = unpad_trajectories(observations, masks)
        z, hidden_states = self.memory_c(observations, masks, hidden_states)
        return z, hidden_states


class AcNet(torch.nn.Module):
    """Custom network."""

    def __init__(
        self,
        is_policy: bool,
        num_out: int,
        num_obs: int,
        hidden_dims: list[int],
        activation: str,
        init_weight: float = 0.1,
    ):
        super().__init__()

        self.is_policy = is_policy
        self.num_out = num_out
        self.num_obs = num_obs

        activation = resolve_nn_activation(activation)  # type: ignore

        # TODO: (MIRROR_loss) add support for different initializations and parametrize the init_weight
        mlp_layers = []
        mlp_layers.append(nn.Linear(self.num_obs, hidden_dims[0]))
        mlp_layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                last = nn.Linear(hidden_dims[layer_index], num_out)
                if is_policy:
                    # Reference: https://arxiv.org/abs/2006.05990
                    # Initialize last layer with small weights to reduce action dependence on observation
                    nn.init.orthogonal_(last.weight, gain=0.01)  # type: ignore
                    nn.init.zeros_(last.bias)  # Bias is set to zero for unbiased initial actions
                mlp_layers.append(last)
            else:
                layer = nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1])
                # Reference: https://arxiv.org/abs/2006.05990
                # Use orthogonal init with gain=1.0 to stabilise gradient propagation
                nn.init.orthogonal_(layer.weight, gain=init_weight)  # type: ignore
                nn.init.zeros_(layer.bias)  # Zero bias avoids introducing unintended shifts
                mlp_layers.append(layer)
                mlp_layers.append(activation)

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._safe_fw(self.mlp(input))

    def _safe_fw(self, mean: torch.Tensor) -> torch.Tensor:
        """Checks for NaN values in mean and returns a random action if NaN is detected."""
        if torch.isnan(mean).any():
            print(f"[WARNING]: NaN detected in FW pass of {self.__class__.__name__}, returning random action instead!")
            self.nan_detected = True
            return torch.rand_like(mean)
        else:
            return mean


class Memory(nn.Module):
    def __init__(
        self,
        input_size,
        type="lstm",
        arch="integrated",
        num_layers=1,
        hidden_size=256,
        out_features=0,
    ):
        super().__init__()
        self.arch = arch
        self.rnn_type = type.lower()
        supported_types = ["gru", "lstm"]
        assert (
            self.rnn_type in supported_types
        ), f"[ERROR]: RNN type {self.rnn_type} not in {supported_types} of Memory module"
        assert out_features >= 0, "[ERROR]: out_features must be a non-negative integer"
        rnn_cls = getattr(nn, self.rnn_type.upper())
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        print(f"[INFO]: Using {self.rnn_type} in Memory module")

        if arch != "integrated":
            self.feature_extractor = nn.Linear(hidden_size, out_features)

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_states is None:
                raise ValueError("[ERROR]: Hidden states not passed to memory module during policy update")
            out, hidden_states = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            out, hidden_states = self.rnn(input.unsqueeze(0) if input.dim() == 2 else input, hidden_states)

        if self.arch != "integrated":
            out = self.feature_extractor(out)

        if self.arch == "residual":
            if batch_mode:
                out = unpad_trajectories(input, masks) + out
            else:
                out = input.unsqueeze(0) + out
        return out, hidden_states

    def reset(self, dones, hidden_states):
        for hidden_state in hidden_states:  # loop over cell and hidden states
            hidden_state[..., dones.bool(), :] = 0.0
        return hidden_states

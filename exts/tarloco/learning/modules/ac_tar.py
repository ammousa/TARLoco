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

from typing import List

import torch
from rsl_rl.utils import resolve_nn_activation

from exts.tarloco.learning.modules.utils import mlp_factory, tcn_factory
from exts.tarloco.utils import unpad_trajectories

from .ac_slr import ActorCriticMlpSlrDblEnc
from .base.ac_base import AcNet, ActorCriticRnn, Memory


class ActorCriticTar(ActorCriticMlpSlrDblEnc):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist: int,
        num_hist_short: int = 4,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        mlp_encoder_dims: list[int] = [256, 128, 64],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist=num_hist,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            mlp_encoder_dims=mlp_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
            **kwargs,
        )
        self.num_hist_short = num_hist_short
        self.vel_estimator = mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=self.num_obs_h1 * num_hist_short + latent_dims,
            out_dims=3,
            hidden_dims=vel_encoder_dims,
        )
        self.encoder = mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=num_actor_obs,
            out_dims=latent_dims,
            hidden_dims=mlp_encoder_dims,
        )

        self.encoder_critic = mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=num_critic_obs,
            out_dims=latent_dims,
            hidden_dims=mlp_encoder_dims,
        )

        self.actor = AcNet(
            is_policy=True,
            num_out=num_actions,
            num_obs=self.num_obs_h1 + self.num_latents + 3,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        self.critic = AcNet(
            is_policy=False,
            num_out=1,  # Critic output is a single value
            num_obs=self.num_obs_h1 + self.num_latents + 3,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

    def post_init(self):
        super().post_init()
        print(f"[INFO]: Velocity Estimator: {self.vel_estimator}")

    def extract(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        prop = observations[..., -1, :]
        hist_short = observations[..., -self.num_hist_short :, :]
        hist = observations
        return hist, prop, hist_short

    def extract_critic(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        prop = observations[..., 3:48]  # [Batch, Time, Dim]
        vel = observations[..., 0:3]
        full_obs = observations
        return full_obs, prop, vel

    def encode(self, obs_tuple, **kwargs):
        obs_hist, _, obs_hist_short = obs_tuple
        z = self.encoder(obs_hist.view(*obs_hist.shape[:-2], -1))
        vel = self.vel_estimator(torch.cat([z, obs_hist_short.view(*obs_hist_short.shape[:-2], -1)], dim=-1))
        return z, vel

    def encode_critic(self, obs_tuple, **kwargs):
        full_obs, _, vel = obs_tuple
        z = self.encoder_critic(full_obs.view(*full_obs.shape[:-2], -1))
        return z, vel

    def act(self, observations, **kwargs):
        mean = self.act_inference(observations, **kwargs)
        self.update_distribution(mean)
        if self.squash_mode == "tanh":
            return torch.tanh(self.distribution.sample()) * self.clip_action
        elif self.squash_mode == "clip":
            return torch.clamp(self.distribution.sample(), -self.clip_action, self.clip_action)

    def act_inference(self, observations, **kwargs):
        obs_tuple = self.extract(observations)
        prop = obs_tuple[1]
        z, vel = self.encode(obs_tuple)
        actor_obs = torch.cat([z.detach(), prop, vel.detach()], dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_tuple = self.extract_critic(critic_observations)
        prop = obs_tuple[1]
        z, vel = self.encode_critic(obs_tuple)
        critic_obs = torch.cat([z, prop.squeeze(), vel.squeeze()], dim=-1)
        value = self.critic(critic_obs)
        return value


class ActorCriticTarRnn(ActorCriticTar, ActorCriticRnn):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist_short: int = 4,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        # RNN
        rnn_hidden_dims: list[int] = [256, 256],
        rnn_type: str = "lstm",
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist=num_hist_short,
            num_hist_short=num_hist_short,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            vel_encoder_dims=vel_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
        )

        self.encoder = Memory(
            input_size=self.num_obs_h1,
            type=rnn_type,
            arch="direct",
            num_layers=len(rnn_hidden_dims),
            hidden_size=rnn_hidden_dims[0],
            out_features=self.num_latents,
        )

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        print(f"RNN Encoder: {self.encoder}")
        print(f"TransModel: {self.trans}")
        print(f"Velocity Estimator: {self.vel_estimator}")

    def reset(self, dones=None):
        if dones is None:
            self.set_hidden_states(None, None)
        else:
            hidden_states = self.get_hidden_states()
            self.set_hidden_states(
                self.encoder.reset(dones, hidden_states[0]), self.encoder.reset(dones, hidden_states[0])
            )

    def encode(self, obs_tuple, **kwargs):
        _, prop, obs_hist_short = obs_tuple
        hidden_states = (
            kwargs.get("hidden_states", None) or kwargs.get("hid_states", [None])[0]
        )  # from direct kwarg or batch dict
        masks = kwargs.get("masks", None)
        if hidden_states is not None and masks is not None:
            if hidden_states[0].shape[1] != prop.shape[1]:  # Check if hidden states are padded
                prop = unpad_trajectories(prop, masks)
                obs_hist_short = unpad_trajectories(obs_hist_short, masks)
        z, hidden_states = self.encoder(prop, masks, hidden_states)
        z = z.squeeze(0)
        if obs_hist_short.dim() == 4:
            unpadded_slices = []
            for i in range(obs_hist_short.shape[2]):  # Loop over the 3rd dimension (4)
                unpadded_slice = unpad_trajectories(obs_hist_short[:, :, i, :], masks=masks)  # Ex shape: (24, 1248, 45)
                unpadded_slices.append(unpadded_slice)
            # Stack the results back to shape of dim == 4
            obs_hist_short = torch.stack(unpadded_slices, dim=2)
        vel = self.vel_estimator(torch.cat([z, obs_hist_short.reshape(*obs_hist_short.shape[:-2], -1)], dim=-1))
        return z, hidden_states, vel

    def encode_critic(self, obs_tuple, **kwargs):
        full_obs, _, vel = obs_tuple
        full_obs = full_obs.view(*full_obs.shape[:-2], -1)
        masks = kwargs.get("masks", None)
        if masks is not None:
            full_obs = unpad_trajectories(full_obs, masks)
            vel = unpad_trajectories(vel, masks)
        z = self.encoder_critic(full_obs)
        return z, vel

    def act(self, observations, **kwargs):
        obs_tuple = self.extract(observations)
        prop = obs_tuple[1]
        masks = kwargs.get("masks", None)
        if masks is not None:
            prop = unpad_trajectories(prop, masks)
        z, hidden_states, vel = self.encode(obs_tuple, **kwargs)
        actor_obs = torch.cat([z.detach(), prop, vel.detach()], dim=-1)  # TODO: check normalization
        mean = self.actor(actor_obs)
        self.update_distribution(mean)
        if self.squash_mode == "tanh":
            return torch.tanh(self.distribution.sample()) * self.clip_action, hidden_states
        elif self.squash_mode == "clip":
            return torch.clamp(self.distribution.sample(), -self.clip_action, self.clip_action), hidden_states

    def act_inference(self, observations, **kwargs):
        obs_tuple = self.extract(observations)
        prop = obs_tuple[1]
        z, _, vel = self.encode(obs_tuple, **kwargs)
        masks = kwargs.get("masks", None)
        if masks is not None:
            prop = unpad_trajectories(prop, masks)
        actor_obs = torch.cat([z.detach(), prop, vel.detach()], dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        hidden_states = kwargs.get("hidden_states", (None, None))
        masks = kwargs.get("masks", None)
        obs_tuple = self.extract_critic(critic_observations)
        prop = obs_tuple[1]
        z, vel = self.encode_critic(obs_tuple, **kwargs)
        if masks is not None:
            prop = unpad_trajectories(prop, masks)
            vel = unpad_trajectories(vel, masks) if vel.shape[:-1] != prop.shape[:-1] else vel
        critic_obs = torch.cat([z, prop.squeeze(), vel.squeeze()], dim=-1)
        value = self.critic(critic_obs)
        return value, hidden_states  # for hidden states


class ActorCriticTarTcn(ActorCriticTar):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist: int,
        num_hist_short: int = 4,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        # TCN
        tcn_hidden_channels: list[int] = [32, 32, 32],
        tcn_kernel_sizes: list[int] = [8, 5, 5],
        tcn_strides: list[int] = [4, 1, 1],
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist=num_hist,
            num_hist_short=num_hist_short,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            vel_encoder_dims=vel_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
            **kwargs,
        )

        self.encoder = tcn_factory(
            input_channels=self.num_obs_h1,
            output_dims=latent_dims,
            num_hist=self.num_hist,
            hidden_channels=tcn_hidden_channels,
            kernel_sizes=tcn_kernel_sizes,
            strides=tcn_strides,
            activation=resolve_nn_activation(activation),  # type: ignore
            last_act=True,
        )

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        print(f"TCN Encoder: {self.encoder}")
        print(f"MLP Encoder Critic: {self.encoder_critic}")
        print(f"TransModel: {self.trans}")
        print(f"[INFO]: Velocity Estimator: {self.vel_estimator}")

    def encode(self, obs_tuple, **kwargs):
        obs_hist, _, obs_hist_short = obs_tuple
        # hist is shape: [B, num_hist, self.num_obs_h1]
        # For Conv1D: [B, channels_in, seq_len] = [B, self.num_obs_h1, num_hist]
        obs_hist = obs_hist.permute(0, 2, 1)  # => [B, self.num_obs_h1, num_hist]
        z = self.encoder(obs_hist)
        vel = self.vel_estimator(torch.cat([z, obs_hist_short.view(*obs_hist_short.shape[:-2], -1)], dim=-1))
        return z, vel

# -----------------------------------------------------------------------------
# ------------------------------ Fine Tuning ---------------------------------
# -----------------------------------------------------------------------------

# MLP without privleged information


class ActorCriticTarFt(ActorCriticTar):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist: int,
        num_hist_short: int = 4,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        mlp_encoder_dims: list[int] = [256, 128, 64],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist=num_hist,
            num_hist_short=num_hist_short,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            mlp_encoder_dims=mlp_encoder_dims,
            vel_encoder_dims=vel_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
            **kwargs,
        )

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        print(f"MLP Encoder: {self.encoder}")
        print(f"Excluding Critic Encoder: {self.encoder_critic}")
        print(f"Velocity Estimator: {self.vel_estimator}")
        print(f"TransModel: {self.trans}")

    def extract_critic(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        prop = observations[..., -1, 3:48]  # [Batch, Time, Dim]
        vel = observations[..., -1, 0:3]
        full_obs = observations
        return full_obs, prop, vel

    def encode_critic(self, obs_tuple, **kwargs):
        obs_hist = obs_tuple[0]
        prop_hist = obs_hist[..., 3:]
        z = self.encoder(prop_hist.reshape(*prop_hist.shape[:-2], -1))
        vel = obs_hist[..., -1, 0:3]
        return z, vel


# MLP without privleged information and velocity estimation
class ActorCriticTarFtNoVel(ActorCriticTarFt):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist: int,
        num_hist_short: int = 4,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        mlp_encoder_dims: list[int] = [256, 128, 64],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist=num_hist,
            num_hist_short=num_hist_short,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            mlp_encoder_dims=mlp_encoder_dims,
            vel_encoder_dims=vel_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
            **kwargs,
        )

    def encode(self, obs_tuple, **kwargs):
        z, vel = super().encode(obs_tuple, **kwargs)
        return z, vel * 0.0

    def extract_critic(self, observations):
        full_obs, prop, vel = super().extract_critic(observations)
        return full_obs, prop, vel * 0.0

# RNN without privleged information


class ActorCriticTarRnnFt(ActorCriticTarRnn):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist_short: int = 4,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        # RNN
        rnn_hidden_dims: list[int] = [256, 256],
        rnn_type: str = "lstm",
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist_short=num_hist_short,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            vel_encoder_dims=vel_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
            rnn_hidden_dims=rnn_hidden_dims,
            rnn_type=rnn_type,
            **kwargs,
        )

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        print(f"RNN Encoder: {self.encoder}")
        print(f"Excluding Critic Encoder: {self.encoder_critic}")
        print(f"TransModel: {self.trans}")
        print(f"Velocity Estimator: {self.vel_estimator}")

    def extract_critic(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        prop = observations[..., -1, 3:48]  # [Batch, Time, Dim]
        vel = observations[..., -1, 0:3]
        full_obs = observations
        return full_obs, prop, vel

    def encode_critic(self, obs_tuple, **kwargs):
        obs_hist, _, vel = obs_tuple
        obs_tuple = (obs_tuple[0], obs_tuple[1], obs_hist[..., -self.num_hist_short :, 3:48])
        z, _, _ = self.encode(obs_tuple, **kwargs)
        return z, vel


# RNN without privleged information and velocity estimation
class ActorCriticTarRnnFtNoVel(ActorCriticTarRnnFt):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist_short: int = 4,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        vel_encoder_dims: list[int] = [64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_action: float = 100.0,
        squash_mode: str = "clip",  # 'tanh' or 'clip'
        trans_hidden_dims: list[int] = [32],
        # RNN
        rnn_hidden_dims: list[int] = [256, 256],
        rnn_type: str = "lstm",
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_hist_short=num_hist_short,
            latent_dims=latent_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            vel_encoder_dims=vel_encoder_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            trans_hidden_dims=trans_hidden_dims,
            rnn_hidden_dims=rnn_hidden_dims,
            rnn_type=rnn_type,
            **kwargs,
        )

    def encode(self, obs_tuple, **kwargs):
        z, hidden_states, vel = super().encode(obs_tuple, **kwargs)
        return z, hidden_states, vel * 0.0

    def extract_critic(self, observations):
        full_obs, prop, vel = super().extract_critic(observations)
        return full_obs, prop, vel * 0.0

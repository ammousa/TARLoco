# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the following repositories:
# https://github.com/leggedrobotics/rsl_rl
# SLR: https://github.com/11chens/SLR-master
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation

from exts.tarloco.learning.modules.utils import mlp_factory

from .base.ac_base import AcNet, ActorCriticMlp


class ActorCriticMlpSlr(ActorCriticMlp):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist: int,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        mlp_encoder_dims: list[int] = [256, 128, 64],
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
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            clip_action=clip_action,
            squash_mode=squash_mode,
            **kwargs,
        )
        self.num_hist = int(num_hist)
        self.num_obs_h1 = int(num_actor_obs / self.num_hist)
        self.num_latents = latent_dims

        self.encoder = mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=num_actor_obs,
            out_dims=latent_dims,
            hidden_dims=mlp_encoder_dims,
        )

        self.trans = mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=latent_dims + num_actions,
            out_dims=latent_dims,
            hidden_dims=trans_hidden_dims,
        )

        self.actor = AcNet(
            is_policy=True,
            num_out=num_actions,
            num_obs=self.num_obs_h1 + self.num_latents,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        self.critic = AcNet(
            is_policy=False,
            num_out=1,  # Critic output is a single value
            num_obs=self.num_obs_h1 + self.num_latents,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        print(f"MLP Encoder: {self.encoder}")
        print(f"TransModel: {self.trans}")

    def extract(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        prop = observations[:, -1, :]
        hist = observations
        return hist, prop

    def encode(self, obs_tuple, **kwargs):
        obs_hist, _ = obs_tuple
        z = self.encoder(obs_hist.reshape(obs_hist.size(0), -1))
        return z

    def encode_critic(self, obs_tuple, **kwargs):
        return self.encode(obs_tuple, **kwargs)

    def act(self, observations, **kwargs):
        mean = self.act_inference(observations, **kwargs)
        self.update_distribution(mean)
        if self.squash_mode == "tanh":
            return torch.tanh(self.distribution.sample()) * self.clip_action
        elif self.squash_mode == "clip":
            return torch.clamp(self.distribution.sample(), -self.clip_action, self.clip_action)

    def act_inference(self, observations, **kwargs):
        obs_tuple = self.extract(observations)
        z = self.encode(obs_tuple)
        actor_obs = torch.cat([z.detach(), obs_tuple[1]], dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_tuple = self.extract(critic_observations)
        z = self.encode_critic(obs_tuple)
        critic_obs = torch.cat([z, obs_tuple[1]], dim=-1)
        value = self.critic(critic_obs)
        return value


class ActorCriticMlpSlrDblEnc(ActorCriticMlpSlr):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_hist: int,
        latent_dims: int = 20,
        actor_hidden_dims: list[int] = [256, 128, 128],
        critic_hidden_dims: list[int] = [512, 256, 256],
        mlp_encoder_dims: list[int] = [256, 128, 64],
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

        self.encoder_critic = mlp_factory(
            activation=resolve_nn_activation(activation),
            input_dims=num_critic_obs,
            out_dims=latent_dims,
            hidden_dims=mlp_encoder_dims,
        )

    def post_init(self):
        super().post_init()
        print(f"MLP Encoder Critic: {self.encoder_critic}")

    def encode_critic(self, obs_tuple, **kwargs):
        obs_hist, _ = obs_tuple
        z = self.encoder_critic(obs_hist.reshape(obs_hist.size(0), -1))
        return z

    def act(self, observations, **kwargs):
        mean = self.act_inference(observations, **kwargs)
        self.update_distribution(mean)
        if self.squash_mode == "tanh":
            return torch.tanh(self.distribution.sample()) * self.clip_action
        elif self.squash_mode == "clip":
            return torch.clamp(self.distribution.sample(), -self.clip_action, self.clip_action)

    def act_inference(self, observations, **kwargs):
        obs_tuple = self.extract(observations)
        z = self.encode(obs_tuple)
        actor_obs = torch.cat([z, obs_tuple[1]], dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_tuple = self.extract(critic_observations)
        z = self.encode_critic(obs_tuple)
        critic_obs = torch.cat([z, obs_tuple[1]], dim=-1)
        value = self.critic(critic_obs)
        return value

# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the following repositories:
# https://github.com/leggedrobotics/rsl_rl
# HIMLoco: https://github.com/OpenRobotLab/HIMLoco
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation
from torch.optim import Adam

from exts.tarloco.learning.modules.base.ac_base import AcNet, ActorCriticMlp


class ActorCriticHIM(ActorCriticMlp):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",  # 'tanh' or 'clip'
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticHIM.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
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
        self.num_hist = 6
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_obs_h1 = int(num_actor_obs / self.num_hist)

        mlp_input_dim_a = self.num_obs_h1 + 3 + 16
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = HIMEstimator(temporal_steps=self.num_hist, num_one_step_obs=self.num_obs_h1)

        self.actor = AcNet(
            is_policy=True,
            num_out=num_actions,
            num_obs=mlp_input_dim_a,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )
        self.critic = AcNet(
            is_policy=False,
            num_out=1,  # Critic output is a single value
            num_obs=mlp_input_dim_c,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

    def post_init(self):
        print(f"[INFO]: Using Policy: {self.__class__.__name__}")
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        print(f"Estimator: {self.estimator.encoder}")

    def extract(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        prop = observations[:, -1, :]
        hist = observations
        return hist, prop

    def encode(self, obs_tuple, **kwargs):
        obs_hist, _ = obs_tuple
        with torch.no_grad():
            vel, latent = self.estimator(obs_hist.reshape(obs_hist.shape[0], -1))
        return latent, vel

    def encode_critic(self, obs_tuple, **kwargs):
        return self.encode(obs_tuple, **kwargs)

    def act(self, observations, **kwargs):
        mean = self.act_inference(observations, **kwargs)
        self.update_distribution(mean)
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        obs_tuple = self.extract(observations)
        obs_history, _ = obs_tuple
        latent, vel = self.encode(obs_tuple)
        actor_input = torch.cat((obs_history[:, -1, :], vel, latent), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations[:, -1, :])
        return value


class MyNet(nn.Module):
    """Custom network."""

    def __init__(
        self,
        is_policy: bool,
        num_out: int,
        num_obs: int,
        hidden_dims: list[int],
        activation: str,
    ):
        """Initialize the network.

        Args:
            is_policy (bool): Whether the network is a policy network.
            num_out (int): Number of outputs.
            num_obs (int): Number of all inputs.
            img_size (tuple[int, int]): Size of the input image.
            hidden_dims (list[int]): List of hidden layer sizes.
            activation (str): Activation function to use.
        """
        super().__init__()

        self.is_policy = is_policy
        self.num_out = num_out
        self.num_obs = num_obs

        activation = resolve_nn_activation(activation)  # type: ignore

        mlp_layers = []
        mlp_layers.append(nn.Linear(self.num_obs, hidden_dims[0]))
        mlp_layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                last = nn.Linear(hidden_dims[layer_index], num_out)
                if is_policy:
                    # last policy layer should be initialized with small weights
                    num_in = hidden_dims[layer_index]
                    torch.nn.init.uniform_(last.weight, -0.01 / num_in, 0.01 / num_in)
                mlp_layers.append(last)
            else:
                mlp_layers.append(
                    nn.Linear(
                        hidden_dims[layer_index],
                        hidden_dims[layer_index + 1],
                    )
                )
                mlp_layers.append(activation)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.mlp(input)


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        enc_hidden_dims=[128, 64, 16],
        tar_hidden_dims=[128, 64],
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        num_prototype=32,
        temperature=3.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "Estimator_CL.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def extract(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(0)
        prop = observations[:, -1, :]
        hist = observations
        return hist, prop

    def encode(self, obs_tuple, **kwargs):
        obs_hist, _ = obs_tuple
        with torch.no_grad():
            vel, latent = self.estimator(obs_hist.reshape(obs_hist.shape[0], -1))
        return vel, latent

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        next_critic_obs = next_critic_obs.reshape(next_critic_obs.shape[0], -1)
        vel = next_critic_obs[
            :, self.num_one_step_obs : self.num_one_step_obs + 3
        ].detach()  # true velocity from privileged obs
        next_obs = next_critic_obs[:, : self.num_one_step_obs].detach()  # without height

        z_s = self.encoder(obs_history.reshape(obs_history.shape[0], -1))
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)  # normalized latent of current obs
        z_t = F.normalize(z_t, dim=-1, p=2)  # normalized latent of next obs

        with torch.no_grad():
            w = self.proto.weight.data.clone()  # cluster centers
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T  # similarity score between current obs and cluster centers
        score_t = z_t @ self.proto.weight.T  # similarity score between next obs and cluster centers

        with torch.no_grad():
            q_s = sinkhorn(score_s)  # sinkhorn score between current obs and cluster centers
            q_t = sinkhorn(score_t)  # sinkhorn score between next obs and cluster centers

        log_p_s = F.log_softmax(
            score_s / self.temperature, dim=-1
        )  # to calculate cross entropy loss between current obs and cluster centers
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = (
            -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        )  # cross entropy loss between current obs and cluster centers and next obs and cluster centers
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss, swap_loss


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T

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
import torch.nn.functional as F

from exts.tarloco.learning.modules import ActorCriticMlp as ActorCritic

from .ppo import PPO


class PPOTAR(PPO):
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        lr_max=1.0e-3,
        lr_min=1.0e-5,
        adam_betas=(0.9, 0.999),
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        aux_loss_coef=[1.0],
        optimizer="Adam",
    ):
        super().__init__(
            actor_critic=actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            lr_max=lr_max,
            lr_min=lr_min,
            adam_betas=adam_betas,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            optimizer=optimizer,
        )

        self.num_envs = 4096
        self.aux_loss_coef = aux_loss_coef

    def _compute_auxiliary_loss(self, batch: dict) -> dict:
        """Compute any auxiliary loss. Override this in subclasses if needed."""
        assert (
            self.num_envs is not None
        ), "[ERROR]: Number of environments must be provided for negative sample indexing."

        # Encode Actor Encoder
        obs_tuple = self.actor_critic.extract(batch["obs"])
        encode_a = self.actor_critic.encode(
            obs_tuple, hidden_states=(batch.get("hid_states") or [None])[0], masks=batch.get("masks", None)
        )
        z_a, vel_estimated = encode_a[0], encode_a[-1]

        # FW through Trans model
        pred_next_z = self.actor_critic.trans(torch.cat([z_a, batch["actions"]], dim=-1))

        # Encode Critic Encoder with current obs and next obs
        obs_tuple_c = self.actor_critic.extract_critic(batch["critic_obs"])
        next_obs_tuple_c = self.actor_critic.extract_critic(batch["next_critic_obs"])
        encode_c = self.actor_critic.encode_critic(
            obs_tuple_c,
            hidden_states=(batch.get("hid_states") or [None])[0],
            masks=batch.get("masks", None)
        )
        next_encode_c = self.actor_critic.encode_critic(
            next_obs_tuple_c,
            hidden_states=(batch.get("next_hid_states") or [None])[0],
            masks=batch.get("masks", None)
        )
        next_z_c, vel_c = next_encode_c[0], encode_c[-1]

        # Generate next_neg_indices ensuring no multiples of number of environments
        batch_size = next_z_c.size(0) if not self.actor_critic.is_recurrent else next_z_c.size(1)
        next_neg_indices = self.get_valid_negative_indices(batch, batch_size)
        next_neg_z = (
            next_z_c[next_neg_indices].detach()
            if not self.actor_critic.is_recurrent
            else next_z_c[:, next_neg_indices, :].detach()
        )

        # Validate that no index in next_neg_indices is a multiple of num_envs
        if torch.any((next_neg_indices % self.num_envs) == 0) and not self.actor_critic.is_recurrent:
            invalid_indices = next_neg_indices[(next_neg_indices % self.num_envs) == 0]
            raise ValueError(f"Validation failed: Invalid indices present in {invalid_indices}.")

        pos_diff = next_z_c - pred_next_z
        neg_diff = next_z_c - next_neg_z

        pos_loss = (pos_diff.pow(2)).sum(1).mean()
        neg_loss = (neg_diff.pow(2)).sum(1)

        zeros = torch.zeros_like(pos_loss)
        neg_loss = torch.max(zeros, 1.0 - neg_loss).mean()
        triplet_loss = pos_loss + neg_loss
        # MSE between vel and vel_estimated
        no_vel_training = vel_c.shape != vel_estimated.shape  # case of no vel rnn ablation study
        vel_loss = torch.tensor(0.0, device=vel_c.device) if no_vel_training else F.mse_loss(vel_c.squeeze(-2), vel_estimated)

        return {
            "tar": triplet_loss * self.aux_loss_coef[0],
            "vel_mse": vel_loss * self.aux_loss_coef[-1],
        }

    # ======================
    # Helper Functions
    # ======================

    @torch.no_grad()
    def get_valid_negative_indices(self, batch, batch_size):
        """
        Generates valid negative indices ensuring no multiples of `num_envs`.

        This function:
        - Randomly offsets indices to create `next_neg_indices`.
        - Ensures that selected indices are not multiples of `self.num_envs`.
        - Uses efficient vectorized operations to minimize while-loop iterations.

        Args:
            self (object): The training instance that holds environment parameters.
            batch (dict): The input batch containing metadata.
            batch_size (int): Number of robots in the batch.

        Returns:
            torch.Tensor: A tensor of shape [batch_size] containing valid negative indices.
        """

        device = batch["returns"].device
        num_envs = self.num_envs  # Number of environments constraint

        # Generate random negative indices
        indices = torch.arange(batch_size, device=device)
        random_offsets = torch.randint(1, batch_size, (batch_size,), device=device)
        next_neg_indices = (indices + random_offsets) % batch_size

        # === Ensure No Multiples of `num_envs` ===
        invalid_mask = (next_neg_indices % num_envs) == 0

        # Efficiently replace invalid indices using a mask
        while invalid_mask.any():
            random_offsets[invalid_mask] = torch.randint(1, batch_size, (invalid_mask.sum(),), device=device)
            next_neg_indices = (indices + random_offsets) % batch_size
            invalid_mask = (next_neg_indices % num_envs) == 0

        return next_neg_indices

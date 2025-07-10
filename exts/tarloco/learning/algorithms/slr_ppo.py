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

import numpy as np
import torch

from exts.tarloco.learning.modules import ActorCriticMlp as ActorCritic

from .ppo import PPO


class PPOSLR(PPO):
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
        optimizer="adam",
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

    def _compute_auxiliary_loss(self, batch: dict) -> dict:
        """Compute any auxiliary loss. Override this in subclasses if needed."""
        obs_tuple = self.actor_critic.extract(batch["obs"])
        z = self.actor_critic.encode(
            obs_tuple, hidden_states=(batch.get("hid_states") or [None])[0], masks=batch.get("masks", None)
        )
        z = z[0] if isinstance(z, tuple) else z
        pred_next_z = self.actor_critic.trans(torch.cat([z, batch["actions"]], dim=-1))
        next_obs_tuple_c = self.actor_critic.extract(batch["next_obs"])
        targ_next_z = self.actor_critic.encode(
            next_obs_tuple_c, hidden_states=(batch.get("next_hid_states") or [None])[0], masks=batch.get("masks", None)
        )
        targ_next_z = targ_next_z[0] if isinstance(targ_next_z, tuple) else targ_next_z
        batch_size = z.size(0)
        perm = np.random.permutation(batch_size)
        next_neg_z = targ_next_z[perm].detach()

        pos_diff = targ_next_z - pred_next_z
        neg_diff = targ_next_z - next_neg_z

        pos_loss = (pos_diff.pow(2)).sum(1).mean()
        neg_loss = (neg_diff.pow(2)).sum(1)

        zeros = torch.zeros_like(pos_loss)
        neg_loss = torch.max(zeros, 1.0 - neg_loss).mean()
        triplet_loss = pos_loss + neg_loss

        return {"triplet": triplet_loss * 1e-3}

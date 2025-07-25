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
from __future__ import annotations

from exts.tarloco.learning.modules import ActorCriticHIM

from .ppo import PPO


class PPOHIM(PPO):
    actor_critic: ActorCriticHIM

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
        estimation_loss, swap_loss = self.actor_critic.estimator.update(
            batch["obs"], batch["next_obs"], batch["next_critic_obs"], lr=self.lr
        )
        return {
            "estimation": estimation_loss.detach(),  # detached cuz backprop is already done in estimator.update
            "swap": swap_loss.detach(),
        }

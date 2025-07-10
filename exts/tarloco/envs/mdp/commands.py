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

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MyVelocityCommand(UniformVelocityCommand):
    cfg: VelocityCommandWithRotateCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: VelocityCommandWithRotateCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.is_rotate_only_env = torch.zeros_like(self.is_heading_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tRotating only probability: {self.cfg.rel_rotate_only_envs}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        # update rotate only envs
        self.is_rotate_only_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_rotate_only_envs

    def _update_command(self):
        super()._update_command()
        # Enforce rotation only for environments where the robots rotate only
        rotate_only_env_ids = self.is_rotate_only_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[rotate_only_env_ids, :2] = 0.0


@configclass
class VelocityCommandWithRotateCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = MyVelocityCommand

    rel_rotate_only_envs: float = MISSING  # type: ignore
    """Probability threshold for environments where the robots rotate only."""

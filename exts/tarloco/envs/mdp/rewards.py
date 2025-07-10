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

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that lasts than a threshold.
    This helps ensure that the robot lifts its feet off the ground and takes steps.
    The reward is computed as the sum of the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # clamped = torch.clamp(last_air_time - threshold, min=0.0)
    clamped = last_air_time - threshold
    reward = torch.sum(clamped * first_contact, dim=1)
    # no reward for zero linear and angular commands
    big_lin_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    big_ang_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2]) > 0.05
    # logical or
    reward *= big_lin_cmd | big_ang_cmd
    return reward


def feet_close_together(env: ManagerBasedRLEnv, feet: list[str], threshold: float) -> torch.Tensor:
    """Penalty for feet being close together.

    If the feet are closer than a threshold, the agent is penalized.
    """
    # extract the used quantities (to enable type-hinting)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: Articulation = env.scene[asset_cfg.name]

    # get feet positions
    # body names: ['trunk', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot', 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot']
    feet_idxs = [asset.data.body_names.index(foot) for foot in feet]
    feet_pos = asset.data.body_pos_w[:, feet_idxs, :]

    # calculate the distances between all feet
    num_envs = feet_pos.shape[0]
    num_feet = len(feet)
    distances = torch.zeros(num_envs, num_feet, num_feet, device=feet_pos.device)
    for i in range(num_feet):
        for j in range(num_feet):
            distances[:, i, j] = torch.norm(feet_pos[:, i, :] - feet_pos[:, j, :], dim=1)
        distances[:, i, i] = float("inf")

    # compute the penalties (normalize to [0, 1] range)
    penalties = torch.clamp(threshold - distances, min=0.0) / threshold

    # compute the reward
    mean_penalties = torch.mean(penalties, dim=(1, 2))
    # subtract 1 to make the reward zero when the feet are not close together
    reward = torch.exp(mean_penalties) - 1.0
    return reward


def feet_standing(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalty for keeping the feet on the ground when command is small."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    clamped = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(clamped * first_contact, dim=1)

    # no reward for zero linear and angular commands
    small_lin_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.05
    small_ang_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2]) < 0.05

    # logical or
    reward *= small_lin_cmd | small_ang_cmd
    return reward

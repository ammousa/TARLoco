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

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# The main difference between the two functions lies in the additional condition present in the terrain_levels_vel_new function.
# In this function, there is an additional check for the robot's yaw velocity (angular velocity around the z-axis).
# The condition torch.abs(yaw_vel - expected_yaw_vel) < 0.2 checks if the absolute difference between the robot's yaw velocity
# and the expected yaw velocity is less than 0.2. If this condition is true, the robot is considered to have the expected yaw
# velocity and is allowed to progress to more complex terrains.
def terrain_levels_vel_new(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    expected_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    yaw_vel = asset.data.root_ang_vel_b[env_ids, 2]
    expected_yaw_vel = command[env_ids, 2]
    # robots that walked far enough or have the expected yaw velocity go to more complex terrains
    move_up = (distance > terrain.cfg.terrain_generator.size[0] / 2) | (torch.abs(yaw_vel - expected_yaw_vel) < 0.2)
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < expected_distance * 0.5
    move_down *= ~move_up  # do not move down robots that moved up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_friction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    values: list,
    interval: int,  # Number of steps between changes
    dt: int,
    cold_start_steps: int = 0,  # Number of steps to wait before changing friction
):
    # Calculate the number of steps and determine the current phase
    num_steps = round(env.common_step_counter / dt)
    phase = (num_steps // interval) % len(values)  # Determines which value to use

    # Set the friction value based on the current phase
    value = values[int(phase)]
    tol = 3  # Tolerance for the step counter

    # Check if we are within the window to change friction
    if (num_steps % interval) < tol and num_steps > cold_start_steps:
        # Obtain term settings
        term_name = "physics_material"
        term_cfg = env.event_manager.get_term_cfg(term_name)

        # Update term settings if the value has changed
        old_value, _ = term_cfg.params["static_friction_range"]
        if old_value != value:
            print(
                f"[INFO] --------------- Friction changed from {old_value} to {value} at step"
                f" {num_steps} --------------- "
            )
            term_cfg.params["static_friction_range"] = (value + 0.05, value + 0.06)
            term_cfg.params["dynamic_friction_range"] = (value + 0.00, value + 0.01)
            env.event_manager.set_term_cfg(term_name, term_cfg)


def curr_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_level: int = 2,
    friction_range: tuple = (0.1, 2.5),  # Range for friction values
    restitution_range: tuple = (0.0, 1.0),  # Range for restitution values
    mass_range: tuple = (-1.0, 10.0),  # Range for mass values
    external_push_velocity_range: tuple = (-0.5, 0.5),  # Range for push velocity
):
    """
    Updates the curriculum level based on the distance walked and applies modifications to
    friction, restitution, mass, and external forces sampled from ranges derived from terrain levels.

    Args:
        env: The environment instance.
        env_ids: The IDs of environments to update.
        asset_cfg: The asset configuration.
        friction_range: The range for friction values (min, max).
        restitution_range: The range for restitution values (min, max).
        mass_range: The range for mass values (min, max).
        velocity_range: The range for push velocity values (min, max).

    Returns:
        The mean curriculum level for the given environment IDs.
    """
    # Extract asset and terrain
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # Compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # Determine terrain progression
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # Update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    # Calculate the current curriculum level
    terrain_levels = terrain.terrain_levels.float()
    curr_level = torch.mean(terrain_levels)
    # max_level = max(terrain_levels.max(), max_level)

    def scale_range(range_tuple, level, max_level, device=None):
        """Scale the range based on the current level and max level."""
        range = range_tuple[1] - range_tuple[0]
        if level > max_level:
            sub = 0
        else:
            sub = range * (max_level - level) / (2 * max_level)

        return torch.tensor([range_tuple[0] + sub, range_tuple[1] - sub], device=(device or level.device))

    # friction, restitution and mass have to be on the CPU
    friction_scaled = scale_range(friction_range, curr_level, max_level, device="cpu")
    restitution_scaled = scale_range(restitution_range, curr_level, max_level, device="cpu")
    mass_scaled = scale_range(mass_range, curr_level, max_level, device="cpu")
    push_velocity_scaled = tuple(scale_range(external_push_velocity_range, curr_level, max_level).tolist())

    # Create a single list of updates for all term configurations
    term_updates = [
        (
            "physics_material",
            {
                "static_friction_range": friction_scaled,
                "dynamic_friction_range": (friction_scaled[0] - 0.1, friction_scaled[1] - 0.1),
                "restitution_range": restitution_scaled,
            },
        ),
        (
            "add_base_mass",
            {
                "mass_distribution_params": mass_scaled,
            },
        ),
        (
            "push_robot",
            {
                "velocity_range": {"x": push_velocity_scaled, "y": push_velocity_scaled},
            },
        ),
    ]

    # Apply all term configurations in one loop
    for term_name, params in term_updates:
        term_cfg = env.event_manager.get_term_cfg(term_name)
        assert term_cfg.mode != "startup", f"[ERROR] Term {term_name} is in startup mode. Change the mode to 'reset'"
        term_cfg.params.update(params)  # Update all parameters at once
        env.event_manager.set_term_cfg(term_name, term_cfg)

    return curr_level


def command_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str, max_curriculum: float = 1.0
) -> None:
    """Curriculum based on the tracking reward of the robot when commanded to move at a desired velocity.

    This term is used to increase the range of commands when the robot's tracking reward is above 80% of the
    maximum.

    Returns:
        The cumulative increase in velocity command range.
    """
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    delta_range = torch.tensor([-0.1, 0.1], device=env.device)
    if not hasattr(env, "delta_lin_vel"):
        env.delta_lin_vel = torch.tensor(0.0, device=env.device)
    # If the tracking reward is above 80% of the maximum, increase the range of commands
    if torch.mean(episode_sums[env_ids]) / env.max_episode_length > 0.8 * reward_term_cfg.weight:
        lin_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        lin_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        base_velocity_ranges.lin_vel_x = torch.clamp(lin_vel_x + delta_range, -max_curriculum, max_curriculum).tolist()
        base_velocity_ranges.lin_vel_y = torch.clamp(lin_vel_y + delta_range, -max_curriculum, max_curriculum).tolist()
        env.delta_lin_vel = torch.clamp(env.delta_lin_vel + delta_range[1], 0.0, max_curriculum)
    return env.delta_lin_vel


def base_velocity_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_forward_curriculum: float,
    max_backward_curriculum: float,
    max_lat_curriculum: float,
    increment: float = 0.1,
):
    """
    Updates the base velocity range in the command manager based on the curriculum logic.
    Expands or reduces the range based on performance thresholds.

    Args:
        env: The environment instance.
        env_ids: The IDs of environments to update.
        max_forward_curriculum: Maximum forward velocity limit.
        max_backward_curriculum: Maximum backward velocity limit.
        max_lat_curriculum: Maximum lateral velocity limit.
        increment: Amount to expand or reduce the range.

    Returns:
        Updated linear velocity ranges.
    """
    # Extract relevant data for tracking performance
    tracking_reward = env.reward_manager._episode_sums["track_lin_vel_xy_exp"][env_ids]
    reward_scale = env.reward_manager.cfg.track_lin_vel_xy_exp.weight

    # Calculate the mean tracking reward over the episode
    mean_tracking_reward = torch.mean(tracking_reward) / env.max_episode_length_s
    cfg = env.command_manager.cfg

    # Adjust ranges based on performance
    if mean_tracking_reward > 0.8 * reward_scale:
        # Expand the ranges
        cfg.base_velocity.ranges.lin_vel_x = (
            np.clip(cfg.base_velocity.ranges.lin_vel_x[0] - increment, -max_backward_curriculum, -0.1),
            np.clip(cfg.base_velocity.ranges.lin_vel_x[1] + increment, 0.1, max_forward_curriculum),
        )
        cfg.base_velocity.ranges.lin_vel_y = (
            np.clip(cfg.base_velocity.ranges.lin_vel_y[0] - increment, -max_lat_curriculum, -0.1),
            np.clip(cfg.base_velocity.ranges.lin_vel_y[1] + increment, 0.1, max_lat_curriculum),
        )

    elif mean_tracking_reward < 0.2 * reward_scale:
        # Reduce the ranges
        cfg.base_velocity.ranges.lin_vel_x = (
            np.clip(cfg.base_velocity.ranges.lin_vel_x[0] + increment, -max_backward_curriculum, -0.1),
            np.clip(cfg.base_velocity.ranges.lin_vel_x[1] - increment, 0.1, max_forward_curriculum),
        )
        cfg.base_velocity.ranges.lin_vel_y = (
            np.clip(cfg.base_velocity.ranges.lin_vel_y[0] + increment, -max_lat_curriculum, -0.1),
            np.clip(cfg.base_velocity.ranges.lin_vel_y[1] - increment, 0.1, max_lat_curriculum),
        )

    # Return the updated maximum forward velocity
    return cfg.base_velocity.ranges.lin_vel_x[-1]

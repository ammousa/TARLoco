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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.observations import *
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
This module provides utility functions for interacting with and extracting information
from the environment's sensors and assets in a reinforcement learning context.

Functions:
----------
- feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    Check if the feet are in contact with the ground by comparing the net contact forces
    against a specified threshold.

- feet_contact_z(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot")) -> torch.Tensor:
    Return the z-component of the net contact forces acting on the feet.

- contact_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    Retrieve the static friction value of the robot's physics material.

- base_mass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    Get the mass of the robot's base.

- base_external_force(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    Retrieve the external forces acting on the robot's base.

- joints_pd(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    Reset and return the proportional-derivative (PD) gains of the robot's joints.
"""


def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Check if the feet are in contact with the ground."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    ).float()

    return is_contact


def feet_contact_z(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Return the z-component of the net contact forces on the feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, [4, 8, 12, 16], 2]
    return net_contact_forces


def contact_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Obtain term settings
    # term_name = "physics_material"
    # term_cfg = env.event_manager.get_term_cfg(term_name)
    # friction_value, _ = term_cfg.params["static_friction_range"]
    # friction_value = env.scene.terrain.cfg.physics_material.static_friction
    asset: Articulation = env.scene[asset_cfg.name]
    friction_value = asset.root_physx_view.get_material_properties().to(env.device)[:, -1, 0].unsqueeze(1)
    return friction_value


def base_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_mass = asset.root_physx_view.get_masses()[:, asset_cfg.body_ids].to(env.device)[:, 0]
    return base_mass.unsqueeze(1)


def base_external_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset._external_force_b[:, 0, :]


def joints_pd(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the PD gains of the joints of the robot.

    Probably best to use this function only on startup."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get actuators
    actuators = asset.actuators
    actuator = actuators["base_legs"]

    # assume all joints in a robot have the same gains
    p_gains = actuator.stiffness[:, 0].clone()
    d_gains = actuator.damping[:, 0].clone()

    gains = torch.stack((p_gains, d_gains), dim=-1)
    return gains

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

from isaaclab.utils import configclass

from .base import ObservationsCfg  # Ensure ObservationsCfg is imported
from .base import (
    BaseLocomotionVelocityEnvCfg,
    EvaluationConfigMixin,
    FullObservationsCfg,
)


# -------------------- Final Configurations --------------------
@configclass
class TarGo1LocomotionVelocityRoughEnvCfg(BaseLocomotionVelocityEnvCfg):
    """Configuration for Rough terrain."""

    def __post_init__(self):
        """Override the observations attribute after initialization."""
        super().__post_init__()

        # Update observation space cfg
        self.observations = FullObservationsCfg()

        # Policy
        self.observations.policy.history_length = 5
        self.observations.policy.flatten_history_dim = False
        del self.observations.policy.base_lin_vel
        del self.observations.policy.height_scan
        del self.observations.policy.base_external_force
        del self.observations.policy.feet_contact_z
        del self.observations.policy.contact_friction
        del self.observations.policy.base_mass

        # Critic
        self.observations.critic.history_length = 1
        self.observations.critic.flatten_history_dim = False


@configclass
class SlrGo1LocomotionVelocityRoughEnvCfg(BaseLocomotionVelocityEnvCfg):
    """Configuration for Rough terrain."""

    def __post_init__(self):
        """Override the observations attribute after initialization."""
        super().__post_init__()

        # Policy
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = False
        del self.observations.policy.base_lin_vel
        del self.observations.policy.height_scan

        # Critic
        self.observations.critic.history_length = 10
        self.observations.critic.flatten_history_dim = False
        del self.observations.critic.base_lin_vel
        del self.observations.critic.height_scan

        # Disable height scanner
        del self.scene.height_scanner


@configclass
class HimGo1LocomotionVelocityRoughEnvCfg(BaseLocomotionVelocityEnvCfg):
    """Configuration for Rough terrain."""

    def __post_init__(self):
        """Override the observations attribute after initialization."""
        super().__post_init__()

        # Update observation space cfg
        self.observations = FullObservationsCfg()

        # Policy
        self.observations.policy.history_length = 6
        self.observations.policy.flatten_history_dim = False
        del self.observations.policy.base_lin_vel
        del self.observations.policy.height_scan
        del self.observations.policy.base_external_force
        del self.observations.policy.feet_contact_z
        del self.observations.policy.contact_friction
        del self.observations.policy.base_mass

        # Critic
        self.observations.critic.history_length = 1
        self.observations.critic.flatten_history_dim = False
        del self.observations.critic.feet_contact_z
        del self.observations.critic.contact_friction
        del self.observations.critic.base_mass

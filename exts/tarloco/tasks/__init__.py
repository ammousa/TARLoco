#  Copyright 2025 University of Manchester, Amr Mousa
#  SPDX-License-Identifier: CC-BY-SA-4.0


from dataclasses import dataclass, field
from typing import List, Type

import gymnasium as gym

from exts.tarloco.envs import wrappers
from exts.tarloco.learning import runners

from . import agents, algorithms, envs

# Define a dataclass to hold the configuration for each task


@dataclass
class TaskConfig:
    env_cfg_entry_point: Type
    rsl_rl_cfg_entry_point: Type
    agent_cfg: Type = algorithms.RslRlOnPolicyRunnerCfg
    runner: Type = runners.OnPolicyRunner
    env_wrappers: List[Type] = field(default_factory=lambda: [wrappers.RslRlVecEnvWrapper])


# Define the registry

registry = {
    # ------
    # TAR
    # ------
    "go1-train-tar-rnn-rough": TaskConfig(
        env_cfg_entry_point=envs.TarGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1RoughRnnTarRunnerCfg,
    ),
    # ------
    # SLR
    # ------
    "go1-train-slr-rough": TaskConfig(
        env_cfg_entry_point=envs.SlrGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1RoughPpoSlrRunnerCfg,
    ),
    # --------
    # HIM
    # --------
    "go1-train-him-rough": TaskConfig(
        env_cfg_entry_point=envs.HimGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1PpoHimRunnerCfg,
    ),
}


# Register each environment
for env_id, config in registry.items():
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": config.env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": config.rsl_rl_cfg_entry_point,
        },
    )

__all__ = ["registry"]

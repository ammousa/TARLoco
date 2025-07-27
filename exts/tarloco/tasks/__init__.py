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
    # ------
    # Teacher
    # ------
    # Plain teacher configuration: Direct feeding to the actor and critic without using an encoder
    "go1-train-teacher-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1RoughPpoRunnerCfg,
    ),
    # Teacher with MLP privileged encoder that concatenates the latents to one-step proprioceptive observations
    "go1-train-teacher-encoder-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1RoughPpoExpertRunnerCfg,
    ),
    # Teacher with RNN privileged encoder that concatenates the latents to one-step proprioceptive observations
    "go1-train-teacher-rnn-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1RoughRnnRunnerCfg,
    ),

    # ------------------------------ Ablation Studies ------------------------------
    # TAR replacing RNN encoder with 10-steps MLP
    "go1-train-tar-mlp-rough": TaskConfig(
        env_cfg_entry_point=envs.TarMlpGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1RoughPpoTarRunnerCfg,
    ),
    # TAR replacing RNN encoder with TCN
    "go1-train-tar-tcn-rough": TaskConfig(
        env_cfg_entry_point=envs.TarTcnGo1LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go1RoughTcnTarRunnerCfg,
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

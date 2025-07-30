#  Copyright 2025 University of Manchester, Amr Mousa
#  SPDX-License-Identifier: CC-BY-SA-4.0

import datetime

import gym
import torch

from exts.tarloco.utils import get_attr_recursively


class EvaluateWrapper(gym.Wrapper):
    """
    Evaluator class for processing environment steps and calculating metrics.

    Attributes:
        env (RslRlVecEnvWrapper): The environment wrapper.
        num_steps (int): Number of steps processed.
        lin_vel_error (float): Accumulated linear velocity error.
        ang_vel_error (float): Accumulated angular velocity error.
        failures (int): Total number of failures.
        timeouts (int): Total number of timeouts.
    """

    def __init__(self, env, robot_idx=0):
        """
        Initialize the Evaluator.

        Args:
            env (RslRlVecEnvWrapper): The environment wrapper.
        """
        super().__init__(env)
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.num_steps = 0
        self.dt = env.unwrapped.step_dt
        self.max_episode_length = env.unwrapped.max_episode_length
        self.robot_idx = robot_idx

        # Logging
        # Log dict contains signals which are timeseries data for the single episode of a single robot while
        # the metrics are scalars values for all the robots
        # Initialize log dictionary with pre-allocated tensors on the GPU
        # The convention is time steps along the rows and signals along the columns
        self.log = {
            "signals": {
                "contact_forces_z": torch.zeros((4), device="cuda"),
                "base_vel_x_y_yaw": torch.zeros((3), device="cuda"),
                "commands_x_y_yaw": torch.zeros((3), device="cuda"),
                "external_force_trunk": torch.zeros((3), device="cuda"),
                "terrain_levels": 0,
            },
            "metrics": {},
            "cma_metrics_data": {},
            "sim_time": 0,
            "robot_position": torch.zeros((3), device="cuda"),
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step_new(action)
        updated_info = self._process_step(observation, terminated, truncated, info)
        return observation, reward, terminated, truncated, updated_info

    def _process_step(self, obs, terminated, truncated, info):
        """
        Process a single step in the environment.

        Args:
            obs (torch.Tensor): Observation tensor.
            action (torch.Tensor): Action tensor.
            reward (torch.Tensor): Reward tensor.
            terminated (torch.Tensor): Termination tensor.
            truncated (torch.Tensor): Truncation tensor.
            info (dict): Additional information.
            env (object): Unwrapped environment object.

        Notes:
            # sim = env.get_attr("sim") or env.unwrapped.sim # same for scene
            # contact_forces = env.unwrapped.scene["contact_forces"].data.net_forces_w
            # contact_forces = env.unwrapped.scene["contact_forces"].data.net_forces_w
            # it has shape of (num_envs,17,3) where 17 are ['trunk',
            # 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
            # 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot',
            # 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot',
            # 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot']
            # contact_forces_z = env.unwrapped.scene["contact_forces"].data.net_forces_w[:, [4, 8, 12, 16], 2]
            # base_vel_x_y = env.unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
            # base_yaw = env.unwrapped.scene["robot"].data.root_ang_vel_b[:,2]
            # env.unwrapped.scene["robot"].data.applied_torque or (joint_pos, joint_pos_target, joint_vel, joint_vel_target, joint_acc)
            # env.unwrapped.scene["robot"].data._sim_timestamp -
            # commands_x_y_yaw = env.unwrapped.command_manager.get_command('base_velocity')
            # external_force_trunk = env.unwrapped.scene["robot"]._external_force_b[:,0,:] # shape (num_envs,3)
            # terrain_levels = env.unwrapped.scene["terrain"].terrain_levels

        """
        mean_lin_err, mean_ang_err = self._calc_mean_tracking_err(obs)

        # Update running averages for metrics per step
        mean_lin_err = self._update_running_avg("lin_vel_error", mean_lin_err.item())
        mean_ang_err = self._update_running_avg("ang_vel_error", mean_ang_err.item())
        self.log["metrics"]["lin_vel_error"] = mean_lin_err
        self.log["metrics"]["ang_vel_error"] = mean_ang_err
        # Update running averages for metrics per min
        mean_failures_per_sec = self._update_running_avg("failures", terminated.sum().item() / self.num_envs) / self.dt
        mean_time_out_per_sec = self._update_running_avg("time_out", truncated.sum().item() / self.num_envs) / self.dt
        self.log["metrics"]["failures"] = mean_failures_per_sec * 60
        self.log["metrics"]["time_out"] = mean_time_out_per_sec * 60

        # Extract additional simulation data
        sim_time = self.env.unwrapped.scene["robot"].data._sim_timestamp
        robot_position = self.env.unwrapped.scene["robot"].data.body_pos_w[self.robot_idx, 1, :]
        contact_forces_z = self.env.unwrapped.scene["contact_forces"].data.net_forces_w[
            self.robot_idx, [4, 8, 12, 16], 2
        ]
        base_vel_x_y = self.env.unwrapped.scene["robot"].data.root_lin_vel_b[self.robot_idx, :2]
        base_yaw = self.env.unwrapped.scene["robot"].data.root_ang_vel_b[self.robot_idx, 2]
        commands_x_y_yaw = self.env.unwrapped.command_manager.get_command("base_velocity")[self.robot_idx]
        external_force_trunk = self.env.unwrapped.scene["robot"]._external_force_b[self.robot_idx, 0, :]
        # safe attributing because it changes from flat to rough terrain
        terrain_levels_attr = getattr(self.env.unwrapped.scene["terrain"], "terrain_levels", None)
        # Now safely access the index in terrain_levels if it is not None
        terrain_levels = terrain_levels_attr[self.robot_idx] if terrain_levels_attr is not None else None

        # Update log signals for robot_index
        self.log["signals"].update(
            {
                "contact_forces_z": contact_forces_z,
                "base_vel_x_y_yaw": torch.cat((base_vel_x_y, base_yaw.unsqueeze(0)), dim=0),
                "commands_x_y_yaw": commands_x_y_yaw,
                "external_force_trunk": external_force_trunk,
                "terrain_levels": terrain_levels,
            }
        )
        self.log["sim_time"] = sim_time
        self.log["robot_position"] = robot_position

        self.num_steps += 1  # Track step count
        return {**info, "updated_log": self.log}

    def _update_running_avg(self, metric_name, new_value):
        """Update cumulative moving average (CMA) per second"""
        if metric_name not in self.log["cma_metrics_data"]:
            self.log["cma_metrics_data"][metric_name] = {"sum": 0.0, "count": 0}

        self.log["cma_metrics_data"][metric_name]["sum"] += new_value
        self.log["cma_metrics_data"][metric_name]["count"] += 1
        return self.log["cma_metrics_data"][metric_name]["sum"] / self.log["cma_metrics_data"][metric_name]["count"]

    def _calc_mean_tracking_err(self, obs):
        """
        Calculate tracking errors for all robots.
        Returns:
            lin_err (torch.Tensor): RMSE of linear velocity tracking in x, y.
            ang_err (torch.Tensor): RMSE of angular velocity tracking in z.
        """
        data = get_attr_recursively(self.env, "scene").articulations["robot"].data
        t_dim_exist = data.body_lin_vel_w.dim() == 3

        lin_vel = data.body_lin_vel_w[:, -1, :] if t_dim_exist else data.body_lin_vel_w
        ang_vel = data.body_ang_vel_w[:, -1, :] if t_dim_exist else data.body_ang_vel_w
        student_obs = obs.shape[-1] == 45

        # Extract commands from observations
        if t_dim_exist:
            cmd = obs[:, -1, 6:9] if student_obs else obs[:, -1, 9:12]
        else:
            cmd = obs[:, 6:9] if student_obs else obs[:, 9:12]

        lin_err = torch.sqrt((torch.norm(lin_vel[:, :2] - cmd[:, :2], p=2, dim=-1) ** 2).mean())
        ang_err = torch.sqrt(((ang_vel[:, 2] - cmd[:, 2]) ** 2).mean())

        return lin_err, ang_err

    def print_metrics(self):
        """Print evaluation metrics."""
        total_time = self.dt * self.num_steps * self.num_envs
        readable_time = str(datetime.timedelta(seconds=total_time))

        print("-------------- Simulation info --------------")
        print(f"Number of Robots: {self.num_envs}")
        print(f"Number of Episodes: {int(self.num_steps/self.max_episode_length)}")
        print(f"Simulation Time Step: {self.dt} sec")
        print(f"Evaluation Time per Robot: {int(self.dt * (self.num_steps+1))} sec")
        print(f"Total Simulation Real Time: {readable_time}")

        print("-------------- Evaluation Metrics --------------")
        print("Note: Rate is per robot")
        print(f"Linear Velocity RMSA per step: {self.log['metrics']['lin_vel_error']:.4f}")
        print(f"Angular Velocity RMSA per step: {self.log['metrics']['ang_vel_error']:.4f}")
        print(f"Timeout Rate per min: {self.log['metrics']['time_out']:.4f}")
        print(f"Failure Rate per min: {self.log['metrics']['failures']:.4f}")

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

import os
import re
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import wandb


class LoggerWrapper:
    def __init__(self, cfg):
        """
        Initialize the LoggerWrapper.

        Args:
            cfg: experiment configuration dictionary.
        """

        # Extract experiment details if needed
        self.cfg = cfg
        self._extract_experiment_details()

        # Initialize data storage
        self.signal_logs = {
            "contact_forces_z": torch.tensor([], device=self.cfg["args_cli"].device),
            "base_vel_x_y_yaw": torch.tensor([], device=self.cfg["args_cli"].device),
            "commands_x_y_yaw": torch.tensor([], device=self.cfg["args_cli"].device),
            "external_force_trunk": torch.tensor([], device=self.cfg["args_cli"].device),
            "terrain_levels": torch.tensor([], device=self.cfg["args_cli"].device),
        }
        self.metric_logs = {
            "calc_lin_vel_error": 0,
            "calc_ang_vel_error": 0,
            "lin_vel_error": 0,
            "ang_vel_error": 0,
            "failures": 0,
            "time_out": 0,
        }
        self.time = 0

        # Initialize the summary writer based on args_cli.logger
        self._initialize_writer()

        # store the code state
        self._upload_code_files()

        self.arg_plot_contact_forces = self.args_cli.plot_contact_forces

    def _initialize_writer(self):
        # Determine logger type
        self.logger_type = self.args_cli.logger.lower() if self.args_cli.logger is not None else "tensorboard"

        # Define log directory
        self.log_dir = getattr(self.args_cli, "log_dir", "./logs")

        if self.log_dir is not None:
            # Initialize the appropriate summary writer
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.summary_writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.args_cli)
            elif self.logger_type == "wandb":
                self.summary_writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, locs=locals())
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.summary_writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found")
        else:
            raise ValueError("log_dir not found")

    def _extract_experiment_details(
        self,
    ):
        self.dt = self.cfg["dt"]
        self.robot_idx = self.cfg["args_cli"].robot_idx
        self.seed = self.cfg["args_cli"].seed
        self.run_path = os.path.dirname(self.cfg["resume_path"])
        self.args_cli = self.cfg["args_cli"]
        self.record_video = self.cfg["args_cli"].video
        self.run_name = os.path.basename(self.run_path)
        # Define the regex pattern and match it against the resume_path
        pattern = r"/rsl_rl/([^/]+)/(?:.*/)?([^/]+)/model_(\d+).*\.pt$"  # works only with rsl_rl
        match = re.search(pattern, self.cfg["resume_path"])
        if match:
            # Extract the experiment name, run name, and checkpoint
            self.experiment_name = match.group(1)
            self.run_name = match.group(2)
            self.checkpoint = int(match.group(3))
        else:
            raise ValueError("The provided resume_path does not match the expected pattern.")
        self.wandb_run_name = f"{self.run_name}_CP{self.checkpoint}_R{self.robot_idx}_S{self.seed}"

    def _upload_code_files(self):
        if self.logger_type == "wandb":
            # Temporarily change the working directory for wandb.save() to maintain the directory structure of /code in wandb
            original_cwd = os.getcwd()  # Save the original working directory
            os.chdir(self.run_path)  # Change to the desired base directory
            code_dir = os.path.join(self.run_path, "code")
            if os.path.exists(code_dir):
                # Loop over all files in the base directory and subdirectories
                for root, _, files in os.walk(code_dir):
                    for file in files:
                        file_path = os.path.join(root, file)  # Full file path
                        relative_file_path = os.path.relpath(file_path)  # Relative file path from base_dir

                        # Print the relative file path
                        wandb.save(relative_file_path)
                print(f"[INFO]: Code state saved to Wandb from path {code_dir}.")
                os.chdir(original_cwd)  # Restore the original working directory
            else:
                print(f"[WARNING]: Code directory not found at path {code_dir}.")

    def log_config(self, config):
        raise NotImplementedError("The log_config method is not implemented yet.")

    def log_scalar(self, tag, value, step=None):
        self.summary_writer.add_scalar(tag, value, global_step=step)

    def log_text(self, tag, text, step=None):
        if hasattr(self.summary_writer, "add_text"):
            self.summary_writer.add_text(tag, text, global_step=step)

    def log_info(self, info):
        # Extract data from info
        signals = info.get("updated_log", {}).get("signals", {})

        # Accumulate the data over time as tensors on GPU
        self.signal_logs["contact_forces_z"] = torch.cat(
            (
                self.signal_logs["contact_forces_z"],
                signals["contact_forces_z"].unsqueeze(0),
            ),
            dim=0,
        )
        self.signal_logs["base_vel_x_y_yaw"] = torch.cat(
            (
                self.signal_logs["base_vel_x_y_yaw"],
                signals["base_vel_x_y_yaw"].unsqueeze(0),
            ),
            dim=0,
        )
        self.signal_logs["commands_x_y_yaw"] = torch.cat(
            (
                self.signal_logs["commands_x_y_yaw"],
                signals["commands_x_y_yaw"].unsqueeze(0),
            ),
            dim=0,
        )
        self.signal_logs["external_force_trunk"] = torch.cat(
            (
                self.signal_logs["external_force_trunk"],
                signals["external_force_trunk"].unsqueeze(0),
            ),
            dim=0,
        )
        if signals["terrain_levels"] is not None:
            self.signal_logs["terrain_levels"] = torch.cat(
                (
                    self.signal_logs["terrain_levels"],
                    signals["terrain_levels"].unsqueeze(0),
                ),
                dim=0,
            )

        # Update self.metric_logs from metrics
        self.metric_logs = info.get("updated_log", {}).get("metrics", {})
        self.time = info.get("updated_log", {}).get("sim_time", 0)

    def upload_logs(self):
        # Log metrics
        if self.logger_type == "wandb":
            # Update summary metrics
            for key, value in self.metric_logs.items():
                wandb.summary[key] = value

            # Prepare data for plotting
            tag = "Evaluation Signals"
            time = np.linspace(0, self.time, self.signal_logs["base_vel_x_y_yaw"].shape[0])

            print("[INFO]: Plotting states to Wandb")

            # Create a dictionary to batch log data
            wandb_log_data = {}

            # Log Contact Forces for each leg
            legs = ["FL", "FR", "RL", "RR"]
            for i, leg in enumerate(legs):
                plot = wandb.plot.line_series(
                    xs=time,
                    ys=[self.signal_logs["contact_forces_z"][:, i].cpu().numpy()],
                    keys=[f"contact_forces_z_{i+1}"],
                    title=f"Contact Force {leg}",
                    xname="Time (sec)",
                )
                wandb_log_data[f"{tag}/Forces/Contact {leg}"] = plot

            # Log Base Velocity and Commands for X, Y, Yaw
            components = ["X", "Y", "Yaw"]
            for i, comp in enumerate(components):
                plot = wandb.plot.line_series(
                    xs=time,
                    ys=[
                        self.signal_logs["base_vel_x_y_yaw"][:, i].cpu().numpy(),
                        self.signal_logs["commands_x_y_yaw"][:, i].cpu().numpy(),
                    ],
                    keys=[f"base_vel_{comp.lower()}", f"commands_{comp.lower()}"],
                    title=f"Base Velocity {comp}",
                    xname="Time (sec)",
                )
                wandb_log_data[f"{tag}/Base Velocity/{comp}"] = plot

            # Log External Forces
            plot = wandb.plot.line_series(
                xs=time,
                ys=[
                    self.signal_logs["external_force_trunk"][:, 0].cpu().numpy(),
                    self.signal_logs["external_force_trunk"][:, 1].cpu().numpy(),
                    self.signal_logs["external_force_trunk"][:, 2].cpu().numpy(),
                ],
                keys=["external_force_x", "external_force_y", "external_force_z"],
                title="External Force Trunk",
                xname="Time (sec)",
            )
            wandb_log_data[f"{tag}/Forces/External Force"] = plot

            # Log Terrain Levels if available
            if self.signal_logs["terrain_levels"].shape[0] > 0:
                plot = wandb.plot.line_series(
                    xs=time,
                    ys=[self.signal_logs["terrain_levels"].cpu().numpy()],
                    keys=["terrain_level"],
                    title="Terrain Levels",
                    xname="Time (sec)",
                )
                wandb_log_data[f"{tag}/Terrain Levels"] = plot

            # Plot contact forces if requested
            if self.arg_plot_contact_forces:
                fig = self._plot_contact_forces_wandb(time, self.signal_logs["contact_forces_z"])
                wandb_log_data[f"{tag}/Forces/Contact Forces"] = fig

            # Log all data in a single call
            wandb.log(wandb_log_data)
            print("[INFO]: Data plotted to Wandb")

    def upload_video(self, path=None):
        if self.logger_type == "wandb":
            video_folder_path = os.path.join(self.run_path, "videos")
            if path is None:
                files = os.listdir(video_folder_path)
                matching_videos = [f for f in files if f.startswith(self.wandb_run_name) and f.endswith(".mp4")]
            else:
                matching_videos = [path]
            if matching_videos:
                for idx, video_path in enumerate(matching_videos):
                    print(f"[INFO]: Matching video to upload to wandb: {video_path}")
                    wandb.log(
                        {
                            f"Robot {self.robot_idx}": wandb.Video(
                                video_path,
                                caption=os.path.basename(video_path),
                                format="mp4",
                            )
                        }
                    )
            else:
                print("[WARNING]: No matching video was found for wandb to upload.")

    def close(self):
        if self.logger_type == "wandb":
            wandb.finish()
        else:
            self.summary_writer.close()

    def _plot_contact_forces_wandb(self, time_steps, contact_forces, threshold=0.15):
        # Extract the contact forces for each leg
        forces_FL = contact_forces[:, 0].cpu().numpy()
        forces_FR = contact_forces[:, 1].cpu().numpy()
        forces_RL = contact_forces[:, 2].cpu().numpy()
        forces_RR = contact_forces[:, 3].cpu().numpy()

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_steps, forces_FL, label="Front Left")
        ax.plot(time_steps, forces_FR, label="Front Right")
        ax.plot(time_steps, forces_RL, label="Rear Left")
        ax.plot(time_steps, forces_RR, label="Rear Right")
        ax.set_title("Contact Forces")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Force")
        ax.legend()
        return fig

    def log_dict_scaler(self, data_dict, step=None, parent_tag=None):
        flattened_dict = self._flatten_dict(data_dict)
        for key, value in flattened_dict.items():
            if parent_tag:
                key = f"{parent_tag}/{key}"
            self.log_scalar(key, value, step)

    def _flatten_dict(self, d, parent_key="", sep="/"):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases with batch logging capabilities."""

    def __init__(self, log_dir: str, flush_secs: int, locs: dict):
        """
        Initialize the WandbSummaryWriter.

        Args:
            log_dir (str): Directory where the logs will be saved.
            flush_secs (int): Interval at which the logs will be flushed.
            locs (dict): Configuration dictionary containing Wandb project information.
        """
        super().__init__(log_dir=log_dir, flush_secs=flush_secs)

        experiment_name = getattr(locs["self"], "experiment_name", None)
        project = (experiment_name + "_Eval") if experiment_name else locs["self"].cfg["experiment_name"]
        entity = "amrmousa-m"
        args_cli = getattr(locs["self"], "args_cli", None)
        note = getattr(args_cli, "note", "") or locs["self"].cfg.get("note", "") or ""
        note = note.replace("_", " ") if note else ""
        group = getattr(args_cli, "group", None) or locs["self"].cfg.get("group", None)

        continue_id = None
        wandb_continue_run = locs["self"].cfg.get("wandb_continue_run", None) or getattr(
            args_cli, "wandb_continue_run", None
        )
        if wandb_continue_run:
            continue_id = self.find_run_id_by_name(path=f"{entity}/{project}", run_name=wandb_continue_run)

        wandb.init(
            project=project,
            entity=entity,
            notes=note,
            group=group,
            id=continue_id,
            resume="allow",
        )

        # Change generated name to project-number format
        if len(note):
            wandb.run.name = note.replace(" ", "_") + "_" + wandb.run.name.split("-")[-1]

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        self.batch_log = {}  # Dictionary to hold batch log data
        self.current_step = None  # Track the current step for batch logging

        # Define the step metric for Wandb to allow logging at different steps simultaneously
        wandb.define_metric("Steps")
        wandb.define_metric("*", step_metric="Steps")

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

        # Upload code files to Wandb
        self._upload_code_files(log_dir)

    def store_config(self, **configs):
        """
        Store configuration settings in Wandb.

        Args:
            **configs: Arbitrary keyword arguments representing configuration dictionaries.
        """
        for key, config in configs.items():
            wandb.config.update({key: config})

    def _map_path(self, path):
        """
        Map the logging path to the appropriate Wandb path.

        Args:
            path (str): The original path of the scalar.

        Returns:
            str: The mapped path.
        """
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        """
        Add a scalar value to the logs and automatically flush if the global step increases.

        Args:
            tag (str): Tag for the scalar value.
            scalar_value (float): The scalar value to log.
            global_step (int, optional): The global step at which this scalar is logged. Defaults to None.
            walltime (float, optional): Wall time at which this scalar is logged. Defaults to None.
            new_style (bool, optional): Flag for new logging style in TensorBoard. Defaults to False.
        """
        if global_step is not None and self.current_step is not None:
            if global_step != self.current_step:
                self.flush_batch()  # Automatically flush if the step changes

        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )

        # Prepare for batch logging
        mapped_tag = self._map_path(tag)
        self.batch_log[mapped_tag] = scalar_value
        self.current_step = global_step

    def flush_batch(self):
        """
        Flush the accumulated logs in batch to Wandb.

        This method is called automatically when the global step changes,
        or it can be called manually.
        """
        if self.batch_log:
            wandb.log({**self.batch_log, "Steps": self.current_step})
            self.batch_log.clear()  # Clear the batch log after flushing

    def stop(self):
        """
        Stop the Wandb logger and ensure all logs are flushed before finishing.
        """
        self.flush_batch()  # Ensure all data is logged before stopping
        wandb.finish()

    def log_config(self, **configs):
        """
        Log the configuration settings to Wandb.

        Args:
            **configs: Arbitrary keyword arguments representing configuration dictionaries.
        """
        self.store_config(**configs)

    def save_model(self, model_path):
        """
        Save the model to Wandb.

        Args:
            model_path (str): Path to the model file.
        """
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        """
        Save a file to Wandb.

        Args:
            path (str): Path to the file.
            iter (int, optional): Iteration number (optional, not used).
        """
        wandb.save(path, base_path=os.path.dirname(path))

    def flush(self):
        """
        Flush both batch logs and TensorBoard logs.

        This method overrides the base flush method to ensure that both Wandb
        and TensorBoard logs are flushed correctly.
        """
        self.flush_batch()  # Flush batch logs before calling the base flush method
        return super().flush()

    def _upload_code_files(self, run_path):
        # Temporarily change the working directory for wandb.save() to maintain the directory structure of /code in wandb
        original_cwd = os.getcwd()  # Save the original working directory
        os.chdir(run_path)  # Change to the desired base directory
        code_dir = os.path.join(run_path, "code")
        if os.path.exists(code_dir):
            # Loop over all files in the base directory and subdirectories
            for root, _, files in os.walk(code_dir):
                for file in files:
                    file_path = os.path.join(root, file)  # Full file path
                    relative_file_path = os.path.relpath(file_path)  # Relative file path from base_dir

                    # Print the relative file path
                    wandb.save(relative_file_path)
            print(f"[INFO]: Code state saved to Wandb from path {code_dir}.")
            os.chdir(original_cwd)  # Restore the original working directory
        else:
            print(f"[WARNING]: Code directory not found at path {code_dir}.")

    @staticmethod
    def find_run_id_by_name(path, run_name):
        api = wandb.Api()
        runs = api.runs(path)
        for run in runs:
            if run.name == run_name:
                return run.id
        return None

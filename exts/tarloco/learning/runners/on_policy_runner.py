# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the rsl_rl repository:
# https://github.com/leggedrobotics/rsl_rl
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from __future__ import annotations

import copy
import json
import os
import statistics
import time
from collections import deque

import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from tqdm import tqdm

import exts.tarloco
from exts.tarloco.learning.algorithms import *  # type: ignore
from exts.tarloco.learning.modules import *  # type: ignore
from exts.tarloco.utils import get_git_root, store_changed_codes


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", **kwargs):
        self.cfg = train_cfg
        self.alg_cfg, self.policy_cfg = train_cfg["algorithm"].copy(), train_cfg["policy"].copy()
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        self.num_obs = obs.shape[1:]  # allow for multi-dim obs in history buffer
        self.num_obs_flat = torch.prod(torch.tensor(self.num_obs)).item()  # allow for multi-dim obs in history buffer
        if "critic" in extras["observations"]:
            self.num_critic_obs = extras["observations"]["critic"].shape[1:]
            self.num_critic_obs_flat = torch.prod(torch.tensor(self.num_critic_obs)).item()
        else:
            self.num_critic_obs = self.num_obs
            self.num_critic_obs_flat = self.num_obs_flat
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic = actor_critic_class(
            num_actor_obs=self.num_obs_flat,
            num_critic_obs=self.num_critic_obs_flat,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        getattr(actor_critic, "post_init", lambda: None)()
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_flat], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.num_critic_obs_flat], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            list(self.num_obs),
            None if self.num_critic_obs == self.num_obs else list(self.num_critic_obs),
            [self.env.num_actions],
        )

        # Log
        self.log_dir = str(log_dir)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [exts.tarloco.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # Store the current code state and initialize the logger
        git_file_paths = self.store_code_state()
        self._initialize_writer()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self.alg.init_schedules(tot_iter)
        # Initialize progress bar
        self.progress_bar = tqdm(
            total=locals()["tot_iter"],
            leave=True,
            bar_format=(
                "\033[1m Learning iteration \033[0m {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
                " [{elapsed}<{remaining}, {rate_fmt}]-> {desc}"
            ),
        )

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rew_buffer = deque(maxlen=100)
        len_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    # all_actions[i] = actions
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs = self.obs_normalizer(obs.view(obs.size(0), -1)).view_as(obs)
                    # all_obs[i] = obs
                    if "critic" in infos["observations"]:
                        critic_obs_size = infos["observations"]["critic"].size()
                        critic_obs = self.critic_obs_normalizer(
                            infos["observations"]["critic"].reshape(critic_obs_size[0], -1)
                        ).reshape(critic_obs_size)
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rew_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)  # TODO: check if you need to pass hidden states

            mean_loss = self.alg.update(it=it)
            skip_learn = mean_loss is None
            if skip_learn:
                # load latest model in the log folder and continue if no the model weights exploded
                self.load(self.get_best_model(), load_optimizer=True)
                self.alg.actor_critic.nan_detected = False
            stop = time.time()
            learn_time = stop - start
            rew_mean = statistics.mean(rew_buffer) if len(rew_buffer) > 0 else 0
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and not skip_learn:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), rew_mean)
            ep_infos.clear()
            if it == start_iter:
                # if possible store them to wandb
                if self.logger_type == "wandb" and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"), rew_mean)
        # Close the progress bar when done
        self.progress_bar.close()

    def _initialize_writer(self):
        if self.log_dir is not None:
            # Launch either Tensorboard or wandb summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from exts.tarloco.utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, locs=locals())
                # log config only if not continuing a run
                if not self.cfg["wandb_continue_run"]:
                    self.writer.log_config(
                        env_cfg=self.env.cfg,
                        runner_cfg=self.cfg,
                        alg_cfg=self.cfg["algorithm"],
                        policy_cfg=self.cfg["policy"],
                    )

            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("logger type not found")
        else:
            raise ValueError("log_dir not found")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                info_tensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, ep_info[key].to(self.device)))
                value = torch.mean(info_tensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # Dynamic Logging for Mean Losses
        if locs["mean_loss"] is not None:
            for loss_name, loss_value in locs.get("mean_loss", {}).items():
                self.writer.add_scalar(f"Loss/{loss_name}", loss_value or 0.0, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.lr, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rew_buffer"]) > 0:
            mean_reward = statistics.mean(locs["rew_buffer"])
            reward_iter_score = mean_reward * locs["it"] / locs["tot_iter"]
            alpha = 0.95  # EMA decay
            rew_weight = 0.6  # weight of reward vs terrain level in the EMA
            if not hasattr(self, "ema_reward_iter_score"):
                self.ema_reward_iter_score = reward_iter_score
            curr_power_value = (
                locs["infos"]["log"].get("Curriculum/terrain_levels", 0) / 5
            ) ** 0.5 * 5  # map to [0, 5]
            self.ema_reward_iter_score = alpha * self.ema_reward_iter_score + (1 - alpha) * (
                rew_weight * reward_iter_score + (1 - rew_weight) * curr_power_value
            )
            self.writer.add_scalar("Train/mean_reward", mean_reward, locs["it"])
            self.writer.add_scalar("Train/reward_iter_score", self.ema_reward_iter_score, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["len_buffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", mean_reward, self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["len_buffer"]),
                    self.tot_time,
                )

            # Update the progress bar with the latest data
            self.update_progress_bar(locs, fps, mean_std, iteration_time)

    # Function to update the progress bar
    def update_progress_bar(self, locs, fps, mean_std, iteration_time):
        # Basic computation info
        comp_info = f"collect: {locs['collection_time']:.3f}s, learn {locs['learn_time']:.3f}s"
        # Totals and time
        time_info = f"timesteps: {self.tot_timesteps/1e6:.3f}M"
        loss_info = (
            f"vf: {(locs['mean_loss']['value_function'] or 0.0):.4f}, "
            f"surr: {(locs['mean_loss']['surrogate'] or 0.0):.4f}, "
            if locs["mean_loss"] is not None
            else "NaN"
        )
        mean_info = (
            f"rew: {statistics.mean(locs['rew_buffer']):.2f}, "
            f"eps_len: {statistics.mean(locs['len_buffer']):.2f}, "
            f"act_noise_std: {mean_std.item():.2f}"
        )
        progress_desc = f"{comp_info} @{time_info} | LOSS> {loss_info} | MEAN> {mean_info}"

        # Update tqdm's description and refresh the bar
        self.progress_bar.set_description_str(progress_desc)
        self.progress_bar.update(1)

    def save(self, path, rew_mean, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Check if this is the best model
        if not hasattr(self, "best_rew_mean") or rew_mean > self.best_rew_mean:
            # Update best reward mean
            self.best_rew_mean = rew_mean
            # Save the new best model with rew_mean in the filename
            self.best_model_path = os.path.join(os.path.dirname(path), "model_best.pt")
            torch.save(saved_dict, self.best_model_path)

            # Save metadata to model_best.metadata
            metadata_path = os.path.join(os.path.dirname(path), "model_best.metadata")
            with open(metadata_path, "w") as metadata_file:
                metadata = {
                    "best_rew_mean": self.best_rew_mean,
                    "iter": self.current_learning_iteration,
                    "path": self.best_model_path,
                }
                json.dump(metadata, metadata_file)

            # Upload the best model to W&B
            if self.logger_type == "wandb":
                self.writer.save_model(self.best_model_path)
                self.writer.save_file(metadata_path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_flat], until=1.0e8).to(self.device)
            obs_norm_state_dict = loaded_dict["obs_norm_state_dict"]
            if "count" not in obs_norm_state_dict:
                obs_norm_state_dict["count"] = self.obs_normalizer.state_dict()["count"]
            self.obs_normalizer.load_state_dict(obs_norm_state_dict)

            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.num_critic_obs_flat], until=1.0e8).to(
                self.device
            )
            critic_obs_norm_state_dict = loaded_dict["critic_obs_norm_state_dict"]
            if "count" not in critic_obs_norm_state_dict:
                critic_obs_norm_state_dict["count"] = self.critic_obs_normalizer.state_dict()["count"]
            self.critic_obs_normalizer.load_state_dict(critic_obs_norm_state_dict)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_latest_model(self):
        # List all files in the log directory
        files = os.listdir(self.log_dir)
        # Filter for model files (assuming they have a specific extension, e.g., '.pt')
        model_files = [f for f in files if f.endswith(".pt")]
        if not model_files:
            return None
        # Get the full path for each model file
        model_files = [os.path.join(self.log_dir, f) for f in model_files]
        # Find the most recently modified model file
        latest_model = max(model_files, key=os.path.getmtime)
        return latest_model

    def get_best_model(self):
        # List all files in the log directory
        files = os.listdir(self.log_dir)
        # Filter for model files that have 'best' in their filename
        best_model_files = [f for f in files if "best" in f and f.endswith(".pt")]
        if not best_model_files:
            print("No best model found. Returning the latest model.")
            return self.get_latest_model()
        # Get the full path for each best model file
        best_model_files = [os.path.join(self.log_dir, f) for f in best_model_files]
        # Find the most recently modified best model file
        best_model = max(best_model_files, key=os.path.getmtime)
        return best_model

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)

            def policy(x):
                return self.alg.actor_critic.act_inference(
                    self.obs_normalizer(x.view(x.size(0), -1)).view_as(x)
                )  # noqa: E731

        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_paths):
        if isinstance(repo_file_paths, str):
            repo_file_paths = [repo_file_paths]
        for repo_file_path in repo_file_paths:
            self.git_status_repos.append(get_git_root(repo_file_path))  # type: ignore

    def store_code_state(self, log_dir=None):
        log_dir = log_dir or self.log_dir
        return store_changed_codes(log_dir, self.git_status_repos)

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

import os
import pathlib
import shutil
from typing import Tuple

import git
import numpy as np
import torch


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the input has the following dimension order: [time, number of envs, additional dimensions]
    """
    # If tensor has four dimensions, make them three and stope the original shape to use it before returning
    tensor_orig_shape = tensor.shape
    if len(tensor_orig_shape) > 3:
        tensor = tensor.view(tensor.size(0), tensor.size(1), -1)
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # add at least one full length trajectory
    trajectories = trajectories + (torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),)
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]
    if len(tensor_orig_shape) > 3:
        padded_trajectories = padded_trajectories.view(*padded_trajectories.shape[:2], *tensor_orig_shape[-2:])
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_changed_codes(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "code")
    if os.path.exists(git_log_dir):
        shutil.rmtree(git_log_dir)
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    repo_dirs = []  # Stores repo directories to return

    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            continue

        # Get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name

        # Get the current commit hash
        commit_hash = repo.head.commit.hexsha

        # Get the remote URL
        try:
            remote_url = repo.remote().url
        except ValueError:
            print(f"Repository '{repo_name}' does not have a remote URL. Skipping.")
            continue

        # Construct the link to the Git commit
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]  # Remove '.git' suffix for proper URL format
        commit_link = f"{remote_url}/commit/{commit_hash}"

        # Save the commit link to a file
        commit_link_file = os.path.join(git_log_dir, f"{repo_name}_commit_link.txt")
        with open(commit_link_file, "w") as f:
            f.write(commit_link)
        print(f"[INFO]: Commit link saved to: {commit_link_file}")

        # Get the list of changed files
        changed_files = (
            repo.git.diff("--name-only", "HEAD").splitlines()
            + repo.git.ls_files("--others", "--exclude-standard").splitlines()
        )

        if not changed_files:
            print(f"[INFO]: No changes detected in {repo_name}. Skipping.")
            continue  # Skip to the next repository

        # Create a directory inside logdir to store the files from this repository
        repo_files_dir = os.path.join(git_log_dir, repo_name)
        os.makedirs(repo_files_dir, exist_ok=True)
        repo_dirs.append(repo_files_dir)  # Track this repo's directory

        for file in changed_files:
            full_file_path = os.path.join(repo.working_dir, file)

            # Skip if it's not a file
            if not os.path.isfile(full_file_path):
                continue

            # Create the destination path for the file
            dest_file_path = os.path.join(git_log_dir, file)
            dest_file_dir = os.path.dirname(dest_file_path)
            os.makedirs(dest_file_dir, exist_ok=True)  # Create directories if they don't exist

            # Copy the file to the destination directory
            shutil.copy2(full_file_path, dest_file_path)

            # Add the file path to the list of files to be uploaded
            file_paths.append(dest_file_path)
        print(f"[INFO]: Changed codes were copied to {repo_files_dir}")

    return repo_dirs


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip((input - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = torch.nn.functional.normalize

    def forward(self, x):
        x = self.normalize(x, dim=-1)
        return x

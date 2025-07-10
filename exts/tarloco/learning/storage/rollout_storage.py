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


from typing import Tuple

import torch

from exts.tarloco.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.critic_observations = None
            self.next_critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.next_hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device="cpu"):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
            self.next_privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)  # type: ignore
        self.next_observations[self.step].copy_(transition.next_observations)  # type: ignore
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)  # type: ignore
            self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)  # type: ignore
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))  # type: ignore
        self.dones[self.step].copy_(transition.dones.view(-1, 1))  # type: ignore
        self.values[self.step].copy_(transition.values)  # type: ignore
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))  # type: ignore
        self.mu[self.step].copy_(transition.action_mean)  # type: ignore
        self.sigma[self.step].copy_(transition.action_sigma)  # type: ignore
        self._save_hidden_states(transition.hidden_states, transition.next_hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states, next_hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a Tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], Tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], Tuple) else (hidden_states[1],)

        # Handle next hidden states
        next_hid_a = next_hidden_states[0] if isinstance(next_hidden_states[0], Tuple) else (next_hidden_states[0],)
        next_hid_c = next_hidden_states[1] if isinstance(next_hidden_states[1], Tuple) else (next_hidden_states[1],)

        # Initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
            self.saved_next_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *next_hid_a[i].shape, device=self.device)
                for i in range(len(next_hid_a))
            ]
            self.saved_next_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *next_hid_c[i].shape, device=self.device)
                for i in range(len(next_hid_c))
            ]

        # Copy the current hidden states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])  # type: ignore

        # Copy the next hidden states
        for i in range(len(next_hid_a)):
            self.saved_next_hidden_states_a[i][self.step].copy_(next_hid_a[i])
            self.saved_next_hidden_states_c[i][self.step].copy_(next_hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_Tuple=False)[:, 0]))  # type: ignore
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
            next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = observations
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield {
                    "obs": observations[batch_idx],
                    "next_obs": next_observations[batch_idx],
                    "critic_obs": critic_observations[batch_idx],
                    "next_critic_obs": next_critic_observations[batch_idx],
                    "actions": actions[batch_idx],
                    "target_values": values[batch_idx],
                    "returns": returns[batch_idx],
                    "old_actions_log_prob": old_actions_log_prob[batch_idx],
                    "advantages": advantages[batch_idx],
                    "old_mu": old_mu[batch_idx],
                    "old_sigma": old_sigma[batch_idx],
                    "hid_states": (None, None),  # No RNN states for feedforward policies
                    "next_hid_states": (None, None),
                    "masks": None,  # No masks required for standard batch processing
                    "env_idx": batch_idx // self.num_transitions_per_env,
                }

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_next_obs_trajectories, next_trajectory_masks = split_and_pad_trajectories(
            self.next_observations, self.dones
        )

        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
            padded_next_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.next_privileged_observations, self.dones
            )
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories
            padded_next_critic_obs_trajectories = padded_obs_trajectories

        assert torch.all(
            trajectory_masks == next_trajectory_masks
        ).item(), "Trajectory masks should be the same for observations and next observations"
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                next_obs_batch = padded_next_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                next_critic_obs_batch = padded_next_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a  # type: ignore
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c  # type: ignore
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                # Prepare next hidden states
                next_hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_next_hidden_states_a
                ]
                next_hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_next_hidden_states_c
                ]
                # remove the tuple for GRU
                next_hid_a_batch = next_hid_a_batch[0] if len(next_hid_a_batch) == 1 else next_hid_a_batch
                next_hid_c_batch = next_hid_c_batch[0] if len(next_hid_c_batch) == 1 else next_hid_c_batch
                yield {
                    "obs": obs_batch,
                    "next_obs": next_obs_batch,
                    "critic_obs": critic_obs_batch,
                    "next_critic_obs": next_critic_obs_batch,
                    "actions": actions_batch,
                    "target_values": values_batch,
                    "advantages": advantages_batch,
                    "returns": returns_batch,
                    "old_actions_log_prob": old_actions_log_prob_batch,
                    "old_mu": old_mu_batch,
                    "old_sigma": old_sigma_batch,
                    "hid_states": (hid_a_batch, hid_c_batch),
                    "next_hid_states": (next_hid_a_batch, next_hid_c_batch),
                    "masks": masks_batch,
                    "epoch": ep,
                    "mini_batch_idx": i,
                }

                first_traj = last_traj

# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the following repositories:
# https://github.com/leggedrobotics/rsl_rl
# HIMLoco: https://github.com/OpenRobotLab/HIMLoco
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

import math
import yaml
import os
import random
import subprocess
from dataclasses import fields, is_dataclass
from typing import Tuple, Any, List, Union

import gymnasium as gym
import numpy as np
import torch
from isaacsim.core.utils.viewports import set_camera_view


def set_robot_camera_view(robot_position, step_count, radius=2.0, height=1.0):
    robot_position_cpu = robot_position.detach().cpu().numpy()
    angle = step_count * 0.01  # Adjust the speed of rotation
    eye = robot_position_cpu + np.array([radius * math.cos(angle), radius * math.sin(angle), height])
    target = robot_position_cpu
    set_camera_view(eye, target)


def seed_everything(seed):
    import isaacsim.core.utils.torch as torch_utils

    # import omni.replicator.core as rep

    print(f"[INFO]: Setting everything's seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # rep.set_global_seed(seed)
    torch_utils.set_seed(seed)


def get_attr_recursively(env, attr_name):
    """Recursively search for an attribute in the environment or its wrappers."""
    current_env = env

    while hasattr(current_env, "env"):
        if hasattr(current_env, attr_name):
            return getattr(current_env, attr_name)
        current_env = current_env.env

    # Final check if the attribute is in the unwrapped environment
    if hasattr(current_env, attr_name):
        return getattr(current_env, attr_name)

    raise AttributeError(f"Attribute '{attr_name}' not found in the environment or its wrappers.")


def remove_empty_dicts(d):
    """Recursively remove all keys with empty dictionary values from a nested dictionary."""
    if isinstance(d, dict):
        return {k: remove_empty_dicts(v) for k, v in d.items() if v != {}}
    return d


def get_git_root(paths: Union[str, List[str]]) -> Union[str, List[str]]:
    """Finds the root directory of the Git repository for each path in `paths`.
    Accepts a single path (str) or a list of paths (list) and returns a single root (str) or a list of roots (list).
    """

    def find_git_root(path: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=os.path.dirname(os.path.abspath(path)),
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            return path

    if isinstance(paths, str):
        return find_git_root(paths)
    else:  # Assume it is list
        return [find_git_root(path) for path in paths]


class RecordVideo(gym.wrappers.RecordVideo):
    def start_recording(self, video_name: str):
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        if self.recording:
            self.stop_recording()

        self.recording = True
        self._video_name = video_name
        self._video_path = os.path.join(self.video_folder, f"{video_name}.mp4")


# ------------------
# Hydra functions
# ------------------

def safe_asdict(obj):
    """Recursively convert dataclass to dict, ignoring inaccessible fields."""
    if not is_dataclass(obj):
        return obj
    result = {}
    for f in fields(obj):
        try:
            value = getattr(obj, f.name)
            result[f.name] = safe_asdict(value)
        except Exception as e:
            result[f.name] = f"[error: {e}]"
    return result


def dump_hydra_config(cfg: Tuple[Any, Any, Any], logdir: str) -> None:
    """
    Dump the Hydra configuration tree to a YAML file in the log directory.

    Args:
        cfg (Tuple): Tuple containing (args, env_config, agent_config)
        logdir (str): Path to the directory where the config should be saved
    """
    hydra_config_path = os.path.join(logdir, "hydra_config.yaml")
    os.makedirs(os.path.dirname(hydra_config_path), exist_ok=True)  # Make sure directory exists

    try:
        hydra_config = {
            "args": cfg[0],
            "env": safe_asdict(cfg[1]),
            "agent": safe_asdict(cfg[2]),
        }

        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_hydra_config = flatten_dict(hydra_config)

        os.makedirs(logdir, exist_ok=True)
        with open(hydra_config_path, "w") as f:
            yaml.dump(flat_hydra_config, f, default_flow_style=False)

        print(f"[INFO]: Flattened Hydra config dumped to {hydra_config_path}")

    except Exception as e:
        print(f"[ERROR]: Failed to dump Hydra config: {e}")


def replace_string_in_object(obj, target_string, replacement):
    """
    Recursively searches through an object (dict, list, object attributes)
    and replaces occurrences of `target_string` with `replacement`.
    """
    if isinstance(obj, dict):  # If it's a dictionary, loop over key-value pairs
        for key, value in list(obj.items()):
            if key == target_string:  # Replace key if it matches
                obj[replacement] = obj.pop(key)
            obj[key] = replace_string_in_object(value, target_string, replacement)

    elif isinstance(obj, list):  # If it's a list, loop over elements
        for i in range(len(obj)):
            obj[i] = replace_string_in_object(obj[i], target_string, replacement)

    elif isinstance(obj, str):  # If it's a string, check if it's the target
        return replacement if obj == target_string else obj

    elif hasattr(obj, "__dict__") and not isinstance(obj, type):  # Ensure obj is NOT a class
        for attr, value in vars(obj).items():
            if attr == target_string:  # Replace attribute name if it matches
                setattr(obj, replacement, getattr(obj, attr))
                delattr(obj, attr)
            setattr(obj, attr, replace_string_in_object(value, target_string, replacement))

    return obj  # Return the modified object

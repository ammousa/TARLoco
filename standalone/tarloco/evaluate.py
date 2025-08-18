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

"""Script to evaluate a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import re
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to simulate.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
# Single robot
parser.add_argument(
    "--robot_idx",
    type=int,
    default=0,
    help="Index of the robot used for logging time signals.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--cam_view",
    type=str,
    default="robot",
    help="Select either 'robot' or 'free' to control the camera view. Default: 'robot'",
)
parser.add_argument(
    "--teacher_path",
    type=str,
    default=None,
    help="Path for teacher model to use.",
)
parser.add_argument(
    "--plot_contact_forces",
    action="store_true",
    default=False,
    help="Plot contact forces during evaluation. Default: False",
)
parser.add_argument(
    "--export_model",
    type=str,
    default=None,
    help="Export the model to onnx or jit. Default: None",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
original_argv = sys.argv[:]  # Make a copy of the original sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import traceback

import carb
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from tqdm import tqdm

from exts.tarloco.envs.wrappers import EvaluateWrapper
from exts.tarloco.tasks import registry
from exts.tarloco.utils import LoggerWrapper, get_checkpoint_path
from exts.tarloco.utils.exporter import export_policy_as_jit, export_policy_as_onnx
from exts.tarloco.utils.utils import (
    RecordVideo,
    get_attr_recursively,
    seed_everything,
    set_robot_camera_view,
)

# Get the task configuration from the registry based on the task name
task_config = registry[args_cli.task]
if not task_config:
    raise ValueError(
        f"Task '{args_cli.task}' is not found in the registry.\n"
        f"Available tasks are: {', '.join(registry.keys())}\n"
        "Please check the task name and try again."
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: task_config.agent_cfg):  # type: ignore
    """Play with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.note = args_cli.note
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # seed everything for reproducibility
    seed_everything(args_cli.seed)
    # note: certain randomization occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = agent_cfg.device = args_cli.device
    sys.argv = original_argv  # Restore the original sys.argv before wandb initialization

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO]: Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    run_path = args_cli.log_dir = os.path.dirname(resume_path)
    checkpoint = int(re.search(r"model_(\d+).*\.pt$", resume_path).group(1))  # type: ignore
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # wrap for video recording
    video_name_prefix = (
        f"{os.path.basename(run_path)}_CP{checkpoint}_R{args_cli.robot_idx}_S{args_cli.seed}"
        if args_cli.note is None
        else args_cli.note.replace(" ", "_")
    )
    if args_cli.video:
        env_cfg.viewer.resolution = (640, 480)  # type: ignore
        video_kwargs = {
            "video_folder": os.path.join(run_path, "videos"),
            # "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "disable_logger": False,
            "name_prefix": video_name_prefix,
        }
        print("[INFO]: Recording videos during training.")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = RecordVideo(env, **video_kwargs)  # type: ignore
    # Apply environment wrappers from the registry (IN ORDER)
    for wrapper in task_config.env_wrappers:
        env = wrapper(env, args_cli=args_cli)
    # wrap around environment for evaluation
    env = EvaluateWrapper(env, args_cli.robot_idx)

    # simulate environment
    num_episodes = args_cli.num_episodes
    max_episode_length = env.unwrapped.max_episode_length  # type: ignore
    dt = env.unwrapped.step_dt  # type: ignore
    sim_time = num_episodes * max_episode_length * dt
    num_steps = int(sim_time / dt)
    print(f"Simulating for {sim_time} seconds.")
    print(f"dt: {dt} seconds.")
    print(f"Num steps: {num_steps}")
    progress_bar = tqdm(total=num_steps, desc="Simulating")

    # Load algo-soecific configurations
    # Teacher student training
    if args_cli.teacher_path is not None:
        runner_kwargs = {"teacher_path": args_cli.teacher_path}
    else:
        runner_kwargs = {}

    # Instantiate the runner from the registry
    runner_class = task_config.runner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, **runner_kwargs)  # type: ignore
    runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)  # type: ignore

    # export policy to onnx/jit
    if args_cli.export_model:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_func = export_policy_as_jit if args_cli.export_model == "jit" else export_policy_as_onnx
        export_func(
            actor_critic=runner.alg.actor_critic,
            normalizer=runner.obs_normalizer,
            path=export_model_dir,
        )
        print(f"[INFO]: Exported the {args_cli.export_model} model to: {export_model_dir}")

    # reset environment
    obs, _ = env.get_observations()

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # logging setup for evaluation
    enable_logger = args_cli.logger
    if enable_logger:
        experiment_cfg = {"resume_path": resume_path, "dt": dt, "args_cli": args_cli}
        logger = LoggerWrapper(
            cfg={
                **experiment_cfg,
                "runner_cfg": agent_cfg.to_dict(),
                "env_cfg": env_cfg.to_dict(),  # type: ignore
            }
        )

    # init vars
    step_count = 0

    while simulation_app.is_running() and step_count < num_steps - 1:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _, info = env.step(actions)

            # logging signals
            if enable_logger:
                logger.log_info(info)

            # Focus the camera on the specified robot
            if args_cli.cam_view == "robot" and (args_cli.video or not args_cli.headless):
                set_robot_camera_view(info["updated_log"]["robot_position"], step_count)

        step_count += 1
        progress_bar.update(1)

    # close environment first to render videos
    env.print_metrics()
    env.close()
    # logging metrics and video
    if enable_logger:
        logger.upload_logs()
        if args_cli.video:
            video_path = get_attr_recursively(env, "_video_path")
            logger.upload_video(path=video_path)
        logger.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()  # type: ignore
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        os._exit(1)
    finally:
        # close sim app
        simulation_app.close()

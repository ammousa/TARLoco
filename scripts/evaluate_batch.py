import sys
import subprocess

# Define your CUDA device
cuda = 0  # replace with argparse if needed

# ------------------------ Task Setup ------------------------
task_dict = {
    "go1-eval-him-rough": {
        "group": "HIM",
        "load_runs": [
            "2025-07-23_11-25-35_HIM_S0_7",
            "2025-07-23_12-30-21_HIM_S0_9"
        ]
    },
    # Add more tasks as needed
}

# ------------------------ Params Setup ------------------------
params_dict = {
    "ID": {
        "static_friction": "[0.7,1.1]",
        "dynamic_friction": "[0.6,1.0]",
        "mass": "[-2,8]",
        "speed": "[-1.0,1.0]",
    },
    "OOD": {
        "static_friction": "[0.1,0.7]",
        "dynamic_friction": "[0.05,0.6]",
        "mass": "[10,20]",
        "speed": "[-2.0,2.0]",
    }
}

robot_idxes = [10]
models = [f"model_{i}" for i in range(2000, 20000 + 1, 2000)]
models = [m if m != "model_20000" else "model_19999" for m in models]

# ------------------------ Run Counter ------------------------
total_runs = sum(
    2 * len(entry["load_runs"]) * len(models) * len(robot_idxes)  # ×2 for ID and OOD
    for entry in task_dict.values()
)

print(f"CUDA Device: cuda:{cuda}", flush=True)
print(f"Total evaluation runs: {total_runs}", flush=True)

# ------------------------ Execution ------------------------
run_count = 1
for task, metadata in task_dict.items():
    group = metadata["group"]
    for load_run in metadata["load_runs"]:
        seed = load_run.split("S")[-1].split("_")[0]
        for model in models:
            for robot_idx in robot_idxes:
                for regime in ["ID", "OOD"]:
                    print(f"\n[{run_count}/{total_runs}] Task: {task}, Run: {load_run}, Model: {model}, Regime: {regime}", flush=True)
                    P = params_dict[regime]

                    command = [
                        "python", "standalone/tarloco/evaluate.py",
                        "--task", task,
                        "--headless",
                        "--logger", "wandb",
                        "--video",
                        "--num_episodes", "1",
                        "--experiment_name", "TAR_workspace",
                        "--robot_idx", str(robot_idx),
                        "--note", f"{group}_s{seed}_eval_@m{model.split('_')[-1]}_{regime}",
                        "--group", group,
                        "--load_run", load_run,
                        "--device", f"cuda:{cuda}",
                        "--checkpoint", model,
                        f"env.commands.base_velocity.ranges.lin_vel_x={P['speed']}",
                        f"env.commands.base_velocity.ranges.lin_vel_y={P['speed']}",
                        f"env.commands.base_velocity.ranges.ang_vel_z={P['speed']}",
                        f"env.events.physics_material.params.static_friction_range={P['static_friction']}",
                        f"env.events.physics_material.params.dynamic_friction_range={P['dynamic_friction']}",
                        f"env.events.add_base_mass.params.mass_distribution_params={P['mass']}",
                        "agent.experiment_name=TAR_workspace",
                    ]

                    print("Command:", " ".join(command), "\n", flush=True)
                    subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
                    run_count += 1

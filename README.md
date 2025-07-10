🚀 **Exciting News!** The code for this project will be released in ***May 2025***. Stay tuned! 🎉✨


Versions used:
Isaacsim 4.5.0
Isaaclab v2.1.0


How to install?

Install isaacsim as in this guide:


# Create and activate Conda environment
conda create -n tar python=3.10
conda activate tar

# Clone TARLoco repo
git clone https://github.com/ammousa/TARLoco.git
cd TARLoco

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip

# Install IsaacSim SDK and dependencies
pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com

# Clone and set up IsaacLab
git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git _isaaclab

# Install dependencies using apt (on Ubuntu):
sudo apt install cmake build-essential
./_isaaclab/isaaclab.sh --install

# Verify the Isaac Lab installation
python _isaaclab/scripts/tutorials/00_sim/create_empty.py

<!-- # Fix OpenCV compatibility
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python==4.7.0.72

# Install supporting libraries
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
pip install optuna optunahub optuna-dashboard
pip install pandas wandb==0.12.21 gym
``` -->


# In the root of the `TARLoco`, install as editable package:
```bash
pip install -e .
```



## License

This repository is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).


### Attribution

This project builds upon the following open source projects:

- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)

These components retain their original licenses and are included for benchmarking purposes only. We do not relicense or modify their licensing terms.

This repository includes implementations from the following projects:

- [HIMLoco](https://github.com/OpenRobotLab/HIMLoco)
- [SLR](https://github.com/11chens/SLR-master)

Please refer to their original repositories and license files for details. Our license applies to the new contributions in this repository.

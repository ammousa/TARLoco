# 🐾 TARLoco: Teacher-Aligned Representations for Quadrupedal Locomotion

## Overview

**TARLoco** provides a modular and scalable learning framework for blind quadrupedal locomotion, integrating representation learning, contrastive distillation, and sim-to-real transfer. Built as an **Isaac Lab** project extension, it enables agile development, high-fidelity simulation with Isaac Sim, and direct deployment on **Unitree Go2**.

<!-- > [!IMPORTANT]
> The code and pretrained models will be released in **May 2025**. Stay tuned!  
> This framework includes support for **teacher-student training**, **privileged information distillation**, and **t-SNE/UMAP** visualization. -->

| Section | Description |
|---------|-------------|
| 🛠️ **Installation** | Setup with Conda, Isaac Sim, Isaac Lab, and dependencies |
| 🚀 **Training & Evaluation** | Train teacher/student policies, visualize embeddings |
| 🤖 **Sim2Real Deployment** | Transfer to Unitree Go2 robot using LCM/SDK2 |
| 📄 **License & Attribution** | Licensing terms and upstream acknowledgements |

---

## 🛠️ Installation

We test our framework on the following stack:

- **Ubuntu:** 20.04 / 22.04
- **CUDA:** 12.1
- **Python:** 3.10
- **PyTorch:** 2.5.1
- **Isaac Sim:** 4.5.0
- **Isaac Lab:** v2.1.0

### Step-by-step Setup

```bash
# 1. Create Conda environment
conda create -n tar python=3.10
conda activate tar

# 2. Clone the TARLoco repository
git clone https://github.com/ammousa/TARLoco.git
cd TARLoco

# 3. Install PyTorch with CUDA 12.1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip

# 4. Install IsaacSim SDK (v4.5.0)
pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com

# 5. Clone IsaacLab and install
git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git _isaaclab
sudo apt install cmake build-essential
./_isaaclab/isaaclab.sh --install

# 6. Test empty simulation
python _isaaclab/scripts/tutorials/00_sim/create_empty.py --headless

# 7. Install TARLoco as an editable package
pip install -e .
```

---

## 🚀 Training & Evaluation

We provide a unified pipeline for:
- Learning privileged teacher policies in simulation.
- Training student agents with proprioceptive observations.
- Distilling representations using contrastive self-supervision.
- Visualizing latent embeddings with t-SNE and UMAP.

### Train a policy (coming soon)

```bash
# Train teacher
python scripts/train_teacher.py --env Go2MassFriction

# Train student with TAR
python scripts/train_student.py --teacher checkpoints/teacher.pth
```

---

## 🤖 Sim2Real Deployment

The policy trained in Isaac Sim can be deployed to **Unitree Go2** via a lightweight deployment interface using **LCM** and **Unitree SDK2**.

### Hardware Requirements

- Unitree Go2 Robot
- Ubuntu 22.04 (recommended)
- LCM pre-installed in SDK2

### Deploy

```bash
# Send trained policy to robot
python deploy/send_policy.py --checkpoint checkpoints/student.pth
```

Real-world tests span varied terrains, surface frictions, and robot masses.

---

## 📄 License

This repository is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🙏 Attribution

This project builds on and incorporates code from:

- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)

We also include comparison baselines or adapted components from:

- [HIMLoco](https://github.com/OpenRobotLab/HIMLoco)
- [SLR](https://github.com/11chens/SLR-master)

These components retain their original licenses. Only new contributions in this repository are licensed under Apache 2.0.

---

## 🔗 Citation

If you use TARLoco in your research, please cite:

```bibtex
@inproceedings{mousa2025tar,
  title={TAR: Teacher-Aligned Representations via Contrastive Learning for Quadrupedal Locomotion},
  author={Mousa, Amr and Pan, Wei and Allmendinger, Richard and Karavis, Neil},
  booktitle={IROS},
  year={2025}
}
```

---

## 🎥 Demos & Website

Coming soon:
- Pretrained model downloads
- Deployment videos on Unitree Go2
- Evaluation results across non-stationary conditions

Stay tuned!

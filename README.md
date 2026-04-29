# CA-AC-MPC: CUDA-Accelerated Actor-Critic Model Predictive Control

This repository contains the implementation of **CA-AC-MPC**, a high-performance framework that integrates Model Predictive Control with Reinforcement Learning to enable agile control of complex dynamical systems, specifically optimized for quadrotor gate racing.

This is the code accompanying the paper: 

A. Buo, V. Cammarota, M. Avagnale, P. Arpenti, V. Lippiello, F. Ruggiero, "CA-AC-MPC: CUDA-Accelerated Actor-Critic Model Predictive Control", 2026 International Conference on Unmanned Aircraft Systems (ICUAS), Corfu, Greece

## Project Description

Actor-Critic Model Predictive Control (AC-MPC) is a hybrid control paradigm where a neural network (the actor) predicts the cost parameters for a differentiable MPC layer. This allows the system to benefit from the interpretability and constraint-handling of MPC while leveraging the adaptive learning capabilities of RL.

The core contribution of this repository is **CA-DiffMPC**, a CUDA-accelerated differentiable MPC solver. Traditionally, the differentiable MPC layer is a computational bottleneck due to the need for repeated optimization and gradient propagation. Our implementation introduces fused C++/CUDA kernels for the iterative Linear Quadratic Regulator (iLQR) algorithm, achieving up to a **10x speed-up** in inference and significantly reducing training times without compromising control performance.

### Key Features
- **Fused CUDA Kernels**: Highly optimized iLQR implementation for both forward and backward passes.
- **Differentiable MPC**: Seamless integration with PyTorch for end-to-end training.
- **Flexible Backends**: Support for `FastMPC` (CUDA), `DiffMPC` (PyTorch), and the original `mpc.pytorch`.
- **Agile Drone Racing**: Pre-configured environments and tracks for benchmarking high-speed flight.

## Requirements

### Python Environment
Ensure you have a Python environment (3.8+) with the following dependencies:
- PyTorch (with CUDA support for the fast backend)
- Gym (0.21.0)
- NumPy, PyYAML, Matplotlib, Pandas
- tqdm, rich

Example setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy gym==0.21 pyyaml matplotlib pandas tqdm rich
```

### CUDA Requirements
The `fast` backend requires a working CUDA compiler toolchain (`nvcc`). The extension is compiled automatically on the first run. Ensure your `CUDA_HOME` environment variable is correctly set.

## Repository Layout

```text
├── train_acmpc.py                # Main training entrypoint
├── configs/                      # YAML training presets
├── envs/                         # Quadrotor environment and track definitions
│   ├── gate_racing_env.py        # Gym environment logic
│   └── tracks/                   # Track YAML files (Zurich, Abu Dhabi, etc.)
├── DifferentialMPC/              # Differentiable MPC (PyTorch backend)
├── differentialMPCPerformance/   # CUDA-accelerated MPC (Fast backend)
├── acmpc_public-master/          # Shared modules and baseline implementations
│   ├── mpc.pytorch/              # Vendored mpc.pytorch backend
│   └── training_modules/         # Actor/Critic policy architectures
└── utils/                        # Support utilities for config, eval, and plotting
```

## Usage

### Training
Start a training session by specifying a configuration file:
```bash
python train_acmpc.py --config configs/train_acmpc_fixed_map.yaml
```

### Configuration Overrides
You can override any parameter from the YAML config via the command line using the `--override` flag:
```bash
python train_acmpc.py \
  --config configs/train_acmpc_fixed_map.yaml \
  --override ppo.total_timesteps=1000000 \
  --override mpc_horizon=5 \
  --override vec_env.n_envs=16
```

### Backends
Configure the MPC solver in the YAML or via override:
- `mpc_backend: fast` (Recommended, requires CUDA)
- `mpc_backend: diffmpc` (Standard PyTorch iLQR)
- `mpc_backend: pytorch` (Legacy `mpc.pytorch`)

### Evaluation
After training, evaluate the best model and generate plots/videos:
```bash
python utils/evaluate_acmpc2.py \
  --model-path runs/acmpc_h_2/zurich/model.zip \
  --log-dir runs/acmpc_h_2/zurich \
  --track-config envs/tracks/zurich.yaml \
  --policy-type acmpc_diffmpc \
  --mpc-backend fast
```

## Adding New Tracks
Tracks are defined in `envs/tracks/*.yaml`. You can create a new track by specifying the gates:
```yaml
track_id: my_new_track
start_pos: [0.0, 0.0, 1.0, 0.0]
gate_1: [5.0, 0.0, 1.0, 0.0]
gate_2: [10.0, 5.0, 1.0, 90.0]
```
Then use it with `--track-config envs/tracks/my_new_track.yaml`.

## Acknowledgements

We would like to express our gratitude to the authors of the following works, which served as the foundation and inspiration for this project:

- **Actor-Critic Model Predictive Control**: [Romero et al.](https://github.com/uzh-rpg/acmpc_public) for the original AC-MPC architecture and its application to agile flight.
- **mpc.pytorch**: [Amos et al.](https://github.com/locuslab/mpc.pytorch) for the pioneering work on differentiable MPC layers.

## License
Refer to the `LICENSE` files in the respective subdirectories for licensing information on external dependencies.
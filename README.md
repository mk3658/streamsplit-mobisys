# StreamSplit: Continuous Audio Representation Learning via Uncertainty-Guided Adaptive Splitting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reference implementation accompanying:

> Minh K. Quan and Pubudu N. Pathirana. **StreamSplit: Continuous Audio
> Representation Learning via Uncertainty-Guided Adaptive Splitting.**
> *The 24th ACM International Conference on Mobile Systems, Applications and
> Services (MobiSys '26)*, June 21-25, 2026, Cambridge, United Kingdom.

## Overview

StreamSplit makes streaming contrastive learning practical on heterogeneous
edge devices by resolving two conflicts:

1. **The Stream-Clip Mismatch (Sec. 4.1)** -- a **distribution-based
   streaming framework** replaces explicit negative-sample storage with a
   compact Gaussian Mixture Model (`<35KB`), synthesizing *virtual*
   negatives via boundary-aware sampling so small on-device batches don't
   degrade representation quality.
2. **The Volatility Conflict (Sec. 4.2)** -- an **Uncertainty-Guided
   Adaptive Splitter** uses a lightweight PPO policy to dynamically choose
   the edge/server split point from real-time CPU load, network bandwidth,
   and the GMM's embedding-uncertainty signal (a zero-cost byproduct of
   step 1).

A **Cloud Refiner (Sec. 4.3)** completes the loop: it maintains a sliding
Temporal Buffer of received embeddings and applies a Hybrid Loss (Diversity
via Sliced-Wasserstein + Affinity via Laplacian regularization) to keep the
global manifold smooth and well-spread despite asynchronous, sometimes
sparse updates from the edge.

## Architecture

This repo mirrors the paper's three coupled subsystems (Figure 1):

```
Edge Learner (edge/)              Control Plane (edge/rl_splitting.py,    Cloud Refiner (server/)
  audio_processing.py               edge/resource_monitor.py)               temporal_buffer.py
  distributional_memory.py        PPO agent: s_t=[U_t, R_cpu, B_net]        hybrid_loss.py
  contrastive_learning.py         -> split layer k in {0,...,L}            refiner.py
        |                                    |                                  |
        +------ embeddings/uncertainty ------+------ split features -----------+
                                                          encoder: models/resnet1d.py
```

## Paper -> Code Mapping

| Paper reference | Code |
|---|---|
| Eq. 7: GMM distributional memory | `edge/distributional_memory.py::DistributionalMemory` |
| Eq. 9: boundary-aware virtual negative sampling | `DistributionalMemory.sample_virtual_negatives` |
| Eq. 10: streaming InfoNCE with virtual negatives | `edge/contrastive_learning.py::StreamingContrastiveLearning` |
| Eq. 11: embedding uncertainty U_t | `DistributionalMemory.entropy` |
| Table 8: Control Plane MDP (state/action/reward) | `edge/rl_splitting.py::SimulatedEdgeCloudEnv` |
| Eq. 12: reward function | `SimulatedEdgeCloudEnv.step` |
| Sec. 4.3.1: Temporal Buffer, temporal k-NN graph | `server/temporal_buffer.py::TemporalBuffer` |
| Eq. 1/3: Sliced-Wasserstein Diversity | `server/hybrid_loss.py::SlicedWassersteinDistance` |
| Eq. 4/6/14: Laplacian Affinity | `server/hybrid_loss.py::LaplacianRegularization` |
| Eq. 13: Hybrid Loss | `server/hybrid_loss.py::HybridLoss` |
| Sec. 4.3.3: Lazy Synchronization | `server/refiner.py::LazySync` |
| Sec. 5: ResNet-18-1D, L=8 splittable blocks | `models/resnet1d.py::AudioResNet18` |

## Installation

```bash
git clone https://github.com/mk3658/StreamSplit-AAAI.git
cd StreamSplit-AAAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Prepare datasets
python scripts/download_audioset.py
python scripts/prepare_edge_data.py

# 2. Verify the installation
python demo.py        # Edge Learner + Cloud Refiner components
python demo_rl.py      # Control Plane (RL) components

# 3. Train the Edge Learner + Cloud Refiner (Phase 1 + 3)
python train.py --config configs/streamsplit.yaml

# 4. Train the Control Plane's PPO split policy (Phase 2)
python train_rl.py --config configs/streamsplit.yaml

# 5. Run the test suite
pytest tests/ -v
```

## Project Structure

```
StreamSplit-AAAI/
├── edge/                          # Edge Learner + Control Plane (Sec. 4.1-4.2)
│   ├── audio_processing.py        # Mel-spectrogram extraction + augmentation
│   ├── distributional_memory.py   # GMM + boundary-aware virtual negatives (Eq. 7-9, 11)
│   ├── contrastive_learning.py    # Streaming InfoNCE (Eq. 10)
│   ├── resource_monitor.py        # CPU + bandwidth EMA (R_cpu, B_net)
│   └── rl_splitting.py            # PPO Control Plane (Table 8, Eq. 12)
├── server/                        # Cloud Refiner (Sec. 4.3)
│   ├── temporal_buffer.py         # Sliding window + temporal k-NN graph
│   ├── hybrid_loss.py             # Sliced-Wasserstein + Laplacian (Eq. 13)
│   └── refiner.py                 # ServerRefiner + Lazy Synchronization
├── models/
│   └── resnet1d.py                # AudioResNet18, L=8 splittable blocks
├── datasets/
│   ├── audioset.py
│   └── edge_audio.py
├── train.py                       # Phase 1 + 3 training
├── train_rl.py                    # Phase 2 (Control Plane) training
├── demo.py / demo_rl.py           # Component smoke tests
├── tests/                         # Unit tests
└── configs/streamsplit.yaml       # Hyperparameters (annotated with paper section/eq. refs)
```

## Configuration

`configs/streamsplit.yaml` is annotated inline with the paper section or
equation each value comes from. Key defaults:

```yaml
encoder:
  num_blocks: 8        # L
  embedding_dim: 128    # d

edge:
  distributional_memory:
    num_components: 64  # C
    tau: 0.1             # boundary-aware sampling temperature (Eq. 9)
  contrastive:
    n_syn: 256           # virtual negatives per anchor

control_plane:
  reward:
    alpha: 10
    beta: 5
    eta: 3

hybrid_loss:
  num_projections: 50   # M
  lambda_sw: 0.1
  lambda_lap: 0.01
  window: 100             # W
  k_neighbors: 5
```

## Scope & Known Simplifications

This is a single-machine reference implementation intended for studying and
extending the algorithms, not a byte-for-byte reproduction of the paper's
hardware deployment:

- **Network/CPU conditions are emulated in Python**, not via Raspberry Pi +
  Linux `tc` hardware-in-the-loop as in the paper's evaluation (Sec. 6). The
  Control Plane's PPO agent (`edge/rl_splitting.py::SimulatedEdgeCloudEnv`)
  trains against a lightweight simulator that reproduces the MDP's
  structure and qualitative trends, not the paper's measured latency/energy
  traces.
- **`train.py` runs Phase 1 and Phase 3 in one process**: embeddings reach
  the Cloud Refiner's Temporal Buffer detached from the encoder's
  computation graph, so the reported Hybrid Loss is a diagnostic of
  representation quality (Diversity/Affinity drift), not a second
  backpropagation pass through the encoder.
- **Table 8's temporal graph edge weights** (`W_ij`) are left uniform
  (1.0); Definition 2 (Sec. 3.2) does not specify a weighting kernel for
  the temporal adjacency graph beyond "connect temporally adjacent frames."
- **Accuracy/latency/energy numbers in Tables 2-7 are the paper's reported
  hardware results** (Raspberry Pi 4B / Apple M2 / cloud GPU server) and
  have not been re-verified by this codebase.

## Hardware Requirements (for the paper's evaluation, not this repo's defaults)

- **Edge**: Raspberry Pi 4B (4GB RAM) or equivalent ARM device
- **Server**: GPU optional; the reference implementation here runs on CPU
- **Tested on**: macOS (Apple Silicon), Linux

## Citation

```bibtex
@inproceedings{quan2026streamsplit,
  title={StreamSplit: Continuous Audio Representation Learning via Uncertainty-Guided Adaptive Splitting},
  author={Quan, Minh K. and Pathirana, Pubudu N.},
  booktitle={Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services (MobiSys '26)},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

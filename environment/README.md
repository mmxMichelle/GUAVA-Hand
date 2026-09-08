# Environment Setup

GUAVA-Hand follows the software structure of the upstream GUAVA implementation.

The formal dissertation experiments were executed in an Apptainer environment
based on the PyTorch 24.02 container. The exact core runtime versions observed
during the experiments are recorded in `versions.txt`.

## Recorded formal runtime

The formal experiment environment used:

- Python 3.10.12
- PyTorch 2.3.0a0+ebedce2
- CUDA 12.3
- cuDNN 9.0
- NumPy 1.24.4
- torchvision 0.18.0a0
- OpenCV 4.7.0
- PyTorch3D 0.7.7

These versions document the environment used for the dissertation experiments.
They are not intended to imply that every dependency must use exactly the same
build on another machine.

## Recommended setup

A Python 3.10 environment is recommended.

```bash
conda create -n guava-hand python=3.10
conda activate guava-hand
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch3D:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
```

Install the Gaussian-rendering dependencies provided with the repository:

```bash
cd submodules

pip install ./diff-gaussian-rasterization-32
pip install ./simple-knn
pip install ./fused-ssim

cd ..
```

The repository also contains project-level or vendored dependencies inherited
from GUAVA. Some modules are therefore imported from the repository rather than
from independently installed top-level Python packages.

## EHM-Tracker

Clone the repository recursively so that the pinned EHM-Tracker submodule is
available:

```bash
git clone --recursive https://github.com/mmxMichelle/GUAVA-Hand.git
cd GUAVA-Hand
```

If the repository was cloned without `--recursive`, initialise the submodule
with:

```bash
git submodule update --init --recursive
```

## External model assets

GUAVA-Hand does not redistribute restricted model assets.

Obtain the required SMPL-X, MANO, and FLAME assets independently from their
official sources and place them according to the paths described in the main
README.

## Dataset

The How2Sign dataset is not redistributed.

Users must obtain How2Sign independently and preprocess it into the
GUAVA/EHM-Tracker representation before training or evaluation.

Frozen experimental cohort identifiers and training-split metadata are
available under:

```text
protocols/
```

## Formal experiment reproduction

Environment setup alone is not sufficient to reproduce the dissertation
experiments.

The complete matched-ablation and evaluation protocol is documented in:

```text
docs/REPRODUCIBILITY.md
```

That document specifies the 160k-to-200k continuation setup, the four formal
ablation configurations, and the frozen self-, cross-, and efficiency-evaluation
protocols.

# GUAVA-Hand

**Hand-aware Gaussian avatar reconstruction for sign-language videos**

GUAVA-Hand is a research implementation developed as part of an MSc dissertation on improving hand reconstruction in upper-body 3D Gaussian avatars for sign language.

This repository builds upon **GUAVA: Generalizable Upper Body 3D Gaussian Avatar** by Zhang et al. and introduces two hand-focused extensions while retaining the original GUAVA reconstruction framework.

> **Note**
>
> This is **not** the official GUAVA repository.
> The original GUAVA implementation is maintained by `Pixel-Talk/GUAVA`.

---

## Overview

GUAVA provides feed-forward reconstruction of animatable upper-body 3D Gaussian avatars from tracked human observations.

GUAVA-Hand focuses specifically on the reconstruction of hands, which are particularly important for sign-language representation and are challenging because of their small image footprint, articulated motion, and frequent self-occlusion.

The dissertation studies two extensions:

### C1 — Hand-Aware Gaussian Representation

C1 introduces hand-aware Gaussian refinement using semantic ownership derived from the canonical SMPL-X surface.

Each valid UV-associated Gaussian is associated with a canonical SMPL-X triangle and barycentric coordinates within that triangle. Hand ownership is obtained from the corresponding SMPL-X vertex semantics and is used to gate a hand-specific residual refinement branch.

The branch is designed to increase modelling capacity around hand regions while preserving the original representation elsewhere.

### C2 — Enhanced Hand Supervision

The original GUAVA hand-crop supervision uses L1 and LPIPS losses.

C2 adds an SSIM-based hand-region term:

```text
L_hand = lambda_h * (
    lambda_1 * L1
    + lambda_p * LPIPS
    + lambda_s * L_SSIM
)

```

For the formal continuation experiments:

```text
lambda_h = 0.1
lambda_1 = 1.0
lambda_p = 0.05
```

and:

```text
lambda_s = 0.0   # C2 disabled
lambda_s = 0.1   # C2 enabled
```

The perceptual coefficient is 0.05 throughout the formal 160k-to-200k continuation because the global training iteration is restored from the GUAVA checkpoint and is already beyond the original 10k perceptual-loss transition.

### Full Model

The Full GUAVA-Hand model combines C1 and C2.

---

## Repository Structure

```text
GUAVA-Hand/
├── configs/                   Training configurations
├── dataset/                   Dataset loading utilities
├── main/
│   ├── train.py               Training entry point
│   ├── trainer.py             Training and validation loop
│   ├── test.py                Reconstruction / reenactment
│   ├── metrics_handcentric.py Hand-region quality evaluation
│   ├── metrics_cross.py       Cross-reenactment identity evaluation
│   ├── metrics_temporal.py    Temporal consistency evaluation
│   ├── benchmark_efficiency.py
│   ├── paired_bootstrap_handcentric.py
│   └── paired_bootstrap_temporal.py
├── models/                    GUAVA-Hand model implementation
├── utils/                     Losses and utility functions
├── submodules/                Rendering / metric dependencies
├── EHM-Tracker/               Upstream tracking submodule
├── assets/                    User-provided model assets
├── requirements.txt
└── LICENSE
```

---

## Upstream GUAVA

This implementation is derived from the official GUAVA codebase:

**GUAVA: Generalizable Upper Body 3D Gaussian Avatar**

Dongbin Zhang, Yunfei Liu, Lijian Lin, Ye Zhu, Yang Li, Minghan Qin, Yu Li, and Haoqian Wang.

Upstream repository:

```text
Pixel-Talk/GUAVA
```

The EHM tracking component is provided by:

```text
Pixel-Talk/EHM-Tracker
```

The EHM-Tracker submodule in this release is pinned to the version used by the dissertation experiments.

GUAVA is distributed under the Apache License 2.0. The upstream license and copyright notices are retained in this derivative implementation. Third-party components retain their respective licenses.

---

## Installation

The implementation follows the original GUAVA software stack.

### Clone

Clone the repository together with the EHM-Tracker submodule:

```bash
git clone --recursive https://github.com/mmxMichelle/GUAVA-Hand.git
cd GUAVA-Hand
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

### Python Environment

Python 3.10 is recommended.

```bash
conda create -n guava-hand python=3.10
conda activate guava-hand

pip install -r requirements.txt
```

Install PyTorch3D following the version compatible with your PyTorch and CUDA environment.

Install the local rendering dependencies:

```bash
cd submodules

pip install diff-gaussian-rasterization-32
pip install simple-knn
pip install fused-ssim

cd ..
```

---

## Required Model Assets

Restricted parametric-model assets are **not redistributed** in this repository.

The implementation expects:

```text
assets/
├── SMPLX/
├── MANO/
└── FLAME/
```

Please obtain SMPL-X, MANO, and FLAME from their respective official sources and follow the upstream GUAVA setup instructions for the expected filenames and directory structure.

The repository also does not redistribute:

- GUAVA pretrained checkpoints
- How2Sign images or videos
- processed training datasets
- tracking outputs
- experiment checkpoints
- rendered evaluation outputs

---

## Dataset

The dissertation experiments use **How2Sign** as the underlying sign-language video dataset.

How2Sign data are first processed into the representation required by the GUAVA/EHM-Tracker pipeline.

The formal training experiments use a frozen subset containing **100,002 eligible training frames from eight How2Sign training signers**. The same frozen training subset is used across the matched dissertation experiments.

Dataset files are not included in this repository.

Before training, update:

```yaml
DATASET:
    data_path: '/path/to/processed/how2sign'
```

in the relevant configuration file.

Please cite the original How2Sign publication when using the dataset.

---

## Training

The formal dissertation experiments use continuation training from the upstream GUAVA checkpoint at global iteration 160k to the predefined global iteration 200k endpoint.

For example, C1 can be trained with:

```bash
python main/train.py \
    -c train/c1_only_formal \
    -d '0,1' \
    --basemodel /path/to/guava_160k_checkpoint.pt
```

The Full model can be trained with:

```bash
python main/train.py \
    -c train/full_method_formal \
    -d '0,1' \
    --basemodel /path/to/guava_160k_checkpoint.pt
```

The optimizer and learning-rate scheduler are freshly constructed when continuation training begins, while the model and renderer weights and the global training iteration are restored from the supplied base checkpoint.

The matched dissertation comparisons use the predefined 200k endpoint rather than selecting a method-specific validation-best checkpoint.

---

## Validation

The original GUAVA validation split is retained.

During training, validation is performed periodically and validation SSIM is used for best-checkpoint tracking.

Validation is **not** used for early stopping or learning-rate adaptation.

For the matched dissertation ablation study, validation-best checkpoints are not used for formal model selection; the compared continuation models are evaluated at the same predefined 200k endpoint.

---

## Evaluation

The repository contains the evaluation code used for the dissertation experiments.

### Hand-Centric Reconstruction Quality

`main/metrics_handcentric.py`

Reports reconstruction quality for:

- full image
- head region
- left hand
- right hand
- combined hands

using:

- PSNR
- L1
- SSIM
- LPIPS

Example:

```bash
python main/metrics_handcentric.py \
    --model_path /path/to/model \
    --data_path /path/to/processed/data \
    --scene_dir /path/to/rendered/sequence \
    --output_dir /path/to/results
```

### Cross-Reenactment Identity Preservation

`main/metrics_cross.py`

Cross-reenactment identity preservation is evaluated using ArcFace-based identity similarity.

The InsightFace model location is provided explicitly:

```bash
python main/metrics_cross.py \
    --pairs_file /path/to/cross_pairs.tsv \
    --cross_root /path/to/cross/renders \
    --output_root /path/to/results \
    --pair_id <PAIR_ID> \
    --insightface_root /path/to/insightface_root
```

Aggregate previously computed pair results with:

```bash
python main/metrics_cross.py \
    --pairs_file /path/to/cross_pairs.tsv \
    --output_root /path/to/results \
    --aggregate
```

### Temporal Consistency

`main/metrics_temporal.py`

Temporal consistency is evaluated from adjacent rendered frames using normalized DINOv2 features.

Run:

```bash
python main/metrics_temporal.py --help
```

for the available evaluation and aggregation options.

### Statistical Evaluation

Paired sequence-level bootstrap utilities are provided in:

```text
main/paired_bootstrap_handcentric.py
main/paired_bootstrap_temporal.py
```

The formal dissertation analysis uses 20,000 bootstrap replicates with seed 10 and 95% percentile confidence intervals.

### Computational Efficiency

`main/benchmark_efficiency.py`

The benchmark reports model size and computational-efficiency measurements including reconstruction latency, rendering throughput, and GPU memory usage.

Run:

```bash
python main/benchmark_efficiency.py --help
```

for benchmark options.

---

## Reproducibility

The dissertation evaluation protocol uses fixed test cohorts and identical workloads across compared methods.

Key design principles include:

- the same frozen training subset for matched continuation experiments;
- a fixed self-reenactment test cohort;
- a fixed cross-reenactment pair manifest;
- predefined 200k checkpoints for matched ablation comparison;
- shared tracking-derived regional annotations;
- fixed sequence-level bootstrap seeds;
- identical sequence subsets for computational-efficiency comparison.

Large datasets, checkpoints, renders, and tracking outputs are intentionally excluded from the repository.

---

## Relationship to the Dissertation Ablation Study

The dissertation considers four matched conditions:

```text
Baseline-FT    original GUAVA architecture and supervision
C1             hand-aware Gaussian representation
C2             enhanced hand-region supervision
Full           C1 + C2
```

All matched comparisons use the same continuation-training endpoint and frozen evaluation protocol.

This repository contains the GUAVA-Hand implementation and the evaluation utilities used for that study. Formal configurations are provided only where they correspond to archived experiment configurations rather than reconstructed or inferred settings.

---

## Citation

If you use this implementation, please cite the original GUAVA work:

```bibtex
@article{GUAVA,
  title={GUAVA: Generalizable Upper Body 3D Gaussian Avatar},
  author={Zhang, Dongbin and Liu, Yunfei and Lin, Lijian and Zhu, Ye and Li, Yang and Qin, Minghan and Li, Yu and Wang, Haoqian},
  journal={arXiv preprint arXiv:2505.03351},
  year={2025}
}
```

Please also cite the original **How2Sign** publication when using How2Sign data.

A citation for the associated MSc dissertation can be added here once the dissertation repository / final record is available.

---

## Acknowledgements

This work builds upon GUAVA and its associated EHM-Tracker pipeline.

The implementation also relies on components from the 3D Gaussian Splatting ecosystem and on SMPL-X, MANO, FLAME, PyTorch3D, InsightFace, LPIPS, and related open-source libraries.

Please refer to the licenses and citation requirements of the corresponding upstream projects.

---

## License

This repository retains the Apache License 2.0 used by the upstream GUAVA implementation.

See `LICENSE` for details.

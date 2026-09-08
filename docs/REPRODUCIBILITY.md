# GUAVA-Hand Reproducibility Guide

This document describes the reproducible training and evaluation pipeline
used for the GUAVA-Hand dissertation experiments.

The commands below are intended to be copy-pasted after the user has obtained
the required external assets and prepared How2Sign in GUAVA/EHM-Tracker format.

## 1. Reproducibility Scope

The public GUAVA-Hand main branch contains the implementation used for:

- C1: hand-aware Gaussian representation;
- Full: C1 + C2;
- self-reenactment rendering;
- hand-centric reconstruction evaluation;
- cross-reenactment rendering and identity evaluation;
- temporal consistency evaluation;
- paired bootstrap analysis;
- computational-efficiency evaluation.

The matched dissertation study also contains:

- Baseline-FT: upstream GUAVA architecture and loss, continued from 160k to 200k;
- C2-only: baseline GUAVA architecture with C2 supervision enabled and C1 absent.

Baseline-FT must be reproduced with the upstream GUAVA architecture.

The formal C2-only condition was executed with a separately preserved
baseline-architecture + C2 code path. Do not run the C1-enabled GUAVA-Hand
main branch and label that run "C2-only".

The exact numerical dissertation results additionally require the exact
frozen training split, test manifests, checkpoints, and evaluation protocol.

---

## 2. Required External Resources

These files are intentionally not redistributed in this repository.

Required for GUAVA-Hand training and inference:

1. How2Sign data.
2. GUAVA/EHM-Tracker processed tracking data.
3. SMPL-X model assets.
4. MANO model assets.
5. FLAME model assets.
6. Upstream GUAVA pretrained checkpoint `best_160000.pt`.
7. EHM-Tracker pretrained tracking assets.

Additional evaluation dependencies:

8. DINOv2 ViT-B/14 pretrained weights for temporal consistency.
9. InsightFace / ArcFace `buffalo_l` assets for cross-reenactment identity
   evaluation.

Follow the licenses and access conditions of each upstream resource.

Expected model-asset locations for this repository are:

```text
assets/
├── GUAVA/
│   └── checkpoints/
│       └── best_160000.pt
├── SMPLX/
├── MANO/
└── FLAME/
```

The base checkpoint may also be stored elsewhere and supplied explicitly with
`--basemodel`.

The exact upstream checkpoint used in the formal dissertation experiments is:

```text
filename:
best_160000.pt

SHA256:
c2ffe92cc01314eb75a48b98a21c07d0636b22e40d1e7108fee7444caf3bc6a1

global_iter:
160000

background:
0
```

---

## 3. Clone and Install

Clone GUAVA-Hand with the pinned EHM-Tracker submodule:

```bash
git clone --recursive <GUAVA-HAND-REPOSITORY>
cd GUAVA-Hand
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Create the Python environment:

```bash
conda create -n guava-hand python=3.10 -y
conda activate guava-hand

python -m pip install -r requirements.txt
```

Install a PyTorch3D build compatible with the local PyTorch/CUDA environment.

Install the bundled CUDA/rendering extensions:

```bash
cd submodules

python -m pip install ./diff-gaussian-rasterization-32
python -m pip install ./simple-knn
python -m pip install ./fused-ssim

cd ..
```

The dissertation experiments were executed on NVIDIA GH200 hardware.
Runtime measurements should only be compared under a controlled
hardware/software environment.

---

## 4. Prepare External Paths

From the GUAVA-Hand repository root, define absolute local paths:

```bash
export REPO="$(pwd)"

export TRAIN_DATA="/absolute/path/to/processed/how2sign/train"
export TEST_TRACKED_ROOT="/absolute/path/to/processed/how2sign/test"

export GUAVA_160K="/absolute/path/to/best_160000.pt"

export INSIGHTFACE_ROOT="/absolute/path/to/insightface_root"
```

`TRAIN_DATA` must be a GUAVA-format training dataset containing the frozen
`dataset_frames.json`.

`TEST_TRACKED_ROOT` must contain one processed directory for every video ID in:

```text
protocols/self_test_178.txt
```

For example:

```text
${TEST_TRACKED_ROOT}/-fZc293MpJk-1-rgb_front/
${TEST_TRACKED_ROOT}/-g0iPSnQt6w-1-rgb_front/
...
```

---

## 5. Verify the Frozen Protocol

### 5.1 Verify upstream GUAVA checkpoint

```bash
EXPECTED_GUAVA_SHA256="c2ffe92cc01314eb75a48b98a21c07d0636b22e40d1e7108fee7444caf3bc6a1"

ACTUAL_GUAVA_SHA256="$(
    sha256sum "$GUAVA_160K" | awk '{print $1}'
)"

echo "expected = $EXPECTED_GUAVA_SHA256"
echo "actual   = $ACTUAL_GUAVA_SHA256"

test "$ACTUAL_GUAVA_SHA256" = "$EXPECTED_GUAVA_SHA256"
```

Verify checkpoint metadata:

```bash
python - "$GUAVA_160K" <<'PY'
import sys
import torch

path = sys.argv[1]

state = torch.load(
    path,
    map_location="cpu",
    weights_only=True,
)

assert int(state["global_iter"]) == 160000

print("[PASS] GUAVA checkpoint")
print("global_iter =", state["global_iter"])

if "meta_cfg" in state:
    try:
        print(
            "bg_color =",
            state["meta_cfg"]["MODEL"]["bg_color"],
        )
    except Exception:
        print("bg_color metadata not available in checkpoint")
PY
```

### 5.2 Verify frozen training split

The exact formal training split contains:

```text
train = 100002
valid = 2235
```

with SHA256:

```text
d577aa3771189947c473e6909dfbffaa6053b3af23da3efbdc44dcb3266a63b0
```

Verify an existing copy:

```bash
EXPECTED_TRAIN_SHA256="d577aa3771189947c473e6909dfbffaa6053b3af23da3efbdc44dcb3266a63b0"

ACTUAL_TRAIN_SHA256="$(
    sha256sum "$TRAIN_DATA/dataset_frames.json" | awk '{print $1}'
)"

echo "expected = $EXPECTED_TRAIN_SHA256"
echo "actual   = $ACTUAL_TRAIN_SHA256"

test "$ACTUAL_TRAIN_SHA256" = "$EXPECTED_TRAIN_SHA256"
```

Verify counts:

```bash
python - "$TRAIN_DATA/dataset_frames.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

assert len(data["train"]) == 100002
assert len(data["valid"]) == 2235

print("[PASS] frozen training manifest")
print("train =", len(data["train"]))
print("valid =", len(data["valid"]))
PY
```

Additional metadata are stored in:

```text
protocols/train_subset_metadata.json
```

Important: the SHA256 verifies an existing frozen manifest; it does not by
itself reconstruct the list of 100,002 frame identifiers.

### 5.3 Verify self-reenactment cohort

```bash
test "$(wc -l < protocols/self_test_178.txt)" -eq 178
test "$(sort -u protocols/self_test_178.txt | wc -l)" -eq 178

echo "[PASS] 178-sequence self protocol"
```

### 5.4 Verify cross-reenactment protocol

The file contains one header plus 80 source-driver pairs:

```bash
test "$(wc -l < protocols/cross_pairs_80.tsv)" -eq 81
test "$(tail -n +2 protocols/cross_pairs_80.tsv | wc -l)" -eq 80

echo "[PASS] 80-pair cross protocol"
```

---

## 6. Configure the Local Training Dataset

Create local untracked copies of the archived formal configurations:

```bash
printf '%s\n' \
    'configs/train/c1_only_repro.yaml' \
    'configs/train/full_method_repro.yaml' \
    >> .git/info/exclude

cp \
    configs/train/c1_only_formal.yaml \
    configs/train/c1_only_repro.yaml

cp \
    configs/train/full_method_formal.yaml \
    configs/train/full_method_repro.yaml
```

Replace the public dataset placeholder with the local processed dataset:

```bash
sed -i \
    "s|/path/to/processed/how2sign|$TRAIN_DATA|g" \
    configs/train/c1_only_repro.yaml

sed -i \
    "s|/path/to/processed/how2sign|$TRAIN_DATA|g" \
    configs/train/full_method_repro.yaml
```

Verify the relevant settings:

```bash
grep -E \
'bg_color|batch_size|train_iter|check_interval|lambda_hand_crop|lambda_hand_ssim|data_path' \
configs/train/c1_only_repro.yaml

echo

grep -E \
'bg_color|batch_size|train_iter|check_interval|lambda_hand_crop|lambda_hand_ssim|data_path' \
configs/train/full_method_repro.yaml
```

The matched continuation protocol uses:

```text
initial global iteration = 160000
target global iteration  = 200000
batch size               = 6
training seed             = 10
background                = 0
hand crop weight          = 0.1

C1:
lambda_hand_ssim = 0.0

Full:
lambda_hand_ssim = 0.1
```

The optimizer and learning-rate scheduler are newly initialized when
continuation training starts. Model weights, renderer weights, and the global
iteration are restored from the 160k checkpoint.

GUAVA's perceptual coefficient changes from 0.025 to 0.05 after global
iteration 10,000. Since continuation begins from global iteration 160,000,
the effective perceptual coefficient is 0.05 throughout the formal
160k-to-200k continuation.

---

## 7. Train C1

Run:

```bash
python main/train.py \
    -c train/c1_only_repro \
    -d '0' \
    --basemodel "$GUAVA_160K"
```

The formal comparison uses the checkpoint at the predefined global
iteration 200,000 endpoint.

After training, identify the corresponding run directory and checkpoint:

```bash
export C1_RUN="/absolute/path/to/c1/training/run"
export C1_CKPT="$C1_RUN/checkpoints/latest.pt"

test -f "$C1_CKPT"
```

Verify the endpoint:

```bash
python - "$C1_CKPT" <<'PY'
import sys
import torch

state = torch.load(
    sys.argv[1],
    map_location="cpu",
    weights_only=True,
)

print("global_iter =", state["global_iter"])

assert int(state["global_iter"]) == 200000

print("[PASS] C1 checkpoint is global iteration 200000")
PY
```

---

## 8. Train Full GUAVA-Hand

Run:

```bash
python main/train.py \
    -c train/full_method_repro \
    -d '0' \
    --basemodel "$GUAVA_160K"
```

After training, identify the corresponding run directory and checkpoint:

```bash
export FULL_RUN="/absolute/path/to/full/training/run"
export FULL_CKPT="$FULL_RUN/checkpoints/latest.pt"

test -f "$FULL_CKPT"
```

Verify the endpoint:

```bash
python - "$FULL_CKPT" <<'PY'
import sys
import torch

state = torch.load(
    sys.argv[1],
    map_location="cpu",
    weights_only=True,
)

print("global_iter =", state["global_iter"])

assert int(state["global_iter"]) == 200000

print("[PASS] Full checkpoint is global iteration 200000")
PY
```

Formal comparison uses `latest.pt` at global iteration 200k rather than
selecting a method-specific validation-best checkpoint.

---

## 9. Baseline-FT and C2-Only

The complete matched dissertation study contains four 200k conditions:

```text
Baseline-FT    C1 OFF    C2 OFF
C1             C1 ON     C2 OFF
C2             C1 OFF    C2 ON
Full           C1 ON     C2 ON
```

Baseline-FT uses the upstream GUAVA baseline architecture and original
supervision while following the same 160k-to-200k continuation protocol.

The formal C2-only condition uses the baseline GUAVA architecture with the
additional C2 hand-region SSIM supervision and no C1 representation branch.

The GUAVA-Hand main branch contains the C1/Full implementation and must not be
used to create a run labelled C2-only simply by changing the SSIM coefficient.

Exact reproduction of the C2-only formal condition therefore requires the
preserved baseline-architecture + C2 implementation used for that experiment.

---

## 10. Self-Reenactment Rendering

Create an output root:

```bash
export SELF_RENDER_ROOT="$REPO/outputs/reproduction/full/self"

mkdir -p "$SELF_RENDER_ROOT"
```

The frozen self-evaluation cohort is listed in:

```text
protocols/self_test_178.txt
```

Each entry corresponds to one processed How2Sign TEST sequence.

For one sequence:

```bash
export VIDEO_ID="$(head -1 protocols/self_test_178.txt)"
export DATA_PATH="$TEST_TRACKED_ROOT/$VIDEO_ID"

python main/test.py \
    -c train/full_method_repro \
    -d '0' \
    -b "$FULL_CKPT" \
    --data_path "$DATA_PATH" \
    -m "$FULL_RUN" \
    -s "$SELF_RENDER_ROOT" \
    -n render
```

For the formal evaluation, repeat the identical rendering procedure for all
178 entries in `protocols/self_test_178.txt`.

---

## 11. Hand-Centric Reconstruction Evaluation

The evaluator is:

```text
main/metrics_handcentric.py
```

It reports:

```text
Full image
Head
Left Hand
Right Hand
Combined Hands
```

using:

```text
PSNR
L1
SSIM
LPIPS
```

The regional crops reuse the GUAVA tracking-derived region boxes.
The same tracked box is used to crop the rendered image and corresponding
ground-truth image.

Define an output root:

```bash
export HAND_ROOT="$REPO/outputs/reproduction/full/handcentric"

mkdir -p "$HAND_ROOT/per_video"
```

For a single sequence, inspect the evaluator interface with:

```bash
python main/metrics_handcentric.py --help
```

A typical invocation is:

```bash
python main/metrics_handcentric.py \
    --model_path "$FULL_RUN" \
    --data_path "$TEST_TRACKED_ROOT/$VIDEO_ID" \
    --scene_dir "$SELF_RENDER_ROOT/render_self_act/$VIDEO_ID" \
    --output_dir "$HAND_ROOT/per_video/$VIDEO_ID"
```

Run the same evaluator over all 178 frozen self-reenactment sequences.

Aggregate the complete cohort with:

```bash
python main/aggregate_handcentric.py \
    --manifest protocols/self_test_178.txt \
    --output_root "$HAND_ROOT"
```

Expected aggregate outputs include:

```text
${HAND_ROOT}/per_video_metrics.csv
${HAND_ROOT}/dataset_summary.json
```

The formal self-reenactment quality evaluation uses all rendered evaluation
frames from the 178-sequence cohort.

---

## 12. Temporal Consistency

The temporal evaluator is:

```text
main/metrics_temporal.py
```

It computes adjacent-frame temporal consistency using normalized frozen
DINOv2 ViT-B/14 representations.

Define the output root:

```bash
export TEMP_ROOT="$REPO/outputs/reproduction/full/temporal"

mkdir -p "$TEMP_ROOT"
```

Evaluate one sequence:

```bash
python main/metrics_temporal.py \
    --video_id "$VIDEO_ID" \
    --render_dir "$SELF_RENDER_ROOT/render_self_act/$VIDEO_ID/render" \
    --output_root "$TEMP_ROOT" \
    --batch_size 16 \
    --max_frames 0 \
    --device cuda:0
```

For the formal temporal protocol, run the evaluator for all 178 entries in:

```text
protocols/self_test_178.txt
```

With `--max_frames 0`, the evaluator does not impose an evaluator-side maximum
number of frames.

Aggregate all completed sequence results:

```bash
python main/metrics_temporal.py \
    --manifest protocols/self_test_178.txt \
    --output_root "$TEMP_ROOT" \
    --aggregate
```

Temporal consistency should not be interpreted as a standalone measure of
animation quality: unusually static or over-smoothed output can also produce
high adjacent-frame similarity.

---

## 13. Cross-Reenactment Protocol

The frozen cross protocol is stored in:

```text
protocols/cross_pairs_80.tsv
```

It contains 80 fixed source-driver pairs.

For each pair:

```text
source = identity / avatar
driver = motion
```

The formal protocol uses:

```text
keep_source_cam = false
```

Define the cross-render output root:

```bash
export CROSS_RENDER_ROOT="$REPO/outputs/reproduction/full/cross"

mkdir -p "$CROSS_RENDER_ROOT"
```

The exact local source and driver data directories are resolved from the video
IDs stored in `protocols/cross_pairs_80.tsv`.

Cross rendering uses `main/test.py` with:

```text
--render_cross_act
--source_data_path
--data_path
```

Do not enable `--keep_source_cam` when reproducing the formal dissertation
cross protocol.

Inspect the exact renderer interface with:

```bash
python main/test.py --help
```

---

## 14. Cross-Reenactment Identity Preservation

The cross evaluator is:

```text
main/metrics_cross.py
```

It uses ArcFace-based identity similarity through InsightFace.

The local InsightFace model root is supplied explicitly and is not hard-coded
in the public repository.

For one frozen pair:

```bash
python main/metrics_cross.py \
    --pairs_file protocols/cross_pairs_80.tsv \
    --cross_root "$CROSS_RENDER_ROOT/render_cross_act" \
    --output_root "$REPO/outputs/reproduction/full/cross_ips" \
    --pair_id <PAIR_ID> \
    --insightface_root "$INSIGHTFACE_ROOT"
```

After all 80 pair results have been computed, aggregate them with:

```bash
python main/metrics_cross.py \
    --pairs_file protocols/cross_pairs_80.tsv \
    --output_root "$REPO/outputs/reproduction/full/cross_ips" \
    --aggregate
```

---

## 15. Statistical Evaluation

The formal dissertation analysis uses sequence-level paired bootstrap
comparisons.

The shared settings are:

```text
bootstrap replicates = 20000
seed                 = 10
confidence interval  = 95%
```

### 15.1 Hand-centric bootstrap

The relevant utility is:

```text
main/paired_bootstrap_handcentric.py
```

Inspect its exact input interface with:

```bash
python main/paired_bootstrap_handcentric.py --help
```

The primary matched comparison is Full versus Baseline-FT at the same 200k
training endpoint.

### 15.2 Temporal bootstrap

The temporal bootstrap utility accepts the four matched 200k per-video CSVs:

```bash
python main/paired_bootstrap_temporal.py \
    --baseline_csv /path/to/baseline_ft/per_video_temporal.csv \
    --c1_csv /path/to/c1/per_video_temporal.csv \
    --c2_csv /path/to/c2/per_video_temporal.csv \
    --full_csv /path/to/full/per_video_temporal.csv \
    --output_root /path/to/bootstrap_temporal \
    --n_bootstrap 20000 \
    --seed 10 \
    --ci 95 \
    --sanity_tolerance 1e-10
```

Use the actual per-video CSV filename produced by the temporal aggregation
step.

---

## 16. Computational-Efficiency Benchmark

The evaluator is:

```text
main/benchmark_efficiency.py
```

The formal benchmark measures:

```text
exact model parameter count
source-to-ready avatar reconstruction latency
rendering throughput
peak CUDA allocated memory
```

The timed reconstruction stage excludes dataset loading, checkpoint loading,
and disk I/O.

The formal per-sequence settings are:

```text
seed = 10
reconstruction warm-up = 2
reconstruction repeats = 5
render warm-up frames = 10
render frames = up to 200
```

For one Full-model sequence:

```bash
export EFF_VIDEO_ID="$(head -1 protocols/self_test_178.txt)"
export EFF_ROOT="$REPO/outputs/reproduction/efficiency"

mkdir -p "$EFF_ROOT/per_run/full"

python main/benchmark_efficiency.py \
    --method full \
    --repo "$REPO" \
    --model_path "$FULL_RUN" \
    --checkpoint "$FULL_CKPT" \
    --data_path "$TEST_TRACKED_ROOT/$EFF_VIDEO_ID" \
    --output "$EFF_ROOT/per_run/full/${EFF_VIDEO_ID}.json" \
    --device 0 \
    --seed 10 \
    --hand_arch \
    --warmup_recon 2 \
    --recon_repeats 5 \
    --warmup_frames 10 \
    --render_frames 200
```

The formal dissertation comparison uses the same fixed 10-sequence workload
for every method.

After all methods have been evaluated on the same 10-sequence cohort,
aggregate with:

```bash
python main/benchmark_efficiency.py \
    --aggregate \
    --input_root "$EFF_ROOT/per_run" \
    --output_root "$EFF_ROOT/summary" \
    --expected_videos 10
```

The exact frozen 10-sequence cohort is released in
`protocols/efficiency_10.txt`. Exact thesis-number reproduction additionally
requires the same hardware, checkpoints, benchmark implementation, warm-up
policy, and timing boundaries.

---

## 17. Validation and Checkpoint Selection

The original GUAVA validation split is retained during continuation training.

Validation is run periodically and validation SSIM is used by the training
framework for best-checkpoint tracking.

Validation is not used for:

- early stopping;
- learning-rate adaptation;
- formal model selection in the matched dissertation comparison.

The matched Baseline-FT, C1, C2, and Full comparison uses the same predefined
global iteration 200000 endpoint.

Validation-best checkpoints may still be produced by the original training
framework, but they are not used as the formal matched-comparison checkpoints.

---

## 18. What Is and Is Not Redistributed

This repository intentionally does not contain:

```text
raw How2Sign videos or images
processed How2Sign image data
tracking outputs
SMPL-X assets
MANO assets
FLAME assets
upstream GUAVA checkpoints
trained dissertation checkpoints
self-reenactment renders
cross-reenactment renders
InsightFace model files
DINOv2 model files
```

The repository contains the implementation, evaluation utilities, frozen
non-image protocol manifests, and reproducibility metadata needed to identify
the formal experimental setup.

### Frozen training-manifest limitation

`protocols/train_subset_metadata.json` records the exact training/validation
counts, signer allocation, and SHA256 of the dissertation
`dataset_frames.json`.

The full How2Sign-derived `dataset_frames.json` is not redistributed here.

Consequently, the published hash can verify an existing frozen manifest, but
the hash alone cannot reconstruct the list of 100,002 selected frame
identifiers.

Exact frame-level reproduction therefore requires access to the original
frozen manifest or a separately verified deterministic reconstruction
procedure that produces the same SHA256.

### Ablation-code provenance

The public GUAVA-Hand main branch contains the C1/Full implementation.

Baseline-FT uses the upstream GUAVA baseline architecture and loss.

The formal C2-only condition uses a separately preserved
baseline-architecture + C2 implementation in which C1 is absent.

The C1-enabled main branch must not be relabelled as C2-only.

---

## 19. Frozen Protocol Summary

```text
Training
--------
Dataset: frozen How2Sign TRAIN subset
Train entries: 100002
Validation entries: 2235
Training seed: 10
Batch size: 6
Initialization: official GUAVA best_160000.pt
Initial global iteration: 160000
Formal endpoint: 200000
Background: 0

Self evaluation
---------------
178 fixed How2Sign TEST sequences
All rendered evaluation frames used for reconstruction quality

Cross evaluation
----------------
80 fixed source-driver pairs
Source provides identity/avatar
Driver provides motion
keep_source_cam = false

Temporal evaluation
-------------------
Frozen DINOv2 ViT-B/14 evaluator
Adjacent rendered-frame cosine similarity
178 frozen self-reenactment sequences

Bootstrap
---------
20000 sequence-level bootstrap replicates
Seed 10
95% percentile confidence intervals

Efficiency
----------
Fixed 10-sequence cohort
2 reconstruction warm-up runs
5 timed reconstruction repeats per sequence
10 render warm-up frames
Up to 200 sampled render frames per sequence
```

---

## 20. Reproducibility Boundaries

There are two different reproducibility goals.

### Protocol reproduction

The public repository provides the implementation, configurations, evaluator
code, hashes, and frozen non-image manifests required to reproduce the
experimental procedure.

### Exact numerical reproduction

Exact reproduction of the dissertation tables additionally requires the exact:

- frozen How2Sign-derived training manifest;
- processed tracked How2Sign inputs;
- GUAVA initialization checkpoint;
- final trained checkpoints;
- C2-only preserved implementation;
- external model assets;
- compatible software environment;
- comparable GPU runtime environment.

These resources are not all redistributable through this Git repository.

No numerical result should be claimed as reproduced unless it can be traced
to the corresponding frozen inputs, model checkpoint, evaluator, and protocol.

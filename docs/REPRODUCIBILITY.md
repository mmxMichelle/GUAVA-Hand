# Reproducing the GUAVA-Hand Experiments

This document describes the experimental protocol used for the matched
GUAVA-Hand ablation study.

## 1. Repository

Clone the repository including the EHM-Tracker submodule:

```bash
git clone --recursive https://github.com/mmxMichelle/GUAVA-Hand.git
cd GUAVA-Hand
```

## 2. External assets

This repository does not redistribute restricted datasets, model assets, or
trained checkpoints.

Obtain the required resources independently from their official sources,
including:

- How2Sign;
- SMPL-X;
- MANO;
- FLAME;
- the upstream GUAVA pretrained checkpoint.

Place the body-model assets in the locations described in the main repository
README.

## 3. Environment

The experiments were run in an Apptainer environment based on the PyTorch 24.02
container.

The core runtime versions observed inside the experiment container are recorded
in:

```text
environment/versions.txt
```

Project-specific and vendored dependencies inherited from GUAVA are included in
the repository where applicable.

## 4. Dataset preparation

How2Sign is first processed into the representation required by the
GUAVA/EHM-Tracker pipeline.

All matched continuation-training experiments use the same frozen training
split.

Metadata describing this split is provided in:

```text
protocols/train_subset_metadata.json
```

The frozen manifest contains:

- 100,002 training entries;
- 2,235 validation entries;
- eight training signers.

The SHA256 checksum of the frozen internal `dataset_frames.json` is recorded in
the metadata file.

Raw How2Sign media and generated tracking outputs are not redistributed.

## 5. Matched continuation training

All matched ablation runs are initialized from the same upstream GUAVA
checkpoint at global iteration 160,000.

Training continues to the predefined global iteration 200,000 endpoint.

The formal configurations used for the matched comparison are:

```text
configs/train/baseline_ft_formal.yaml
configs/train/c1_only_formal.yaml
configs/train/c2_only_formal.yaml
configs/train/full_method_formal.yaml
```

The four experimental conditions are:

| Condition | C1: hand-aware representation | C2: added hand SSIM supervision |
| --- | --- | --- |
| Baseline-FT | No | No |
| C1 | Yes | No |
| C2 | No | Yes |
| Full | Yes | Yes |

The formal runs use the same frozen training data, initialization point, and
predefined 200k endpoint.

## 6. Validation and checkpoint policy

The original GUAVA validation split is retained for periodic training-time
evaluation and best-checkpoint tracking based on validation SSIM.

Validation is not used for early stopping or learning-rate adaptation.

For the matched ablation study, Baseline-FT, C1, C2, and Full are evaluated at
the same predefined 200k training endpoint. Method-specific validation-best
checkpoints are therefore not used for formal model selection.

## 7. Self-reenactment evaluation

The frozen self-reenactment cohort is defined in:

```text
protocols/self_test_178.txt
```

It contains 178 unique sequence identifiers.

The identical cohort is used for all compared methods.

The released evaluation utilities support full-image and regional quality
evaluation, including hand-centric reconstruction metrics.

## 8. Cross-reenactment evaluation

The fixed cross-reenactment protocol is defined in:

```text
protocols/cross_pairs_80.tsv
```

It contains 80 fixed source-driver pairs.

The public version preserves:

- pair identifier;
- source signer;
- source video identifier;
- driver signer;
- driver video identifier.

Machine-specific tracking paths from the internal experiment manifest have been
removed.

The same source-driver protocol is used for all compared methods.

## 9. Temporal consistency evaluation

Temporal consistency is evaluated on the same frozen 178-sequence
self-reenactment cohort.

The released temporal evaluation utility evaluates adjacent rendered frames
according to the protocol used in the dissertation experiments.

## 10. Computational-efficiency evaluation

The fixed efficiency cohort is defined in:

```text
protocols/efficiency_10.txt
```

It contains 10 sequences corresponding to the following zero-based indices in
the frozen 178-sequence self-reenactment manifest:

```text
0, 20, 39, 59, 79, 98, 118, 138, 157, 177
```

The identical 10-sequence subset is used for all compared methods.

For each selected sequence:

- reconstruction latency is measured over repeated runs;
- rendering throughput is measured using up to 200 approximately uniformly
  sampled evaluation frames;
- peak GPU memory is measured under the same benchmark workload.

## 11. Paired statistical analysis

The repository includes paired bootstrap utilities for hand-centric and
temporal comparisons.

For the hand-centric paired analysis, the statistical unit is the sequence and
the same frozen 178-sequence cohort is used across compared methods.

## 12. Released reproducibility artifacts

This repository releases:

- the GUAVA-Hand implementation;
- the four formal ablation configurations;
- frozen self-, cross-, and efficiency-evaluation protocol identifiers;
- frozen training-split metadata and checksum;
- evaluation utilities;
- paired statistical-analysis utilities;
- environment metadata.

The repository does not redistribute:

- How2Sign media;
- generated tracking outputs;
- pretrained or trained checkpoints;
- rendered videos;
- restricted SMPL-X, MANO, or FLAME assets.

These resources must be obtained independently from their respective official
sources.

## 13. Reproducibility scope

The released materials are intended to reproduce the GUAVA-Hand method,
matched-ablation settings, and evaluation protocol while keeping restricted
datasets, model assets, checkpoints, and machine-specific paths outside the
public repository.

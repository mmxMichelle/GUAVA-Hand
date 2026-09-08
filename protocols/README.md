# Frozen Experimental Protocols

This directory contains portable metadata describing the frozen experimental
protocols used for the GUAVA-Hand dissertation experiments.

Raw How2Sign data, tracking outputs, checkpoints, and rendered videos are not
redistributed.

## Files

### `self_test_178.txt`

Fixed self-reenactment evaluation cohort containing 178 unique How2Sign
sequence identifiers.

The original experiment manifest stored absolute paths to locally generated
tracking outputs. This public version retains only the sequence identifiers so
that the evaluation cohort is preserved without exposing machine-specific
paths.

### `cross_pairs_80.tsv`

Fixed set of 80 source-driver pairs used for cross-reenactment evaluation.

The public manifest contains:

- `pair_id`
- `source_signer`
- `source_video_id`
- `driver_signer`
- `driver_video_id`

Machine-specific source and driver tracking paths from the internal experiment
manifest have been removed.

### `train_subset_metadata.json`

Metadata describing the frozen How2Sign training split used by the matched
continuation-training experiments.

The metadata records:

- number of training and validation entries;
- per-signer entry counts;
- SHA256 checksum of the frozen `dataset_frames.json`.

The full How2Sign dataset is not redistributed.

## Data access

Users should obtain How2Sign from its official source and construct the
required GUAVA/EHM tracking representation separately.

The identifiers in these manifests define the experimental cohorts used in the
dissertation but do not include the underlying media or tracking data.

### `efficiency_10.txt`

Fixed 10-sequence subset used for computational-efficiency evaluation.

The sequences correspond to zero-based indices

`0, 20, 39, 59, 79, 98, 118, 138, 157, 177`

of the frozen 178-sequence self-reenactment manifest. The same subset was used
for all compared methods.

For each selected sequence, reconstruction latency was measured repeatedly,
while rendering throughput was evaluated on up to 200 approximately uniformly
sampled frames from that sequence.

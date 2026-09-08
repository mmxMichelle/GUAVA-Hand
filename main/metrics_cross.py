#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from main.metrics_face import FaceComparator

CROSS_FACE_MODEL = "buffalo_l"
CROSS_FACE_ROOT = None
CROSS_FACE_PROVIDERS = ["CPUExecutionProvider"]
CROSS_FACE_ALLOWED_MODULES = ["detection", "recognition"]



IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_pairs(path):
    """
    cross_pairs_80.tsv columns:
      pair_id
      source_signer
      source_video_id
      source_path
      driver_signer
      driver_video_id
      driver_path

    Paths contain no whitespace, so split() is robust to tabs/spaces.
    """
    path = Path(path)

    with path.open("r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if not lines:
        raise RuntimeError(f"Empty pair manifest: {path}")

    header = lines[0].split()

    expected = [
        "pair_id",
        "source_signer",
        "source_video_id",
        "source_path",
        "driver_signer",
        "driver_video_id",
        "driver_path",
    ]

    if header != expected:
        raise RuntimeError(
            f"Unexpected manifest header.\n"
            f"Expected: {expected}\n"
            f"Actual:   {header}"
        )

    pairs = []

    for line in lines[1:]:
        fields = line.split()

        if len(fields) != len(header):
            raise RuntimeError(
                f"Malformed manifest line:\n{line}"
            )

        pairs.append(dict(zip(header, fields)))

    return pairs


def get_pair(pairs, pair_id):
    pair_id = str(pair_id).zfill(3)

    matches = [
        p for p in pairs
        if p["pair_id"] == pair_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one pair_id={pair_id}, "
            f"found {len(matches)}"
        )

    return matches[0]


def frame_sort_key(path):
    stem = path.stem

    if stem.isdigit():
        return (0, int(stem))

    return (1, stem)


def evaluate_pair(
    pair,
    cross_root,
    output_root,
    max_frames=0,
    ctx_id=0,
):
    pair_id = pair["pair_id"]

    source_id = pair["source_video_id"]
    driver_id = pair["driver_video_id"]

    cross_root = Path(cross_root)
    output_root = Path(output_root)

    pair_dir = (
        cross_root
        / "render_cross_act"
        / source_id
        / f"{source_id}_{driver_id}"
    )

    source_image = pair_dir / "source_image.png"
    render_dir = pair_dir / "render"

    if not pair_dir.is_dir():
        raise RuntimeError(
            f"Pair directory missing: {pair_dir}"
        )

    if not source_image.is_file():
        raise RuntimeError(
            f"source_image.png missing: {source_image}"
        )

    if not render_dir.is_dir():
        raise RuntimeError(
            f"Render directory missing: {render_dir}"
        )

    render_files = sorted(
        [
            p for p in render_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        ],
        key=frame_sort_key,
    )

    available_frames = len(render_files)

    if available_frames == 0:
        raise RuntimeError(
            f"No rendered frames found: {render_dir}"
        )

    if max_frames and max_frames > 0:
        render_files = render_files[:max_frames]

    print("=" * 70)
    print("GUAVA Cross Identity Evaluation")
    print("=" * 70)
    print("Pair ID          :", pair_id)
    print("Source           :", source_id)
    print("Driver           :", driver_id)
    print("Pair directory   :", pair_dir)
    print("Source image     :", source_image)
    print("Available frames :", available_frames)
    print("Evaluating       :", len(render_files))
    print("=" * 70)

    print("Face model        :", CROSS_FACE_MODEL)
    print("Face model root   :", CROSS_FACE_ROOT)
    print("Face providers    :", CROSS_FACE_PROVIDERS)
    print("Face modules      :", CROSS_FACE_ALLOWED_MODULES)

    comparator = FaceComparator(
        ctx_id=ctx_id,
        model_name=CROSS_FACE_MODEL,
        root=CROSS_FACE_ROOT,
        providers=CROSS_FACE_PROVIDERS,
        allowed_modules=CROSS_FACE_ALLOWED_MODULES,
    )

    # --------------------------------------------------------
    # Source identity embedding
    # --------------------------------------------------------

    source_feat = comparator.get_features(
        str(source_image)
    )

    if source_feat is None:
        raise RuntimeError(
            f"ArcFace failed on source image: {source_image}"
        )

    source_feat = np.asarray(
        source_feat,
        dtype=np.float64
    )

    if not np.isfinite(source_feat).all():
        raise RuntimeError(
            "Source ArcFace embedding contains NaN/Inf"
        )

    source_norm = np.linalg.norm(source_feat)

    print(
        "Source embedding norm:",
        float(source_norm)
    )

    # InsightFace normed_embedding should already be unit norm.
    if not np.isfinite(source_norm) or source_norm <= 0:
        raise RuntimeError(
            "Invalid source embedding norm"
        )

    # --------------------------------------------------------
    # Rendered frames
    # --------------------------------------------------------

    rows = []
    similarities = []

    for frame_path in tqdm(
        render_files,
        desc=f"Pair {pair_id}",
    ):
        feat = comparator.get_features(
            str(frame_path)
        )

        if feat is None:
            rows.append(
                {
                    "frame": frame_path.name,
                    "identity_similarity": "",
                    "valid": 0,
                }
            )
            continue

        feat = np.asarray(
            feat,
            dtype=np.float64
        )

        if not np.isfinite(feat).all():
            rows.append(
                {
                    "frame": frame_path.name,
                    "identity_similarity": "",
                    "valid": 0,
                }
            )
            continue

        # Both embeddings are InsightFace normed_embedding.
        # Therefore dot product = cosine similarity.
        similarity = float(
            np.dot(source_feat, feat)
        )

        if not math.isfinite(similarity):
            rows.append(
                {
                    "frame": frame_path.name,
                    "identity_similarity": "",
                    "valid": 0,
                }
            )
            continue

        similarities.append(similarity)

        rows.append(
            {
                "frame": frame_path.name,
                "identity_similarity": similarity,
                "valid": 1,
            }
        )

    total_frames = len(render_files)
    valid_frames = len(similarities)
    invalid_frames = total_frames - valid_frames

    if valid_frames == 0:
        raise RuntimeError(
            f"No valid ArcFace render embeddings "
            f"for pair {pair_id}"
        )

    sims = np.asarray(
        similarities,
        dtype=np.float64
    )

    identity_sum = float(
        sims.sum()
    )

    identity_sq_sum = float(
        np.square(sims).sum()
    )

    identity_mean = float(
        sims.mean()
    )

    identity_std = float(
        sims.std(ddof=0)
    )

    valid_rate = (
        valid_frames / total_frames
        if total_frames > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pair_output = (
        output_root
        / "per_pair"
        / pair_id
    )

    pair_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        pair_output
        / "per_frame.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "identity_similarity",
                "valid",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pair_id": pair_id,

        "source_signer":
            pair["source_signer"],

        "source_video_id":
            source_id,

        "driver_signer":
            pair["driver_signer"],

        "driver_video_id":
            driver_id,

        "source_path":
            pair["source_path"],

        "driver_path":
            pair["driver_path"],

        "pair_dir":
            str(pair_dir),

        "source_image":
            str(source_image),

        "available_render_frames":
            available_frames,

        "total_frames":
            total_frames,

        "valid_frames":
            valid_frames,

        "invalid_frames":
            invalid_frames,

        "valid_rate":
            valid_rate,

        "identity_sum":
            identity_sum,

        "identity_sq_sum":
            identity_sq_sum,

        "identity_mean":
            identity_mean,

        "identity_std":
            identity_std,
    }

    summary_path = (
        pair_output
        / "summary.json"
    )

    with summary_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("PAIR RESULT")
    print("=" * 70)

    print(
        f"Total frames   : {total_frames}"
    )

    print(
        f"Valid frames   : {valid_frames}"
    )

    print(
        f"Invalid frames : {invalid_frames}"
    )

    print(
        f"Valid rate     : {valid_rate:.6f}"
    )

    print(
        f"Identity mean  : {identity_mean:.8f}"
    )

    print(
        f"Identity std   : {identity_std:.8f}"
    )

    print(
        f"CSV            : {csv_path}"
    )

    print(
        f"Summary        : {summary_path}"
    )

    print("=" * 70)

    return summary


def aggregate(
    pairs_file,
    output_root,
):
    pairs = load_pairs(
        pairs_file
    )

    output_root = Path(
        output_root
    )

    summaries = []

    missing = []

    for pair in pairs:
        pair_id = pair["pair_id"]

        summary_path = (
            output_root
            / "per_pair"
            / pair_id
            / "summary.json"
        )

        if not summary_path.is_file():
            missing.append(
                str(summary_path)
            )
            continue

        with summary_path.open("r") as f:
            summary = json.load(f)

        summaries.append(summary)

    if missing:
        print(
            f"Missing {len(missing)} pair summaries:"
        )

        for p in missing[:20]:
            print("  ", p)

        raise RuntimeError(
            "Cross aggregation aborted because "
            "not all expected pairs are complete."
        )

    if len(summaries) != len(pairs):
        raise RuntimeError(
            f"Expected {len(pairs)} summaries, "
            f"got {len(summaries)}"
        )

    total_frames = sum(
        s["total_frames"]
        for s in summaries
    )

    valid_frames = sum(
        s["valid_frames"]
        for s in summaries
    )

    invalid_frames = sum(
        s["invalid_frames"]
        for s in summaries
    )

    total_sim = sum(
        s["identity_sum"]
        for s in summaries
    )

    total_sq_sim = sum(
        s["identity_sq_sum"]
        for s in summaries
    )

    if valid_frames == 0:
        raise RuntimeError(
            "No valid ArcFace frames "
            "across dataset."
        )

    # --------------------------------------------------------
    # Primary GUAVA-aligned result:
    # global mean over all valid rendered frames
    # --------------------------------------------------------

    global_mean = (
        total_sim / valid_frames
    )

    global_variance = (
        total_sq_sim / valid_frames
        - global_mean ** 2
    )

    global_variance = max(
        0.0,
        global_variance,
    )

    global_std = math.sqrt(
        global_variance
    )

    # Auxiliary pair-macro statistic
    pair_means = np.asarray(
        [
            s["identity_mean"]
            for s in summaries
        ],
        dtype=np.float64,
    )

    pair_macro_mean = float(
        pair_means.mean()
    )

    pair_macro_std = float(
        pair_means.std(ddof=0)
    )

    overall_valid_rate = (
        valid_frames / total_frames
    )

    # --------------------------------------------------------
    # Per-pair CSV
    # --------------------------------------------------------

    per_pair_csv = (
        output_root
        / "per_pair_metrics.csv"
    )

    with per_pair_csv.open(
        "w",
        newline="",
    ) as f:

        fieldnames = [
            "pair_id",
            "source_signer",
            "source_video_id",
            "driver_signer",
            "driver_video_id",
            "total_frames",
            "valid_frames",
            "invalid_frames",
            "valid_rate",
            "identity_mean",
            "identity_std",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for s in summaries:
            writer.writerow(
                {
                    k: s[k]
                    for k in fieldnames
                }
            )

    dataset_summary = {
        "num_pairs":
            len(summaries),

        "total_frames":
            total_frames,

        "valid_frames":
            valid_frames,

        "invalid_frames":
            invalid_frames,

        "valid_rate":
            overall_valid_rate,

        # Primary thesis / GUAVA-aligned score
        "identity_global_frame_weighted_mean":
            float(global_mean),

        "identity_global_frame_weighted_std":
            float(global_std),

        # Auxiliary diagnostic
        "identity_pair_macro_mean":
            pair_macro_mean,

        "identity_pair_macro_std":
            pair_macro_std,

        "primary_metric":
            "identity_global_frame_weighted_mean",

        "aggregation_definition":
            (
                "Sum of ArcFace cosine similarities "
                "over all valid rendered frames divided "
                "by the total number of valid rendered frames."
            ),
    }

    dataset_json = (
        output_root
        / "dataset_summary.json"
    )

    with dataset_json.open("w") as f:
        json.dump(
            dataset_summary,
            f,
            indent=2,
        )

    print("=" * 70)
    print("CROSS DATASET SUMMARY")
    print("=" * 70)

    print(
        "Pairs              :",
        len(summaries),
    )

    print(
        "Total frames       :",
        total_frames,
    )

    print(
        "Valid frames       :",
        valid_frames,
    )

    print(
        "Invalid frames     :",
        invalid_frames,
    )

    print(
        "Valid rate         :",
        f"{overall_valid_rate:.6f}",
    )

    print(
        "Global identity    :",
        f"{global_mean:.8f}",
    )

    print(
        "Global std         :",
        f"{global_std:.8f}",
    )

    print(
        "Pair-macro mean    :",
        f"{pair_macro_mean:.8f}",
    )

    print(
        "Pair-macro std     :",
        f"{pair_macro_std:.8f}",
    )

    print("=" * 70)

    print(
        "Per-pair CSV :",
        per_pair_csv,
    )

    print(
        "Dataset JSON :",
        dataset_json,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "GUAVA cross-reenactment "
            "ArcFace identity evaluator"
        )
    )

    parser.add_argument(
        "--pairs_file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--cross_root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--pair_id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--ctx_id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--insightface_root",
        type=str,
        default=None,
        help=(
            "Root directory for InsightFace model assets. "
            "Required for pair evaluation."
        ),
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
    )

    args = parser.parse_args()

    if args.aggregate:
        aggregate(
            args.pairs_file,
            args.output_root,
        )
        return

    if args.pair_id is None:
        raise RuntimeError(
            "--pair_id is required "
            "unless --aggregate is used"
        )

    if args.cross_root is None:
        raise RuntimeError(
            "--cross_root is required "
            "for pair evaluation"
        )

    if args.insightface_root is None:
        raise RuntimeError(
            "--insightface_root is required "
            "for pair evaluation"
        )

    global CROSS_FACE_ROOT
    CROSS_FACE_ROOT = args.insightface_root

    pairs = load_pairs(
        args.pairs_file
    )

    pair = get_pair(
        pairs,
        args.pair_id,
    )

    evaluate_pair(
        pair=pair,
        cross_root=args.cross_root,
        output_root=args.output_root,
        max_frames=args.max_frames,
        ctx_id=args.ctx_id,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import median


PAIR_COUNT_COL = "num_adjacent_pairs"
METRIC_COL = "temporal_cosine_mean"
EXPECTED_N_VIDEOS = 178


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sequence-paired bootstrap comparison of temporal "
            "consistency against Baseline-FT."
        )
    )

    parser.add_argument(
        "--baseline_csv",
        required=True,
    )

    parser.add_argument(
        "--c1_csv",
        required=True,
    )

    parser.add_argument(
        "--c2_csv",
        required=True,
    )

    parser.add_argument(
        "--full_csv",
        required=True,
    )

    parser.add_argument(
        "--output_root",
        required=True,
    )

    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ci",
        type=float,
        default=95.0,
    )

    parser.add_argument(
        "--sanity_tolerance",
        type=float,
        default=1e-10,
    )

    return parser.parse_args()


def load_csv(path):
    path = Path(path)

    rows = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "video_id",
            "evaluated_frames",
            PAIR_COUNT_COL,
            METRIC_COL,
            "temporal_cosine_std",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"{path}: missing columns: {sorted(missing)}"
            )

        for row in reader:
            video_id = row["video_id"]

            if video_id in rows:
                raise RuntimeError(
                    f"{path}: duplicate video_id: {video_id}"
                )

            rows[video_id] = {
                "video_id": video_id,
                "evaluated_frames": int(
                    row["evaluated_frames"]
                ),
                PAIR_COUNT_COL: int(
                    row[PAIR_COUNT_COL]
                ),
                METRIC_COL: float(
                    row[METRIC_COL]
                ),
            }

    return rows


def load_formal_summary(csv_path):
    csv_path = Path(csv_path)
    summary_path = csv_path.parent / "dataset_summary.json"

    if not summary_path.is_file():
        raise RuntimeError(
            f"Missing formal summary: {summary_path}"
        )

    with summary_path.open() as f:
        summary = json.load(f)

    required = {
        "num_videos",
        "total_adjacent_pairs",
        "temporal_global_pair_weighted_mean",
    }

    missing = required - set(summary)

    if missing:
        raise RuntimeError(
            f"{summary_path}: missing fields: {sorted(missing)}"
        )

    return summary_path, summary


def pair_weighted_mean(rows, indices):
    numerator = 0.0
    denominator = 0

    for idx in indices:
        row = rows[idx]
        n_pairs = row[PAIR_COUNT_COL]
        value = row[METRIC_COL]

        if n_pairs <= 0:
            continue

        if not math.isfinite(value):
            continue

        numerator += n_pairs * value
        denominator += n_pairs

    if denominator <= 0:
        raise RuntimeError(
            "No valid adjacent pairs in weighted mean."
        )

    return numerator / denominator


def percentile(sorted_values, q):
    if not sorted_values:
        raise RuntimeError(
            "Cannot calculate percentile of empty list."
        )

    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return sorted_values[lo]

    frac = pos - lo

    return (
        sorted_values[lo] * (1.0 - frac)
        + sorted_values[hi] * frac
    )


def sample_std(values):
    n = len(values)

    if n < 2:
        return 0.0

    mean_value = sum(values) / n

    variance = sum(
        (x - mean_value) ** 2
        for x in values
    ) / (n - 1)

    return math.sqrt(variance)


def verify_method(
    name,
    csv_path,
    rows_map,
    tolerance,
):
    summary_path, summary = load_formal_summary(
        csv_path
    )

    video_ids = sorted(rows_map)

    if len(video_ids) != EXPECTED_N_VIDEOS:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_N_VIDEOS} videos, "
            f"got {len(video_ids)}"
        )

    rows = [
        rows_map[v]
        for v in video_ids
    ]

    indices = list(range(len(rows)))

    reconstructed = pair_weighted_mean(
        rows,
        indices,
    )

    reconstructed_pairs = sum(
        r[PAIR_COUNT_COL]
        for r in rows
    )

    formal_value = float(
        summary[
            "temporal_global_pair_weighted_mean"
        ]
    )

    formal_pairs = int(
        summary["total_adjacent_pairs"]
    )

    formal_n = int(
        summary["num_videos"]
    )

    if formal_n != EXPECTED_N_VIDEOS:
        raise RuntimeError(
            f"{name}: dataset_summary has "
            f"{formal_n} videos"
        )

    if reconstructed_pairs != formal_pairs:
        raise RuntimeError(
            f"{name}: adjacent-pair mismatch: "
            f"CSV={reconstructed_pairs}, "
            f"summary={formal_pairs}"
        )

    diff = abs(
        reconstructed - formal_value
    )

    if diff > tolerance:
        raise RuntimeError(
            f"{name}: reconstructed TCS does not "
            f"match formal dataset summary.\n"
            f"CSV reconstructed = {reconstructed:.12f}\n"
            f"Formal summary     = {formal_value:.12f}\n"
            f"Absolute diff      = {diff:.3e}\n"
            f"Tolerance          = {tolerance:.3e}"
        )

    return {
        "summary_path": str(summary_path),
        "formal_global_tcs": formal_value,
        "csv_reconstructed_global_tcs":
            reconstructed,
        "absolute_difference": diff,
        "total_adjacent_pairs":
            reconstructed_pairs,
    }


def main():
    args = parse_args()

    paths = {
        "baseline_ft": args.baseline_csv,
        "c1": args.c1_csv,
        "c2": args.c2_csv,
        "full": args.full_csv,
    }

    maps = {
        name: load_csv(path)
        for name, path in paths.items()
    }

    # ---------------------------------------------------------
    # Require identical frozen TEST sequence sets.
    # ---------------------------------------------------------
    baseline_ids = set(
        maps["baseline_ft"]
    )

    for name in ["c1", "c2", "full"]:
        method_ids = set(maps[name])

        if method_ids != baseline_ids:
            only_baseline = sorted(
                baseline_ids - method_ids
            )

            only_method = sorted(
                method_ids - baseline_ids
            )

            raise RuntimeError(
                f"{name}: video set does not match Baseline-FT.\n"
                f"Only baseline: {only_baseline}\n"
                f"Only method: {only_method}"
            )

    if len(baseline_ids) != EXPECTED_N_VIDEOS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_VIDEOS} paired "
            f"videos, got {len(baseline_ids)}"
        )

    video_ids = sorted(
        baseline_ids
    )

    rows = {
        name: [
            maps[name][video_id]
            for video_id in video_ids
        ]
        for name in maps
    }

    # ---------------------------------------------------------
    # Sanity check against the frozen formal aggregator.
    # ---------------------------------------------------------
    sanity = {}

    for name in paths:
        sanity[name] = verify_method(
            name=name,
            csv_path=paths[name],
            rows_map=maps[name],
            tolerance=args.sanity_tolerance,
        )

    # ---------------------------------------------------------
    # Check pair-count equality across methods.
    # ---------------------------------------------------------
    pair_count_mismatches = []

    for i, video_id in enumerate(video_ids):
        counts = {
            name:
                rows[name][i][PAIR_COUNT_COL]
            for name in rows
        }

        if len(set(counts.values())) != 1:
            pair_count_mismatches.append(
                {
                    "video_id":
                        video_id,
                    **counts,
                }
            )

    # ---------------------------------------------------------
    # Observed primary dataset statistics.
    # ---------------------------------------------------------
    all_indices = list(
        range(len(video_ids))
    )

    observed = {
        name: pair_weighted_mean(
            rows[name],
            all_indices,
        )
        for name in rows
    }

    # ---------------------------------------------------------
    # Per-video paired gains:
    # positive = method has higher TCS than Baseline-FT.
    # ---------------------------------------------------------
    methods = [
        "c1",
        "c2",
        "full",
    ]

    per_video_gains = {
        method: []
        for method in methods
    }

    paired_rows = []

    for i, video_id in enumerate(video_ids):
        baseline_value = (
            rows["baseline_ft"][i][METRIC_COL]
        )

        out = {
            "video_id":
                video_id,
            "baseline_num_adjacent_pairs":
                rows["baseline_ft"][i][
                    PAIR_COUNT_COL
                ],
            "baseline_tcs":
                baseline_value,
        }

        for method in methods:
            method_value = (
                rows[method][i][METRIC_COL]
            )

            gain = (
                method_value
                - baseline_value
            )

            per_video_gains[
                method
            ].append(
                gain
            )

            out[
                f"{method}_num_adjacent_pairs"
            ] = (
                rows[method][i][
                    PAIR_COUNT_COL
                ]
            )

            out[
                f"{method}_tcs"
            ] = method_value

            out[
                f"{method}_gain"
            ] = gain

        paired_rows.append(out)

    # ---------------------------------------------------------
    # Paired sequence-cluster bootstrap.
    #
    # Same resampled sequence indices are used for all methods.
    # Each method's formal pair-weighted dataset TCS is
    # recomputed independently within every replicate.
    # ---------------------------------------------------------
    rng = random.Random(
        args.seed
    )

    bootstrap_gains = {
        method: []
        for method in methods
    }

    bootstrap_rows = []

    n = len(video_ids)

    for bootstrap_id in range(
        args.n_bootstrap
    ):
        sampled_indices = [
            rng.randrange(n)
            for _ in range(n)
        ]

        baseline_boot = (
            pair_weighted_mean(
                rows["baseline_ft"],
                sampled_indices,
            )
        )

        out = {
            "bootstrap_id":
                bootstrap_id,
            "baseline_tcs":
                baseline_boot,
        }

        for method in methods:
            method_boot = (
                pair_weighted_mean(
                    rows[method],
                    sampled_indices,
                )
            )

            gain = (
                method_boot
                - baseline_boot
            )

            bootstrap_gains[
                method
            ].append(
                gain
            )

            out[
                f"{method}_tcs"
            ] = method_boot

            out[
                f"{method}_gain"
            ] = gain

        bootstrap_rows.append(
            out
        )

    # ---------------------------------------------------------
    # Confidence intervals and diagnostics.
    # ---------------------------------------------------------
    alpha = (
        100.0 - args.ci
    ) / 100.0

    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    comparisons = {}

    for method in methods:
        samples = sorted(
            bootstrap_gains[method]
        )

        gains = (
            per_video_gains[method]
        )

        wins = sum(
            g > 0
            for g in gains
        )

        losses = sum(
            g < 0
            for g in gains
        )

        ties = (
            len(gains)
            - wins
            - losses
        )

        ci_low = percentile(
            samples,
            lower_q,
        )

        ci_high = percentile(
            samples,
            upper_q,
        )

        point_gain = (
            observed[method]
            - observed["baseline_ft"]
        )

        comparisons[method] = {
            "baseline_ft_global_tcs":
                observed["baseline_ft"],

            "method_global_tcs":
                observed[method],

            "gain_method_minus_baseline_ft":
                point_gain,

            "bootstrap_ci_low":
                ci_low,

            "bootstrap_ci_high":
                ci_high,

            "bootstrap_standard_error":
                sample_std(
                    bootstrap_gains[method]
                ),

            "ci_excludes_zero":
                (
                    ci_low > 0
                    or ci_high < 0
                ),

            "median_per_video_gain":
                median(gains),

            "wins_method_better":
                wins,

            "losses_baseline_ft_better":
                losses,

            "ties":
                ties,

            "win_rate":
                wins / len(gains),
        }

    # ---------------------------------------------------------
    # Save outputs.
    # ---------------------------------------------------------
    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired_path = (
        output_root
        / "per_video_paired_deltas.csv"
    )

    with paired_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                paired_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            paired_rows
        )

    bootstrap_path = (
        output_root
        / "bootstrap_samples.csv"
    )

    with bootstrap_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                bootstrap_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            bootstrap_rows
        )

    summary = {
        "analysis":
            "TCS paired sequence-cluster bootstrap",

        "baseline":
            "Baseline-FT@200k",

        "comparisons": {
            "c1":
                "C1@200k vs Baseline-FT@200k",

            "c2":
                "C2@200k vs Baseline-FT@200k",

            "full":
                "Full@200k vs Baseline-FT@200k",
        },

        "n_sequences":
            len(video_ids),

        "metric":
            "DINOv2 adjacent-frame temporal cosine",

        "primary_estimator":
            (
                "Arithmetic mean over all "
                "adjacent-frame pairs; implemented "
                "as num_adjacent_pairs-weighted mean "
                "of per-video temporal_cosine_mean."
            ),

        "bootstrap": {
            "unit":
                "video sequence",

            "paired":
                True,

            "n_bootstrap":
                args.n_bootstrap,

            "seed":
                args.seed,

            "confidence_level_percent":
                args.ci,

            "interval":
                "percentile",

            "gain_definition":
                "method - Baseline-FT",

            "positive_gain":
                "method has higher TCS",
        },

        "sanity_against_formal_dataset_summary":
            sanity,

        "pair_count_mismatches": {
            "count":
                len(pair_count_mismatches),

            "examples":
                pair_count_mismatches[:10],
        },

        "results":
            comparisons,
    }

    summary_path = (
        output_root
        / "bootstrap_summary.json"
    )

    with summary_path.open(
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console report.
    # ---------------------------------------------------------
    print("=" * 112)
    print(
        "TCS PAIRED SEQUENCE-CLUSTER BOOTSTRAP"
    )
    print("=" * 112)

    print(
        f"Paired sequences       : "
        f"{len(video_ids)}"
    )

    print(
        f"Bootstrap samples      : "
        f"{args.n_bootstrap}"
    )

    print(
        f"Seed                   : "
        f"{args.seed}"
    )

    print(
        f"CI                     : "
        f"{args.ci:.1f}%"
    )

    print(
        f"Pair-count mismatches  : "
        f"{len(pair_count_mismatches)}"
    )

    print()

    print(
        "Sanity reconstruction:"
    )

    for name in [
        "baseline_ft",
        "c1",
        "c2",
        "full",
    ]:
        s = sanity[name]

        print(
            f"  {name:<12} "
            f"{s['formal_global_tcs']:.8f} "
            f"(diff={s['absolute_difference']:.3e}, "
            f"pairs={s['total_adjacent_pairs']})"
        )

    print()

    print(
        f"{'Method':<12}"
        f"{'Baseline':>14}"
        f"{'Method TCS':>14}"
        f"{'Gain':>14}"
        f"{'CI low':>14}"
        f"{'CI high':>14}"
        f"{'SE':>14}"
        f"{'Wins':>8}"
        f"{'Win %':>10}"
    )

    print(
        "-" * 114
    )

    for method in methods:
        r = comparisons[method]

        print(
            f"{method.upper():<12}"
            f"{r['baseline_ft_global_tcs']:>14.8f}"
            f"{r['method_global_tcs']:>14.8f}"
            f"{r['gain_method_minus_baseline_ft']:>14.8f}"
            f"{r['bootstrap_ci_low']:>14.8f}"
            f"{r['bootstrap_ci_high']:>14.8f}"
            f"{r['bootstrap_standard_error']:>14.8f}"
            f"{r['wins_method_better']:>8d}"
            f"{100.0 * r['win_rate']:>9.2f}%"
        )

    print()
    print(
        "Positive gain means higher TCS "
        "than Baseline-FT."
    )

    print()
    print("Saved:")
    print(f"  {paired_path}")
    print(f"  {bootstrap_path}")
    print(f"  {summary_path}")

    print("=" * 112)


if __name__ == "__main__":
    main()

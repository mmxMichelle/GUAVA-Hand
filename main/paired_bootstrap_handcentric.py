#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import median


METRICS = {
    # metric_name: (csv_column, higher_is_better)
    "psnr": ("hand_combined_psnr", True),
    "ssim": ("hand_combined_ssim", True),
    "lpips": ("hand_combined_lpips", False),
    "l1": ("hand_combined_l1", False),
}

WEIGHT_COLUMN = "hand_combined_valid_frames"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sequence-paired bootstrap comparison of Combined Hands metrics "
            "between Baseline-FT and Full."
        )
    )
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--full_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--ci", type=float, default=95.0)
    return parser.parse_args()


def load_csv(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "video_id",
            WEIGHT_COLUMN,
            *[column for column, _ in METRICS.values()],
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"{path}: missing required columns: {sorted(missing)}"
            )

        for row in reader:
            video_id = row["video_id"]

            if video_id in rows:
                raise RuntimeError(
                    f"{path}: duplicate video_id: {video_id}"
                )

            parsed = {
                "video_id": video_id,
                WEIGHT_COLUMN: int(row[WEIGHT_COLUMN]),
            }

            for _, (column, _) in METRICS.items():
                parsed[column] = float(row[column])

            rows[video_id] = parsed

    return rows


def weighted_mean(rows, indices, metric_column):
    numerator = 0.0
    denominator = 0

    for idx in indices:
        row = rows[idx]
        weight = row[WEIGHT_COLUMN]
        value = row[metric_column]

        if weight <= 0 or not math.isfinite(value):
            continue

        numerator += weight * value
        denominator += weight

    if denominator == 0:
        raise RuntimeError(
            f"No valid frames available for {metric_column}"
        )

    return numerator / denominator


def percentile(sorted_values, q):
    """
    Linear interpolation percentile, q in [0, 1].
    """
    n = len(sorted_values)

    if n == 0:
        raise RuntimeError("Cannot take percentile of empty list")

    if n == 1:
        return sorted_values[0]

    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return sorted_values[lo]

    frac = pos - lo
    return (
        sorted_values[lo] * (1.0 - frac)
        + sorted_values[hi] * frac
    )


def gain(full_value, baseline_value, higher_is_better):
    """
    Always define positive gain as Full being better.
    """
    if higher_is_better:
        return full_value - baseline_value

    return baseline_value - full_value


def main():
    args = parse_args()

    baseline_map = load_csv(args.baseline_csv)
    full_map = load_csv(args.full_csv)

    baseline_ids = set(baseline_map)
    full_ids = set(full_map)

    if baseline_ids != full_ids:
        only_baseline = sorted(baseline_ids - full_ids)
        only_full = sorted(full_ids - baseline_ids)

        raise RuntimeError(
            "Video sets do not match.\n"
            f"Only Baseline-FT: {only_baseline}\n"
            f"Only Full: {only_full}"
        )

    video_ids = sorted(baseline_ids)

    if len(video_ids) != 178:
        raise RuntimeError(
            f"Expected 178 paired videos, got {len(video_ids)}"
        )

    baseline_rows = [baseline_map[v] for v in video_ids]
    full_rows = [full_map[v] for v in video_ids]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Check valid-frame agreement.
    # ------------------------------------------------------------
    valid_count_mismatches = []

    for i, video_id in enumerate(video_ids):
        b = baseline_rows[i][WEIGHT_COLUMN]
        f = full_rows[i][WEIGHT_COLUMN]

        if b != f:
            valid_count_mismatches.append(
                {
                    "video_id": video_id,
                    "baseline_valid_frames": b,
                    "full_valid_frames": f,
                }
            )

    # ------------------------------------------------------------
    # Observed dataset-level primary statistics.
    # These reproduce the existing valid-frame-weighted aggregate.
    # ------------------------------------------------------------
    all_indices = list(range(len(video_ids)))
    observed = {}

    for metric_name, (column, higher_is_better) in METRICS.items():
        baseline_value = weighted_mean(
            baseline_rows,
            all_indices,
            column,
        )

        full_value = weighted_mean(
            full_rows,
            all_indices,
            column,
        )

        observed_gain = gain(
            full_value,
            baseline_value,
            higher_is_better,
        )

        observed[metric_name] = {
            "baseline": baseline_value,
            "full": full_value,
            "gain_positive_means_full_better": observed_gain,
        }

    # ------------------------------------------------------------
    # Per-video paired deltas.
    # These support median gain and win-rate diagnostics.
    # ------------------------------------------------------------
    paired_delta_rows = []
    per_video_gains = {metric_name: [] for metric_name in METRICS}

    for i, video_id in enumerate(video_ids):
        out = {
            "video_id": video_id,
            "baseline_valid_frames": baseline_rows[i][WEIGHT_COLUMN],
            "full_valid_frames": full_rows[i][WEIGHT_COLUMN],
        }

        for metric_name, (column, higher_is_better) in METRICS.items():
            baseline_value = baseline_rows[i][column]
            full_value = full_rows[i][column]

            g = gain(
                full_value,
                baseline_value,
                higher_is_better,
            )

            out[f"baseline_{metric_name}"] = baseline_value
            out[f"full_{metric_name}"] = full_value
            out[f"gain_{metric_name}"] = g

            per_video_gains[metric_name].append(g)

        paired_delta_rows.append(out)

    paired_delta_path = output_root / "per_video_paired_deltas.csv"

    with open(paired_delta_path, "w", newline="") as f:
        fieldnames = list(paired_delta_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_delta_rows)

    # ------------------------------------------------------------
    # Paired cluster bootstrap.
    #
    # Resampling unit = video/sequence.
    # The SAME sampled video indices are used for both methods.
    # Within each replicate, each method's formal valid-frame
    # weighted dataset statistic is recomputed.
    # ------------------------------------------------------------
    rng = random.Random(args.seed)

    bootstrap_gains = {
        metric_name: []
        for metric_name in METRICS
    }

    bootstrap_rows = []

    n = len(video_ids)

    for bootstrap_id in range(args.n_bootstrap):
        sampled_indices = [
            rng.randrange(n)
            for _ in range(n)
        ]

        row_out = {
            "bootstrap_id": bootstrap_id,
        }

        for metric_name, (column, higher_is_better) in METRICS.items():
            baseline_value = weighted_mean(
                baseline_rows,
                sampled_indices,
                column,
            )

            full_value = weighted_mean(
                full_rows,
                sampled_indices,
                column,
            )

            g = gain(
                full_value,
                baseline_value,
                higher_is_better,
            )

            bootstrap_gains[metric_name].append(g)
            row_out[f"gain_{metric_name}"] = g

        bootstrap_rows.append(row_out)

    bootstrap_samples_path = output_root / "bootstrap_samples.csv"

    with open(bootstrap_samples_path, "w", newline="") as f:
        fieldnames = list(bootstrap_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bootstrap_rows)

    # ------------------------------------------------------------
    # Confidence intervals + win rates.
    # ------------------------------------------------------------
    alpha = (100.0 - args.ci) / 100.0
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    summary = {
        "comparison": "Full@200k vs Baseline-FT@200k",
        "region": "Combined Hands",
        "paired_unit": "video sequence",
        "n_sequences": len(video_ids),
        "bootstrap": {
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "confidence_level_percent": args.ci,
            "method": "paired sequence-level percentile bootstrap",
            "primary_estimator": (
                "valid-frame-weighted mean of per-video mean "
                "frame-level metrics"
            ),
            "gain_definition": {
                "psnr": "Full - Baseline-FT",
                "ssim": "Full - Baseline-FT",
                "lpips": "Baseline-FT - Full",
                "l1": "Baseline-FT - Full",
                "positive": "Full better",
            },
        },
        "valid_frame_count_mismatches": {
            "count": len(valid_count_mismatches),
            "examples": valid_count_mismatches[:10],
        },
        "metrics": {},
    }

    for metric_name in METRICS:
        samples = sorted(bootstrap_gains[metric_name])
        video_gains = per_video_gains[metric_name]

        wins = sum(g > 0 for g in video_gains)
        losses = sum(g < 0 for g in video_gains)
        ties = len(video_gains) - wins - losses

        ci_low = percentile(samples, lower_q)
        ci_high = percentile(samples, upper_q)

        summary["metrics"][metric_name] = {
            **observed[metric_name],
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "ci_excludes_zero": (ci_low > 0 or ci_high < 0),
            "median_per_video_gain": median(video_gains),
            "wins_full_better": wins,
            "losses_baseline_ft_better": losses,
            "ties": ties,
            "win_rate": wins / len(video_gains),
        }

    summary_path = output_root / "bootstrap_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------
    print("=" * 100)
    print("PAIRED COMBINED-HANDS BOOTSTRAP")
    print("=" * 100)
    print(f"Paired sequences : {len(video_ids)}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Seed             : {args.seed}")
    print(f"CI               : {args.ci:.1f}%")
    print(
        "Valid-count mismatches:",
        len(valid_count_mismatches),
    )
    print()

    print(
        f"{'Metric':<8}"
        f"{'Baseline-FT':>15}"
        f"{'Full':>15}"
        f"{'Gain':>15}"
        f"{'CI low':>15}"
        f"{'CI high':>15}"
        f"{'Wins':>10}"
        f"{'Win %':>10}"
    )

    print("-" * 113)

    for metric_name in METRICS:
        m = summary["metrics"][metric_name]

        print(
            f"{metric_name.upper():<8}"
            f"{m['baseline']:>15.8f}"
            f"{m['full']:>15.8f}"
            f"{m['gain_positive_means_full_better']:>15.8f}"
            f"{m['bootstrap_ci_low']:>15.8f}"
            f"{m['bootstrap_ci_high']:>15.8f}"
            f"{m['wins_full_better']:>10d}"
            f"{100.0 * m['win_rate']:>9.2f}%"
        )

    print()
    print("Positive gain always means Full is better.")
    print()
    print("Saved:")
    print(f"  {paired_delta_path}")
    print(f"  {bootstrap_samples_path}")
    print(f"  {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()

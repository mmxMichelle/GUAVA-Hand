#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


REGIONS = [
    "full",
    "head",
    "left_hand",
    "right_hand",
    "hand_combined",
]

METRICS = [
    "psnr",
    "l1",
    "ssim",
    "lpips",
]


def load_manifest_video_ids(manifest_path):
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise RuntimeError(
            f"Manifest not found: {manifest_path}"
        )

    lines = [
        line.strip()
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]

    video_ids = [
        Path(line).name
        for line in lines
    ]

    if len(video_ids) != 178:
        raise RuntimeError(
            f"Expected 178 manifest entries, got {len(video_ids)}"
        )

    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError(
            "Duplicate video IDs found in manifest"
        )

    return video_ids


def read_summary(path):
    with path.open("r") as f:
        data = json.load(f)

    if "video_id" not in data:
        raise RuntimeError(
            f"Missing video_id: {path}"
        )

    if "regions" not in data:
        raise RuntimeError(
            f"Missing regions: {path}"
        )

    for region in REGIONS:
        if region not in data["regions"]:
            raise RuntimeError(
                f"Missing region '{region}': {path}"
            )

        region_data = data["regions"][region]

        for key in ["valid_frames"] + METRICS:
            if key not in region_data:
                raise RuntimeError(
                    f"Missing {region}.{key}: {path}"
                )

        valid_frames = region_data["valid_frames"]

        if not isinstance(valid_frames, int):
            raise RuntimeError(
                f"{region}.valid_frames is not int: {path}"
            )

        if valid_frames <= 0:
            raise RuntimeError(
                f"{region}.valid_frames <= 0: {path}"
            )

        for metric in METRICS:
            value = float(region_data[metric])

            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite {region}.{metric}: {path}"
                )

    return data


def weighted_mean(summaries, region, metric):
    numerator = 0.0
    denominator = 0

    for summary in summaries:
        region_data = summary["regions"][region]

        n = int(region_data["valid_frames"])
        value = float(region_data[metric])

        numerator += n * value
        denominator += n

    if denominator == 0:
        raise RuntimeError(
            f"No valid frames for {region}.{metric}"
        )

    return numerator / denominator


def macro_mean(summaries, region, metric):
    values = [
        float(summary["regions"][region][metric])
        for summary in summaries
    ]

    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate 178-sequence hand-centric "
            "self-reenactment evaluation results."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="baseline_178_data_paths.txt",
    )

    parser.add_argument(
        "--output_root",
        required=True,
        help=(
            "Root containing per_video/<video_id>/summary.json"
        ),
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    per_video_root = output_root / "per_video"

    if not per_video_root.is_dir():
        raise RuntimeError(
            f"per_video directory not found: {per_video_root}"
        )

    # ---------------------------------------------------------
    # 1. Fixed 178-sequence cohort from manifest
    # ---------------------------------------------------------

    expected_video_ids = load_manifest_video_ids(
        manifest_path
    )

    expected_set = set(expected_video_ids)

    actual_dirs = sorted(
        p.name
        for p in per_video_root.iterdir()
        if p.is_dir()
    )

    actual_set = set(actual_dirs)

    missing_dirs = sorted(
        expected_set - actual_set
    )

    extra_dirs = sorted(
        actual_set - expected_set
    )

    if missing_dirs:
        raise RuntimeError(
            "Missing expected video directories:\n"
            + "\n".join(missing_dirs)
        )

    if extra_dirs:
        raise RuntimeError(
            "Unexpected extra video directories:\n"
            + "\n".join(extra_dirs)
        )

    if len(actual_dirs) != 178:
        raise RuntimeError(
            f"Expected 178 video dirs, got {len(actual_dirs)}"
        )

    # ---------------------------------------------------------
    # 2. Load and validate all summaries
    # ---------------------------------------------------------

    summaries = []

    for video_id in expected_video_ids:
        summary_path = (
            per_video_root
            / video_id
            / "summary.json"
        )

        if not summary_path.is_file():
            raise RuntimeError(
                f"Missing summary: {summary_path}"
            )

        summary = read_summary(
            summary_path
        )

        if summary["video_id"] != video_id:
            raise RuntimeError(
                f"Video ID mismatch:\n"
                f"expected={video_id}\n"
                f"summary={summary['video_id']}"
            )

        summaries.append(summary)

    if len(summaries) != 178:
        raise RuntimeError(
            f"Expected 178 summaries, got {len(summaries)}"
        )

    # ---------------------------------------------------------
    # 3. Dataset-level weighted aggregation
    # ---------------------------------------------------------

    dataset_regions = {}

    for region in REGIONS:
        total_valid_frames = sum(
            int(
                summary["regions"][region]["valid_frames"]
            )
            for summary in summaries
        )

        region_result = {
            "valid_frames":
                total_valid_frames
        }

        for metric in METRICS:
            region_result[metric] = weighted_mean(
                summaries,
                region,
                metric,
            )

        # Auxiliary statistics only.
        # Main thesis values are the valid-frame-weighted metrics above.
        region_result["video_macro"] = {
            metric: macro_mean(
                summaries,
                region,
                metric,
            )
            for metric in METRICS
        }

        dataset_regions[region] = region_result

    # ---------------------------------------------------------
    # 4. Sanity checks
    # ---------------------------------------------------------

    full_frames = dataset_regions["full"]["valid_frames"]

    if full_frames != 454936:
        raise RuntimeError(
            f"Expected 454936 full valid frames, "
            f"got {full_frames}"
        )

    left_frames = (
        dataset_regions["left_hand"]["valid_frames"]
    )

    right_frames = (
        dataset_regions["right_hand"]["valid_frames"]
    )

    combined_frames = (
        dataset_regions["hand_combined"]["valid_frames"]
    )

    if combined_frames > left_frames:
        raise RuntimeError(
            "Combined-hand valid frames exceed left-hand frames"
        )

    if combined_frames > right_frames:
        raise RuntimeError(
            "Combined-hand valid frames exceed right-hand frames"
        )

    # ---------------------------------------------------------
    # 5. Save per-video compact CSV
    # ---------------------------------------------------------

    per_video_csv = (
        output_root
        / "per_video_metrics.csv"
    )

    fieldnames = [
        "video_id",
        "rendered_frames",
        "evaluated_frames",
    ]

    for region in REGIONS:
        fieldnames.extend([
            f"{region}_valid_frames",
            f"{region}_psnr",
            f"{region}_l1",
            f"{region}_ssim",
            f"{region}_lpips",
        ])

    with per_video_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for summary in summaries:
            row = {
                "video_id":
                    summary["video_id"],

                "rendered_frames":
                    summary["frames"]["rendered"],

                "evaluated_frames":
                    summary["frames"]["evaluated"],
            }

            for region in REGIONS:
                region_data = (
                    summary["regions"][region]
                )

                row[
                    f"{region}_valid_frames"
                ] = region_data["valid_frames"]

                for metric in METRICS:
                    row[
                        f"{region}_{metric}"
                    ] = region_data[metric]

            writer.writerow(row)

    # ---------------------------------------------------------
    # 6. Save dataset summary
    # ---------------------------------------------------------

    dataset_summary = {
        "num_videos": 178,

        "cohort_manifest":
            str(manifest_path),

        "aggregation": (
            "Valid-frame-weighted mean of "
            "per-video mean frame-level metrics."
        ),

        "primary_statistics":
            "valid-frame-weighted",

        "regions":
            dataset_regions,
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

    # ---------------------------------------------------------
    # 7. Human-readable terminal result
    # ---------------------------------------------------------

    print("=" * 84)
    print("HAND-CENTRIC SELF-REENACTMENT DATASET SUMMARY")
    print("=" * 84)

    print(f"Videos: {len(summaries)}")
    print(f"Full valid frames: {full_frames}")
    print()

    print(
        f"{'Region':<18}"
        f"{'Valid':>10}"
        f"{'PSNR ↑':>14}"
        f"{'SSIM ↑':>14}"
        f"{'LPIPS ↓':>14}"
    )

    print("-" * 84)

    display_names = {
        "full": "Full",
        "head": "Head",
        "left_hand": "Left Hand",
        "right_hand": "Right Hand",
        "hand_combined": "Combined Hands",
    }

    for region in REGIONS:
        r = dataset_regions[region]

        print(
            f"{display_names[region]:<18}"
            f"{r['valid_frames']:>10d}"
            f"{r['psnr']:>14.6f}"
            f"{r['l1']:>14.6f}"
            f"{r['ssim']:>14.6f}"
            f"{r['lpips']:>14.6f}"
        )

    print("=" * 84)

    print()
    print("Saved:")
    print(f"  {per_video_csv}")
    print(f"  {dataset_json}")


if __name__ == "__main__":
    main()

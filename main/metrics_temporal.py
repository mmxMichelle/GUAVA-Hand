#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.modules.net_module.dino_encoder import DINO_Enocder


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

DINO_MODEL_NAME = "dinov2_vitb14"
FEATURE_DIM = 768
FEATURE_IMG_SIZE = 518


def frame_sort_key(path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


class RawDinoGlobalExtractor:
    """
    Reuse GUAVA's existing DINOv2 backbone and normalization,
    but extract ONLY the raw frozen global CLS feature.

    We deliberately do NOT use:
      - global_feature_mapping
      - uv_style_mapping
      - f_map1
      - f_map2
      - any GUAVA checkpoint-dependent projection
    """

    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)

        # Head dimensions match the standard GUAVA configuration,
        # but these heads are NOT used by the temporal evaluator.
        self.encoder = DINO_Enocder(
            output_dim=32,
            output_dim_2=128,
            hidden_dims=64,
        ).to(self.device)

        self.encoder.eval()

        self.dino_model = self.encoder.dino_model
        self.dino_model.eval()

        for param in self.dino_model.parameters():
            param.requires_grad = False

        # Reuse the exact ImageNet normalization defined by GUAVA.
        self.dino_normalize = self.encoder.dino_normlize

    @torch.inference_mode()
    def extract(self, images):
        """
        images:
            [B, 3, 518, 518], RGB, float [0,1]

        returns:
            [B, 768], L2-normalized raw DINOv2 CLS features
        """

        images = images.to(
            self.device,
            non_blocking=True,
        )

        # Exact GUAVA DINO normalization.
        images = self.dino_normalize(images)

        # Mirror GUAVA:
        # image_features =
        #     self.dino_model.get_intermediate_layers(images, 5)
        #
        # low_level_features, *image_features = image_features
        # out_global = image_features[-1][:, 0]

        image_features = (
            self.dino_model
            .get_intermediate_layers(images, 5)
        )

        _, *image_features = image_features

        f_global = image_features[-1][:, 0]

        if f_global.shape[-1] != FEATURE_DIM:
            raise RuntimeError(
                f"Unexpected DINO global dimension: "
                f"{f_global.shape}"
            )

        # Explicit normalization for cosine evaluation.
        f_global = F.normalize(
            f_global.float(),
            p=2,
            dim=-1,
        )

        return f_global


def load_render_image(path):
    """
    Saved PNG -> RGB -> float [0,1] ->
    518x518 resize with antialiasing.

    ImageNet normalization is applied by
    RawDinoGlobalExtractor using GUAVA's
    existing normalization object.
    """

    with Image.open(path) as image:
        image = image.convert("RGB")

        tensor = (
            TF.pil_to_tensor(image)
            .float()
            .div(255.0)
        )

    tensor = TF.resize(
        tensor,
        [FEATURE_IMG_SIZE, FEATURE_IMG_SIZE],
        antialias=True,
    )

    return tensor


def evaluate_sequence(
    video_id,
    render_dir,
    output_root,
    batch_size=16,
    max_frames=0,
    device="cuda:0",
):
    render_dir = Path(render_dir)
    output_root = Path(output_root)

    if not render_dir.is_dir():
        raise RuntimeError(
            f"Render directory missing: {render_dir}"
        )

    frame_files = sorted(
        [
            p for p in render_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        ],
        key=frame_sort_key,
    )

    available_frames = len(frame_files)

    if available_frames < 2:
        raise RuntimeError(
            f"Need at least 2 frames, got "
            f"{available_frames}: {render_dir}"
        )

    if max_frames and max_frames > 0:
        frame_files = frame_files[:max_frames]

    if len(frame_files) < 2:
        raise RuntimeError(
            "max_frames leaves fewer than 2 frames"
        )

    print("=" * 72)
    print("GUAVA TEMPORAL CONSISTENCY EVALUATION")
    print("=" * 72)
    print("Video ID          :", video_id)
    print("Render directory  :", render_dir)
    print("Available frames  :", available_frames)
    print("Evaluating frames :", len(frame_files))
    print("Adjacent pairs    :", len(frame_files) - 1)
    print("DINO model        :", DINO_MODEL_NAME)
    print("Feature           : raw CLS f_global")
    print("Feature dimension :", FEATURE_DIM)
    print("Resize             :", FEATURE_IMG_SIZE)
    print("Input protocol     : full rendered RGB, unmasked")
    print("Batch size         :", batch_size)
    print("Device             :", device)
    print("=" * 72)

    extractor = RawDinoGlobalExtractor(
        device=device
    )

    rows = []
    similarities = []

    previous_feature = None
    previous_name = None

    for start in tqdm(
        range(0, len(frame_files), batch_size),
        desc=video_id,
    ):
        batch_files = frame_files[
            start:start + batch_size
        ]

        images = torch.stack(
            [
                load_render_image(p)
                for p in batch_files
            ],
            dim=0,
        )

        features = extractor.extract(images)

        # Across batch boundary:
        # previous batch last frame ->
        # current batch first frame
        if previous_feature is not None:
            similarity = torch.sum(
                previous_feature
                * features[0]
            )

            similarity = float(
                torch.clamp(
                    similarity,
                    -1.0,
                    1.0,
                ).item()
            )

            rows.append(
                {
                    "frame_t": previous_name,
                    "frame_t1": batch_files[0].name,
                    "temporal_cosine": similarity,
                }
            )

            similarities.append(similarity)

        # Within current batch
        if features.shape[0] > 1:
            batch_sims = torch.sum(
                features[:-1]
                * features[1:],
                dim=-1,
            )

            batch_sims = torch.clamp(
                batch_sims,
                -1.0,
                1.0,
            ).detach().cpu().tolist()

            for i, similarity in enumerate(
                batch_sims
            ):
                similarity = float(similarity)

                rows.append(
                    {
                        "frame_t":
                            batch_files[i].name,

                        "frame_t1":
                            batch_files[i + 1].name,

                        "temporal_cosine":
                            similarity,
                    }
                )

                similarities.append(similarity)

        previous_feature = (
            features[-1].detach()
        )

        previous_name = (
            batch_files[-1].name
        )

        del images, features

    expected_pairs = len(frame_files) - 1

    if len(similarities) != expected_pairs:
        raise RuntimeError(
            f"Adjacent pair mismatch: "
            f"expected={expected_pairs}, "
            f"actual={len(similarities)}"
        )

    sims = np.asarray(
        similarities,
        dtype=np.float64,
    )

    if not np.isfinite(sims).all():
        raise RuntimeError(
            "Temporal similarities contain NaN/Inf"
        )

    temporal_sum = float(sims.sum())
    temporal_sq_sum = float(
        np.square(sims).sum()
    )

    temporal_mean = float(sims.mean())
    temporal_std = float(
        sims.std(ddof=0)
    )

    video_output = (
        output_root
        / "per_video"
        / video_id
    )

    video_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        video_output
        / "per_adjacent_pair.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_t",
                "frame_t1",
                "temporal_cosine",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "video_id":
            video_id,

        "render_dir":
            str(render_dir),

        "available_frames":
            available_frames,

        "evaluated_frames":
            len(frame_files),

        "num_adjacent_pairs":
            expected_pairs,

        "temporal_cosine_sum":
            temporal_sum,

        "temporal_cosine_sq_sum":
            temporal_sq_sum,

        "temporal_cosine_mean":
            temporal_mean,

        "temporal_cosine_std":
            temporal_std,

        "feature_extractor":
            "DINOv2 ViT-B/14",

        "feature_tensor":
            "raw f_global CLS token",

        "feature_dimension":
            FEATURE_DIM,

        "feature_resize":
            FEATURE_IMG_SIZE,

        "feature_normalization":
            (
                "GUAVA ImageNet normalization "
                "followed by L2 normalization"
            ),

        "input_protocol":
            "full rendered RGB, unmasked",

        "temporal_definition":
            (
                "Cosine similarity between "
                "L2-normalized raw DINOv2 CLS "
                "features of adjacent rendered frames."
            ),
    }

    summary_path = (
        video_output
        / "summary.json"
    )

    with summary_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TEMPORAL RESULT")
    print("=" * 72)
    print(
        "Evaluated frames :",
        len(frame_files),
    )
    print(
        "Adjacent pairs   :",
        expected_pairs,
    )
    print(
        "Temporal mean    :",
        f"{temporal_mean:.8f}",
    )
    print(
        "Temporal std     :",
        f"{temporal_std:.8f}",
    )
    print("CSV              :", csv_path)
    print("Summary          :", summary_path)
    print("=" * 72)


def load_expected_video_ids(manifest):
    manifest = Path(manifest)

    paths = [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]

    video_ids = [
        Path(p).name
        for p in paths
    ]

    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError(
            "Duplicate video IDs in manifest"
        )

    return video_ids


def aggregate_dataset(
    manifest,
    output_root,
):
    output_root = Path(output_root)

    video_ids = load_expected_video_ids(
        manifest
    )

    summaries = []
    missing = []

    for video_id in video_ids:
        path = (
            output_root
            / "per_video"
            / video_id
            / "summary.json"
        )

        if not path.is_file():
            missing.append(str(path))
            continue

        with path.open("r") as f:
            summaries.append(
                json.load(f)
            )

    if missing:
        print(
            f"Missing {len(missing)} summaries"
        )

        for path in missing[:20]:
            print("  ", path)

        raise RuntimeError(
            "Aggregation aborted: "
            "not all expected videos are complete."
        )

    total_frames = sum(
        s["evaluated_frames"]
        for s in summaries
    )

    total_pairs = sum(
        s["num_adjacent_pairs"]
        for s in summaries
    )

    total_sum = sum(
        s["temporal_cosine_sum"]
        for s in summaries
    )

    total_sq_sum = sum(
        s["temporal_cosine_sq_sum"]
        for s in summaries
    )

    if total_pairs <= 0:
        raise RuntimeError(
            "No adjacent frame pairs"
        )

    # Primary dataset metric:
    # all adjacent frame pairs are equally weighted.
    global_mean = (
        total_sum / total_pairs
    )

    global_variance = (
        total_sq_sum / total_pairs
        - global_mean ** 2
    )

    global_variance = max(
        0.0,
        global_variance,
    )

    global_std = math.sqrt(
        global_variance
    )

    # Auxiliary video-macro result.
    video_means = np.asarray(
        [
            s["temporal_cosine_mean"]
            for s in summaries
        ],
        dtype=np.float64,
    )

    video_macro_mean = float(
        video_means.mean()
    )

    video_macro_std = float(
        video_means.std(ddof=0)
    )

    per_video_csv = (
        output_root
        / "per_video_metrics.csv"
    )

    fieldnames = [
        "video_id",
        "evaluated_frames",
        "num_adjacent_pairs",
        "temporal_cosine_mean",
        "temporal_cosine_std",
    ]

    with per_video_csv.open(
        "w",
        newline="",
    ) as f:
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
        "num_videos":
            len(summaries),

        "total_frames":
            total_frames,

        "total_adjacent_pairs":
            total_pairs,

        # Primary result
        "temporal_global_pair_weighted_mean":
            float(global_mean),

        "temporal_global_pair_weighted_std":
            float(global_std),

        # Auxiliary result
        "temporal_video_macro_mean":
            video_macro_mean,

        "temporal_video_macro_std":
            video_macro_std,

        "primary_metric":
            "temporal_global_pair_weighted_mean",

        "feature_extractor":
            "DINOv2 ViT-B/14",

        "feature_tensor":
            "raw 768D CLS f_global",

        "input_protocol":
            "full rendered RGB, unmasked",

        "aggregation_definition":
            (
                "Arithmetic mean of adjacent-frame "
                "DINOv2 cosine similarities over all "
                "adjacent frame pairs in the dataset."
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

    print("=" * 72)
    print("TEMPORAL DATASET SUMMARY")
    print("=" * 72)
    print("Videos              :", len(summaries))
    print("Frames              :", total_frames)
    print("Adjacent pairs      :", total_pairs)
    print(
        "Global temporal     :",
        f"{global_mean:.8f}",
    )
    print(
        "Global std          :",
        f"{global_std:.8f}",
    )
    print(
        "Video-macro mean    :",
        f"{video_macro_mean:.8f}",
    )
    print(
        "Video-macro std     :",
        f"{video_macro_std:.8f}",
    )
    print("Per-video CSV       :", per_video_csv)
    print("Dataset JSON        :", dataset_json)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Offline DINOv2 adjacent-frame "
            "temporal consistency evaluator"
        )
    )

    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--render_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
    )

    args = parser.parse_args()

    if args.aggregate:
        if args.manifest is None:
            raise RuntimeError(
                "--manifest required for --aggregate"
            )

        aggregate_dataset(
            manifest=args.manifest,
            output_root=args.output_root,
        )

        return

    if args.video_id is None:
        raise RuntimeError(
            "--video_id required"
        )

    if args.render_dir is None:
        raise RuntimeError(
            "--render_dir required"
        )

    evaluate_sequence(
        video_id=args.video_id,
        render_dir=args.render_dir,
        output_root=args.output_root,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        device=args.device,
    )


if __name__ == "__main__":
    main()

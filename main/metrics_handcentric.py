import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from omegaconf import OmegaConf

from dataset import TrackedData_infer
from utils.general_utils import ConfigDict, add_extra_cfgs
from submodules.lpipsPyTorch import LPIPS

# Reuse the original GUAVA evaluation definitions.
from metrics import psnr, ssim


REGION_TO_BOX = {
    "head": "head_box",
    "left_hand": "left_hand_box",
    "right_hand": "right_hand_box",
}

METRICS = ("psnr", "l1", "ssim", "lpips")


def load_saved_image(path):
    """
    Match original GUAVA main/metrics.py:
    PIL -> to_tensor -> RGB only.
    """
    image = Image.open(path)
    image = TF.to_tensor(image)[:3]

    if image.shape[0] != 3:
        raise RuntimeError(
            f"Expected RGB image at {path}, got {tuple(image.shape)}"
        )

    return image


def box_to_list(box):
    return [int(x) for x in box.detach().cpu().tolist()]


def box_to_string(box):
    return ",".join(str(x) for x in box_to_list(box))


def valid_box(box):
    left, right, top, bottom = box_to_list(box)
    return right > left and bottom > top


def compute_metrics(render, gt, lpips_model, device):
    render = render.to(device, non_blocking=True)
    gt = gt.to(device, non_blocking=True)

    with torch.inference_mode():
        return {
            "psnr": float(psnr(render, gt).mean().item()),
            "l1": float(torch.abs(render - gt).mean().item()),
            "ssim": float(ssim(render, gt).item()),
            "lpips": float(lpips_model(render, gt).mean().item()),
        }

def crop_region(dataset, image, box):
    """
    Reuse GUAVA's own ROI protocol:
    [left, right, top, bottom]
    -> crop
    -> resize to feature_part_size.
    """
    crop, _ = dataset._crop_image_part(image, box)

    if crop.numel() == 0:
        return None

    return crop.unsqueeze(0)


def mean_or_none(values):
    values = [x for x in values if x is not None]

    if not values:
        return None

    return float(sum(values) / len(values))


def save_debug_crop(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(
        image.detach().cpu().clamp(0, 1)
    ).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="GUAVA hand-centric self-reenactment evaluator"
    )

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--video_id", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug_frames", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=None)

    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    gt_dir = scene_dir / "gt"
    render_dir = scene_dir / "render"

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else scene_dir / "handcentric_metrics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.is_dir():
        raise RuntimeError(f"Missing GT directory: {gt_dir}")

    if not render_dir.is_dir():
        raise RuntimeError(f"Missing render directory: {render_dir}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(args.device)

    # ------------------------------------------------------------
    # Rebuild GUAVA tracked-data configuration.
    # ------------------------------------------------------------
    config_path = Path(args.model_path) / "config.yaml"

    if not config_path.exists():
        raise RuntimeError(f"Missing config: {config_path}")

    cfg = ConfigDict(model_config_path=str(config_path))
    cfg = add_extra_cfgs(cfg)

    # ConfigDict maintains both a normal dict representation
    # and an internal OmegaConf dot-access representation.
    # TrackedData reads cfg.DATASET.data_path, so update both.
    dataset_cfg = dict(cfg["DATASET"])
    dataset_cfg["data_path"] = args.data_path

    cfg.update("DATASET", dataset_cfg)

    # test_full=True is fine here because the exact evaluation subset is
    # recovered from the number of already-rendered PNGs.
    dataset = TrackedData_infer(
        cfg=cfg,
        split="test",
        device="cpu",
        test_full=True,
    )

    try:
        video_ids = list(dataset.videos_info.keys())

        if args.video_id is None:
            if len(video_ids) != 1:
                raise RuntimeError(
                    "Multiple videos in data_path; pass --video_id explicitly."
                )
            video_id = video_ids[0]
        else:
            video_id = args.video_id

        if video_id not in dataset.videos_info:
            raise RuntimeError(
                f"{video_id} not found. Available video IDs: {video_ids}"
            )

        # ------------------------------------------------------------
        # Match saved output PNGs.
        # ------------------------------------------------------------
        render_files = sorted(render_dir.glob("*.png"))

        if not render_files:
            raise RuntimeError(f"No render PNGs found in {render_dir}")

        actual_names = [x.name for x in render_files]
        expected_names = [
            f"{idx:05d}.png"
            for idx in range(len(render_files))
        ]

        if actual_names != expected_names:
            raise RuntimeError(
                "Render files are not contiguous 00000.png ... NNNNN.png."
            )

        for render_path in render_files:
            gt_path = gt_dir / render_path.name
            if not gt_path.exists():
                raise RuntimeError(f"Missing paired GT: {gt_path}")

        # GUAVA self rendering uses:
        #   frames = videos_info[video_id]["frames_keys"]
        #   for idx, frame in enumerate(frames[-test_num:])
        frames = list(dataset.videos_info[video_id]["frames_keys"])

        n_rendered = len(render_files)

        if n_rendered > len(frames):
            raise RuntimeError(
                f"{n_rendered} rendered frames > {len(frames)} tracked frames."
            )

        frame_keys = frames[-n_rendered:]

        if args.max_frames is not None:
            n = min(args.max_frames, n_rendered)
            render_files = render_files[:n]
            frame_keys = frame_keys[:n]

        n_eval = len(render_files)

        print("===================================================")
        print("GUAVA Hand-Centric Self Evaluation")
        print("===================================================")
        print("video_id          :", video_id)
        print("scene_dir         :", scene_dir)
        print("tracked frames    :", len(frames))
        print("rendered frames   :", n_rendered)
        print("evaluated frames  :", n_eval)
        print("image_size        :", dataset.image_size)
        print("ROI output size   :", dataset.feature_part_size)
        print("device            :", device)
        print("===================================================")

        lpips_model = LPIPS("vgg", "0.1").to(device)
        lpips_model.eval()

        regions = [
            "full",
            "head",
            "left_hand",
            "right_hand",
            "hand_combined",
        ]

        values = {
            region: {metric: [] for metric in METRICS}
            for region in regions
        }

        valid_counts = {region: 0 for region in regions}
        rows = []

        for idx, (render_path, frame_key) in enumerate(
            tqdm(
                zip(render_files, frame_keys),
                total=n_eval,
                desc="Evaluating",
            )
        ):
            gt_path = gt_dir / render_path.name

            render_image = load_saved_image(render_path)
            gt_image = load_saved_image(gt_path)

            if render_image.shape != gt_image.shape:
                raise RuntimeError(
                    f"Shape mismatch at {render_path.name}: "
                    f"{tuple(render_image.shape)} vs {tuple(gt_image.shape)}"
                )

            row = {
                "frame_index": idx,
                "image_name": render_path.name,
                "frame_key": str(frame_key),
            }

            # --------------------------------------------------------
            # FULL
            # --------------------------------------------------------
            full_result = compute_metrics(
                render_image.unsqueeze(0),
                gt_image.unsqueeze(0),
                lpips_model,
                device,
            )

            for metric, result in full_result.items():
                row[f"full_{metric}"] = result
                values["full"][metric].append(result)

            valid_counts["full"] += 1

            # --------------------------------------------------------
            # HEAD / LEFT HAND / RIGHT HAND
            # --------------------------------------------------------
            target_info = dataset._load_target_info(
                video_id,
                frame_key,
            )

            current_hands = {
                "left_hand": None,
                "right_hand": None,
            }

            for region, box_key in REGION_TO_BOX.items():
                box = (
                    target_info[box_key][0]
                    .detach()
                    .cpu()
                    .long()
                )

                row[f"{region}_box"] = box_to_string(box)

                if not valid_box(box):
                    row[f"{region}_valid"] = 0

                    for metric in METRICS:
                        row[f"{region}_{metric}"] = None

                    continue

                render_crop = crop_region(dataset, render_image, box)
                gt_crop = crop_region(dataset, gt_image, box)

                if render_crop is None or gt_crop is None:
                    row[f"{region}_valid"] = 0

                    for metric in METRICS:
                        row[f"{region}_{metric}"] = None

                    continue

                result = compute_metrics(
                    render_crop,
                    gt_crop,
                    lpips_model,
                    device,
                )

                row[f"{region}_valid"] = 1

                for metric, metric_value in result.items():
                    row[f"{region}_{metric}"] = metric_value
                    values[region][metric].append(metric_value)

                valid_counts[region] += 1

                if region in current_hands:
                    current_hands[region] = result

                if idx < args.debug_frames:
                    debug_dir = (
                        output_dir
                        / "debug_crops"
                        / f"{idx:05d}"
                    )

                    save_debug_crop(
                        debug_dir / f"{region}_gt.png",
                        gt_crop[0],
                    )

                    save_debug_crop(
                        debug_dir / f"{region}_render.png",
                        render_crop[0],
                    )

            # --------------------------------------------------------
            # COMBINED HAND
            #
            # Bilateral hand average.
            #
            # A frame contributes only when BOTH left- and right-hand
            # ROIs are valid. For each metric:
            #
            #   combined = (left + right) / 2
            # --------------------------------------------------------

            left_result = current_hands["left_hand"]
            right_result = current_hands["right_hand"]

            if left_result is None or right_result is None:

                row["hand_combined_valid"] = 0

                for metric in METRICS:
                    row[f"hand_combined_{metric}"] = None

            else:

                row["hand_combined_valid"] = 1

                for metric in METRICS:
                    combined = (
                        left_result[metric]
                        + right_result[metric]
                    ) / 2.0

                    row[f"hand_combined_{metric}"] = combined

                    values["hand_combined"][metric].append(
                        combined
                    )

                valid_counts["hand_combined"] += 1

            rows.append(row)

        # ------------------------------------------------------------
        # PER-FRAME CSV
        # ------------------------------------------------------------
        fieldnames = [
            "frame_index",
            "image_name",
            "frame_key",

            "full_psnr",
            "full_l1",
            "full_ssim",
            "full_lpips",

            "head_box",
            "head_valid",
            "head_psnr",
            "head_l1",
            "head_ssim",
            "head_lpips",

            "left_hand_box",
            "left_hand_valid",
            "left_hand_psnr",
            "left_hand_l1",
            "left_hand_ssim",
            "left_hand_lpips",

            "right_hand_box",
            "right_hand_valid",
            "right_hand_psnr",
            "right_hand_l1",
            "right_hand_ssim",
            "right_hand_lpips",

            "hand_combined_valid",
            "hand_combined_psnr",
            "hand_combined_l1",
            "hand_combined_ssim",
            "hand_combined_lpips",
        ]

        csv_path = output_dir / "per_frame_metrics.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # ------------------------------------------------------------
        # VIDEO SUMMARY
        # ------------------------------------------------------------
        summary = {
            "video_id": video_id,
            "data_path": args.data_path,
            "scene_dir": str(scene_dir),

            "protocol": {
                "full_metric_definition":
                    "Original GUAVA PSNR/SSIM/LPIPS",
                "lpips":
                    "VGG, version 0.1",
                "box_order":
                    "[left,right,top,bottom]",
                "roi_crop":
                    "TrackedData_infer._crop_image_part",
                "roi_output_size":
                    int(dataset.feature_part_size),
                "frame_mapping":
                    "frames_keys[-num_rendered:][render_index]",
                "combined_hand":
                    "Arithmetic mean of left- and right-hand metrics "
                    "for frames where both hand ROIs are valid; "
                    "then mean over valid frames.",
            },

            "frames": {
                "rendered": n_rendered,
                "evaluated": n_eval,
            },

            "regions": {},
        }

        for region in regions:
            summary["regions"][region] = {
                "valid_frames": int(valid_counts[region]),
                "psnr": mean_or_none(values[region]["psnr"]),
                "l1": mean_or_none(values[region]["l1"]),
                "ssim": mean_or_none(values[region]["ssim"]),
                "lpips": mean_or_none(values[region]["lpips"]),
            }

        summary_path = output_dir / "summary.json"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)

        print()
        print("================ RESULTS ================")

        for region in regions:
            x = summary["regions"][region]

            print(
                f"{region:15s} "
                f"N={x['valid_frames']:4d}  "
                f"PSNR={x['psnr']}  "
                f"L1={x['l1']}  "
                f"SSIM={x['ssim']}  "
                f"LPIPS={x['lpips']}"
            )

        print()
        print("CSV     :", csv_path)
        print("Summary :", summary_path)
        print("Debug   :", output_dir / "debug_crops")
        print("=========================================")

    finally:
        if hasattr(dataset, "_lmdb_engine"):
            dataset._lmdb_engine.close()


if __name__ == "__main__":
    main()

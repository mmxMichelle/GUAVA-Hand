#!/usr/bin/env python3

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


METHOD_ORDER = ["official", "baseline_ft", "c1", "c2", "full"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(xs):
    xs = [float(x) for x in xs]
    return {
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
        "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def load_api(repo):
    repo = str(Path(repo).resolve())

    sys.path.insert(0, repo)

    m = importlib.import_module("main.test")

    loaded = str(Path(m.__file__).resolve())
    if not loaded.startswith(repo + os.sep):
        raise RuntimeError(
            f"Wrong main.test imported: {loaded}"
        )

    model_mod = importlib.import_module(
        m.Ubody_Gaussian.__module__
    )

    model_file = str(Path(model_mod.__file__).resolve())

    if not model_file.startswith(repo + os.sep):
        raise RuntimeError(
            f"Wrong model code imported: {model_file}"
        )

    return m


def clean_hand_init_ehm(self, ehm=None):
    """
    C1/Full init_ehm without the temporary debug-only prints/asserts.

    Actual hand ownership computation is retained.
    """

    if ehm is None:
        raise RuntimeError(
            "Benchmark expects init_ehm(infer_model.ehm)."
        )

    self.ehm = ehm.to(self._smplx_xyz.device)
    self.smplx = self.ehm.smplx

    binding_vertices = self.smplx.faces_tensor[
        self._uv_binding_face.long()
    ]

    # Original SMPL-X hand semantic mask.
    num_smplx_vertices = int(
        self.smplx.hand_vertex_mask.shape[0]
    )

    # Augmented GUAVA topology may contain auxiliary teeth vertices.
    num_mesh_vertices = (
        int(self.smplx.faces_tensor.max().item()) + 1
    )

    if num_mesh_vertices < num_smplx_vertices:
        raise RuntimeError(
            "Unexpected augmented-mesh topology."
        )

    if num_mesh_vertices > num_smplx_vertices:
        expanded = torch.zeros(
            num_mesh_vertices,
            dtype=self.smplx.hand_vertex_mask.dtype,
            device=self.smplx.hand_vertex_mask.device,
        )

        expanded[:num_smplx_vertices] = (
            self.smplx.hand_vertex_mask
        )
    else:
        expanded = self.smplx.hand_vertex_mask

    if int(binding_vertices.max().item()) >= num_mesh_vertices:
        raise RuntimeError(
            "Binding vertex index out of range."
        )

    membership = expanded[
        binding_vertices.long()
    ]

    ownership = (
        membership * self._uv_face_bary
    ).sum(dim=-1)

    self.register_buffer(
        "_uv_hand_ownership",
        ownership.float(),
    )


def validate_and_patch_hand_init(
    Ubody_Gaussian,
    infer_model,
    meta_cfg,
    source_info,
):
    """
    First execute the original C1 init_ehm outside formal timing,
    then execute the cleaned version and require exact equality
    of the resulting UV hand ownership.

    Only after this test succeeds is the cleaned method used in
    the timed benchmark.
    """

    original_init = Ubody_Gaussian.init_ehm

    with torch.no_grad():

        vertex, uv, _ = infer_model(source_info)

        ref_avatar = Ubody_Gaussian(
            meta_cfg.MODEL,
            vertex,
            uv,
            pruning=True,
        )

        original_init(
            ref_avatar,
            infer_model.ehm,
        )

        ref = (
            ref_avatar
            ._uv_hand_ownership
            .detach()
            .clone()
        )

        clean_avatar = Ubody_Gaussian(
            meta_cfg.MODEL,
            vertex,
            uv,
            pruning=True,
        )

        clean_hand_init_ehm(
            clean_avatar,
            infer_model.ehm,
        )

        clean = (
            clean_avatar
            ._uv_hand_ownership
            .detach()
            .clone()
        )

        max_abs = float(
            (ref - clean)
            .abs()
            .max()
            .item()
        )

        if (
            ref.shape != clean.shape
            or not torch.allclose(
                ref,
                clean,
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise RuntimeError(
                "Clean hand-init mismatch; "
                f"max_abs={max_abs}"
            )

        del (
            vertex,
            uv,
            ref_avatar,
            clean_avatar,
            ref,
            clean,
        )

        gc.collect()
        torch.cuda.empty_cache()

    Ubody_Gaussian.init_ehm = clean_hand_init_ehm

    return {
        "enabled": True,
        "ownership_exact_match": True,
        "max_abs_diff": max_abs,
    }


def build_avatar(
    infer_model,
    Ubody_Gaussian,
    meta_cfg,
    source_info,
):
    """
    Formal reconstruction/setup boundary:

      source_info
        -> infer_model
        -> Ubody_Gaussian(pruning=True)
        -> init_ehm / hand ownership
        -> eval
        -> ready-to-render avatar
    """

    vertex, uv, _ = infer_model(source_info)

    avatar = Ubody_Gaussian(
        meta_cfg.MODEL,
        vertex,
        uv,
        pruning=True,
    )

    avatar.init_ehm(
        infer_model.ehm
    )

    avatar.eval()

    return avatar


def timed_ms(fn):
    torch.cuda.synchronize()

    t0 = time.perf_counter()

    out = fn()

    torch.cuda.synchronize()

    elapsed_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    return elapsed_ms, out


def benchmark(args):

    repo = Path(args.repo).resolve()
    model_path = Path(args.model_path).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    data_path = Path(args.data_path).resolve()
    output = Path(args.output).resolve()

    if args.config_path:
        config_path = Path(
            args.config_path
        ).resolve()
    else:
        config_path = (
            model_path / "config.yaml"
        )

    for p in [repo, data_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    for p in [checkpoint, config_path]:
        if not p.is_file():
            raise FileNotFoundError(p)

    api = load_api(repo)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device(
        f"cuda:{args.device}"
    )

    torch.cuda.set_device(device)

    try:
        api.lightning.fabric.seed_everything(
            args.seed
        )
    except Exception:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    meta_cfg = api.add_extra_cfgs(
        api.ConfigDict(
            model_config_path=str(
                config_path
            )
        )
    )

    meta_cfg = copy.deepcopy(meta_cfg)

    infer_model = (
        api.Ubody_Gaussian_inferer(
            meta_cfg.MODEL
        )
        .to(device)
        .eval()
    )

    render_model = (
        api.GaussianRenderer(
            meta_cfg.MODEL
        )
        .to(device)
        .eval()
    )

    # Exact parameter counts.
    trainable_params = sum(
        p.numel()
        for m in [
            infer_model,
            render_model,
        ]
        for p in m.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for m in [
            infer_model,
            render_model,
        ]
        for p in m.parameters()
    )

    hand_detected = any(
        "hand_local_pos_residual_head" in n
        for n, _ in
        infer_model.named_parameters()
    )

    if (
        bool(args.hand_arch)
        != bool(hand_detected)
    ):
        raise RuntimeError(
            "Architecture guard failed: "
            f"requested hand={args.hand_arch}, "
            f"detected={hand_detected}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    global_iter = state.get(
        "global_iter",
        None,
    )

    if global_iter is not None:
        global_iter = int(global_iter)

    a = infer_model.load_state_dict(
        state["model"],
        strict=False,
    )

    b = render_model.load_state_dict(
        state["render_model"],
        strict=False,
    )

    missing = (
        list(a.missing_keys)
        + [
            "render::" + x
            for x in b.missing_keys
        ]
    )

    unexpected = (
        list(a.unexpected_keys)
        + [
            "render::" + x
            for x in b.unexpected_keys
        ]
    )

    # Match GUAVA's evaluation behaviour: checkpoints are loaded
    # with strict=False. Missing current-model keys are unsafe because
    # they would leave part of the evaluated model uninitialised.
    # Checkpoint-only legacy keys are harmless and are retained only
    # for audit/reproducibility.
    if missing:
        raise RuntimeError(
            "Checkpoint is missing current-model keys\n"
            f"Missing={missing}\n"
            f"Unexpected={unexpected}"
        )

    if unexpected:
        print(
            f"[WARN] strict=False: ignoring "
            f"{len(unexpected)} checkpoint-only keys"
        )
        for key in unexpected:
            print(f"  [unexpected] {key}")

    del state

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    api.OmegaConf.set_readonly(
        meta_cfg["DATASET"],
        False,
    )

    meta_cfg["DATASET"][
        "data_path"
    ] = str(data_path)

    dataset = api.TrackedData_infer(
        cfg=meta_cfg,
        split="test",
        device=str(device),
        test_full=True,
    )

    video_ids = list(
        dataset.videos_info.keys()
    )

    if len(video_ids) != 1:
        raise RuntimeError(
            f"Expected one video, "
            f"got {video_ids}"
        )

    video_id = video_ids[0]

    source_info = (
        dataset._load_source_info(
            video_id
        )
    )

    # --------------------------------------------------------
    # Validate + remove C1 temporary debug diagnostics.
    # --------------------------------------------------------

    hand_patch = {
        "enabled": False
    }

    if args.hand_arch:
        hand_patch = (
            validate_and_patch_hand_init(
                api.Ubody_Gaussian,
                infer_model,
                meta_cfg,
                source_info,
            )
        )

    # --------------------------------------------------------
    # Fixed rendering frames
    # --------------------------------------------------------

    frames = list(
        dataset.videos_info[
            video_id
        ]["frames_keys"]
    )

    test_num = int(
        dataset.testing_split[
            video_id
        ]
    )

    eval_frames = frames[
        -test_num:
    ]

    if not eval_frames:
        raise RuntimeError(
            "No evaluation frames"
        )

    n_render = min(
        args.render_frames,
        len(eval_frames),
    )

    indices = np.linspace(
        0,
        len(eval_frames) - 1,
        n_render,
        dtype=np.int64,
    )

    render_frames = [
        eval_frames[int(i)]
        for i in indices
    ]

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    with torch.no_grad():

        # Reconstruction warm-up.
        warm_avatar = None

        for _ in range(
            args.warmup_recon
        ):
            if warm_avatar is not None:
                del warm_avatar

            warm_avatar = build_avatar(
                infer_model,
                api.Ubody_Gaussian,
                meta_cfg,
                source_info,
            )

        torch.cuda.synchronize()

        # Rendering warm-up.
        for frame in render_frames[
            :args.warmup_frames
        ]:
            target = (
                dataset._load_target_info(
                    video_id,
                    frame,
                )
            )

            deform = warm_avatar(
                target
            )

            rendered = render_model(
                deform,
                target[
                    "render_cam_params"
                ],
                bg=0.0,
            )

            del (
                target,
                deform,
                rendered,
            )

        torch.cuda.synchronize()

        del warm_avatar

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Peak memory starts with loaded model + source resident.
        torch.cuda.reset_peak_memory_stats(
            device
        )

        # --------------------------------------------
        # Reconstruction / avatar setup latency
        # --------------------------------------------

        recon_ms = []
        avatar = None

        for _ in range(
            args.recon_repeats
        ):

            if avatar is not None:
                del avatar
                gc.collect()

            ms, avatar = timed_ms(
                lambda: build_avatar(
                    infer_model,
                    api.Ubody_Gaussian,
                    meta_cfg,
                    source_info,
                )
            )

            recon_ms.append(ms)

        # --------------------------------------------
        # Reenactment rendering FPS
        # --------------------------------------------

        render_ms = []

        for frame in render_frames:

            # Loading excluded from timing.
            target = (
                dataset._load_target_info(
                    video_id,
                    frame,
                )
            )

            def do_render():

                deform = avatar(
                    target
                )

                result = render_model(
                    deform,
                    target[
                        "render_cam_params"
                    ],
                    bg=0.0,
                )

                return deform, result

            ms, out = timed_ms(
                do_render
            )

            render_ms.append(ms)

            deform, result = out

            del (
                target,
                deform,
                result,
            )

        torch.cuda.synchronize()

        peak_alloc = int(
            torch.cuda.max_memory_allocated(
                device
            )
        )

        peak_reserved = int(
            torch.cuda.max_memory_reserved(
                device
            )
        )

    render_fps = (
        1000.0
        * len(render_ms)
        / sum(render_ms)
    )

    # --------------------------------------------------------
    # Save per-video result
    # --------------------------------------------------------

    result = {
        "schema_version": 1,

        "method": args.method,

        "host": socket.gethostname(),

        "repo": str(repo),

        "config_path": str(
            config_path
        ),

        "model_path": str(
            model_path
        ),

        "checkpoint": str(
            checkpoint
        ),

        "checkpoint_sha256":
            sha256_file(
                checkpoint
            ),

        "checkpoint_global_iter":
            global_iter,

        "checkpoint_load": {
            "strict": False,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },

        "data_path": str(
            data_path
        ),

        "video_id": video_id,

        "device_name":
            torch.cuda.get_device_name(
                device
            ),

        "seed": args.seed,

        "parameter_count": {
            "trainable":
                int(trainable_params),

            "total":
                int(total_params),

            "total_million":
                total_params
                / 1_000_000.0,
        },

        "architecture": {
            "hand_arch_requested":
                bool(args.hand_arch),

            "hand_residual_detected":
                bool(hand_detected),

            "benchmark_clean_hand_init":
                hand_patch,
        },

        "protocol": {
            "reconstruction":
                "source_info -> infer_model -> "
                "Ubody_Gaussian(pruning=True) -> "
                "init_ehm -> eval",

            "rendering":
                "Ubody_Gaussian(target_info) + "
                "GaussianRenderer; target loading "
                "and file I/O excluded",

            "warmup_recon":
                args.warmup_recon,

            "recon_repeats":
                args.recon_repeats,

            "warmup_frames":
                args.warmup_frames,

            "timed_render_frames":
                len(render_ms),

            "frame_sampling":
                "evenly spaced over the same "
                "self-reenactment test-frame range",

            "timing":
                "torch.cuda.synchronize + "
                "time.perf_counter wall-clock",
        },

        "reconstruction_ms":
            recon_ms,

        "reconstruction_summary_ms":
            stats(recon_ms),

        "render_frame_ms":
            render_ms,

        "render_summary_ms":
            stats(render_ms),

        "render_fps":
            float(render_fps),

        "peak_cuda_allocated_bytes":
            peak_alloc,

        "peak_cuda_allocated_mib":
            peak_alloc
            / (1024.0 ** 2),

        "peak_cuda_reserved_bytes":
            peak_reserved,

        "peak_cuda_reserved_mib":
            peak_reserved
            / (1024.0 ** 2),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        json.dumps(
            {
                "method":
                    args.method,

                "video_id":
                    video_id,

                "params":
                    total_params,

                "recon_median_ms":
                    result[
                        "reconstruction_summary_ms"
                    ]["median"],

                "render_fps":
                    render_fps,

                "peak_allocated_mib":
                    result[
                        "peak_cuda_allocated_mib"
                    ],

                "ownership_exact_match":
                    hand_patch.get(
                        "ownership_exact_match",
                        None,
                    ),

                "output":
                    str(output),
            },
            indent=2,
        )
    )


def aggregate(args):

    input_root = Path(
        args.input_root
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    run_files = sorted(
        input_root.rglob(
            "*.json"
        )
    )

    if not run_files:
        raise RuntimeError(
            f"No run JSONs under "
            f"{input_root}"
        )

    runs = []

    for p in run_files:

        row = json.loads(
            p.read_text()
        )

        row["_path"] = str(p)

        runs.append(row)

    grouped = {
        m: []
        for m in METHOD_ORDER
    }

    for r in runs:
        if r["method"] not in grouped:
            raise RuntimeError(
                "Unexpected method "
                f"{r['method']}"
            )

        grouped[
            r["method"]
        ].append(r)

    ref_ids = None
    all_gpu_models = set()
    results = []

    for method in METHOD_ORDER:

        rows = grouped[
            method
        ]

        if (
            args.expected_videos
            is not None
            and len(rows)
            != args.expected_videos
        ):
            raise RuntimeError(
                f"{method}: expected "
                f"{args.expected_videos}, "
                f"found {len(rows)}"
            )

        ids = [
            r["video_id"]
            for r in rows
        ]

        if len(ids) != len(set(ids)):
            raise RuntimeError(
                f"{method}: "
                "duplicate video IDs"
            )

        if ref_ids is None:
            ref_ids = set(ids)
        elif set(ids) != ref_ids:
            raise RuntimeError(
                f"{method}: "
                "video cohort mismatch"
            )

        params = {
            int(
                r[
                    "parameter_count"
                ]["total"]
            )
            for r in rows
        }

        if len(params) != 1:
            raise RuntimeError(
                f"{method}: "
                f"inconsistent params "
                f"{params}"
            )

        total_params = (
            params.pop()
        )

        gpu_models = {
            r["device_name"]
            for r in rows
        }

        all_gpu_models.update(
            gpu_models
        )

        if len(gpu_models) != 1:
            raise RuntimeError(
                f"{method}: "
                f"mixed GPU models "
                f"{gpu_models}"
            )

        recon = [
            x
            for r in rows
            for x in
            r["reconstruction_ms"]
        ]

        render_ms = [
            x
            for r in rows
            for x in
            r["render_frame_ms"]
        ]

        video_fps = [
            float(r["render_fps"])
            for r in rows
        ]

        results.append({
            "method":
                method,

            "num_videos":
                len(rows),

            "gpu_model":
                next(
                    iter(gpu_models)
                ),

            "total_params":
                total_params,

            "params_million":
                total_params
                / 1_000_000.0,

            "reconstruction_ms_mean":
                statistics.fmean(
                    recon
                ),

            "reconstruction_ms_median":
                statistics.median(
                    recon
                ),

            "reconstruction_ms_std":
                statistics.pstdev(
                    recon
                )
                if len(recon) > 1
                else 0.0,

            "timed_render_frames":
                len(render_ms),

            "render_fps_global":
                1000.0
                * len(render_ms)
                / sum(render_ms),

            "render_fps_video_macro_mean":
                statistics.fmean(
                    video_fps
                ),

            "render_fps_video_macro_std":
                statistics.pstdev(
                    video_fps
                )
                if len(video_fps) > 1
                else 0.0,

            "peak_cuda_allocated_mib_max":
                max(
                    float(
                        r[
                            "peak_cuda_allocated_mib"
                        ]
                    )
                    for r in rows
                ),
        })

    if len(all_gpu_models) != 1:
        raise RuntimeError(
            "Formal comparison mixed "
            f"GPU models: "
            f"{all_gpu_models}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_root
        / "results.csv"
    ).open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(results)

    metadata = {
        "schema_version": 1,

        "experiment":
            "GUAVA formal efficiency benchmark",

        "methods":
            METHOD_ORDER,

        "num_videos":
            len(ref_ids),

        "video_ids":
            sorted(ref_ids),

        "gpu_model":
            next(
                iter(
                    all_gpu_models
                )
            ),

        "protocol": {
            "reconstruction":
                "synchronized source-to-ready "
                "avatar setup",

            "rendering":
                "synchronized deformation + "
                "Gaussian rasterization",

            "params":
                "exact numel over "
                "infer_model + render_model",

            "peak_memory":
                "max torch.cuda allocated memory "
                "with model/source resident",

            "hand_debug_policy":
                "C1/Full temporary diagnostics "
                "excluded; exact ownership "
                "equality validated before timing",
        },

        "checkpoint_sha256": {
            m: sorted({
                r[
                    "checkpoint_sha256"
                ]
                for r in grouped[m]
            })
            for m in METHOD_ORDER
        },

        "run_files": [
            r["_path"]
            for r in runs
        ],
    }

    (
        output_root
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    (
        output_root
        / "README.md"
    ).write_text(
        "# GUAVA Formal Efficiency Benchmark\n\n"
        "Compares Official GUAVA, Baseline-FT, "
        "C1, C2 and Full using the same fixed "
        "How2Sign test sequences.\n\n"
        "Primary metrics: exact total parameters, "
        "source-to-ready reconstruction/setup "
        "latency, reenactment FPS, and peak CUDA "
        "allocated memory. Dataset/checkpoint "
        "loading and all disk I/O are excluded.\n"
    )

    (
        output_root
        / "notes.md"
    ).write_text(
        "# Notes\n\n"
        "- `render_fps_global` is total timed "
        "frames divided by total synchronized "
        "deformation+render time.\n"
        "- Reconstruction summaries pool all "
        "repeats across the fixed video cohort.\n"
        "- Peak memory is the maximum "
        "allocated-memory peak across the cohort.\n"
        "- C1/Full benchmark-time hand init "
        "removes temporary debug-only diagnostics "
        "and verifies exact ownership equality "
        "before timing.\n"
    )

    print(
        "[PASS] wrote "
        f"{output_root / 'results.csv'}"
    )

    for r in results:
        print(
            f"{r['method']:<12} "
            f"params="
            f"{r['params_million']:.6f}M "
            f"recon_med="
            f"{r['reconstruction_ms_median']:.3f}ms "
            f"fps="
            f"{r['render_fps_global']:.3f} "
            f"peak="
            f"{r['peak_cuda_allocated_mib_max']:.1f}MiB"
        )


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--aggregate",
        action="store_true",
    )

    p.add_argument(
        "--method",
        choices=METHOD_ORDER,
    )

    p.add_argument("--repo")
    p.add_argument("--model_path")
    p.add_argument("--config_path")
    p.add_argument("--checkpoint")
    p.add_argument("--data_path")
    p.add_argument("--output")

    p.add_argument(
        "--device",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=10,
    )

    p.add_argument(
        "--hand_arch",
        action="store_true",
    )

    p.add_argument(
        "--warmup_recon",
        type=int,
        default=2,
    )

    p.add_argument(
        "--recon_repeats",
        type=int,
        default=5,
    )

    p.add_argument(
        "--warmup_frames",
        type=int,
        default=10,
    )

    p.add_argument(
        "--render_frames",
        type=int,
        default=200,
    )

    p.add_argument("--input_root")
    p.add_argument("--output_root")

    p.add_argument(
        "--expected_videos",
        type=int,
    )

    a = p.parse_args()

    if a.aggregate:
        if (
            not a.input_root
            or not a.output_root
        ):
            p.error(
                "--aggregate requires "
                "--input_root and "
                "--output_root"
            )

    else:
        required = [
            "method",
            "repo",
            "model_path",
            "checkpoint",
            "data_path",
            "output",
        ]

        missing = [
            x
            for x in required
            if not getattr(a, x)
        ]

        if missing:
            p.error(
                "missing: "
                + ", ".join(missing)
            )

    return a


if __name__ == "__main__":

    args = parse_args()

    if args.aggregate:
        aggregate(args)
    else:
        benchmark(args)

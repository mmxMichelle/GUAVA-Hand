import os
import json
import glob
import argparse
import statistics


def main(root, output):
    files = sorted(
        glob.glob(
            os.path.join(root, "**", "speed_info.json"),
            recursive=True
        )
    )

    if not files:
        raise RuntimeError(
            f"No speed_info.json found under: {root}"
        )

    rows = []

    for path in files:
        with open(path, "r") as f:
            data = json.load(f)

        if "render_speed (fps)" not in data:
            print(f"[SKIP] No render_speed in {path}")
            continue

        fps = float(data["render_speed (fps)"])

        rows.append({
            "path": path,
            "fps": fps,
            "infer_time_ms": data.get("infer_time (ms)")
        })

    if not rows:
        raise RuntimeError(
            "No valid render_speed (fps) values found."
        )

    fps_values = [r["fps"] for r in rows]

    result = {
        "num_sequences": len(rows),

        # Macro average of the original GUAVA
        # per-video render_speed values.
        "mean_render_fps": statistics.mean(fps_values),

        "std_render_fps": (
            statistics.stdev(fps_values)
            if len(fps_values) > 1
            else 0.0
        ),

        "median_render_fps": statistics.median(fps_values),

        "min_render_fps": min(fps_values),
        "max_render_fps": max(fps_values),

        "per_sequence": rows,
    }

    os.makedirs(
        os.path.dirname(output) or ".",
        exist_ok=True
    )

    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print("========================================")
    print("GUAVA-compatible FPS aggregation")
    print("========================================")
    print("Sequences :", result["num_sequences"])
    print(
        "Mean FPS  :",
        f'{result["mean_render_fps"]:.6f}'
    )
    print(
        "Std FPS   :",
        f'{result["std_render_fps"]:.6f}'
    )
    print(
        "Median FPS:",
        f'{result["median_render_fps"]:.6f}'
    )
    print(
        "Range     :",
        f'{result["min_render_fps"]:.6f}',
        "-",
        f'{result["max_render_fps"]:.6f}'
    )
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        help="Root containing per-video speed_info.json files"
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    main(args.root, args.output)

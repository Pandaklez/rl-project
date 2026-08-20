"""
Side-by-side TensorBoard scalars for two training runs.

Reads the event files directly rather than eyeballing the web UI, so a
comparison can be quoted with numbers instead of described. Prints, per tag,
the first and last windowed mean and the trend between them — which is the
shape that matters here, because the (B) failure was diagnosed from
monotonicity rather than from any single value.

    python -m scripts.compare_runs checkpoints/transfix/frozen checkpoints/transfix/uv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    "Reward / Instantaneous reward (mean)",
    "Reprojection / improvement over lifted (px)",
    "Reprojection / error corrected (px)",
    "Reprojection / error lifted (px)",
    "Reward / reprojection",
    "Reward / smoothness",
    "Loss / Policy loss",
    "Loss / Value loss",
    "Policy / Standard deviation",
    "Learning / Learning rate",
    "Translation / du |abs| (bbox heights)",
    "Translation / dv |abs| (bbox heights)",
    "Translation / dlog_tz",
    "pose/img_improvement_px",
    "pose/img_err_corrected_px",
]


def load(run_dir: str) -> dict:
    paths = sorted(Path(run_dir).glob("*/events.out.tfevents.*"))
    if not paths:
        paths = sorted(Path(run_dir).glob("events.out.tfevents.*"))
    if not paths:
        raise SystemExit(f"no event file under {run_dir}")
    ea = EventAccumulator(str(paths[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: np.array([(s.step, s.value) for s in ea.Scalars(t)])
            for t in ea.Tags()["scalars"]}


def window(series: np.ndarray, frac: float = 0.2, head: bool = True) -> float:
    n = max(1, int(len(series) * frac))
    return float(series[:n, 1].mean() if head else series[-n:, 1].mean())


def fmt(v) -> str:
    return "     —" if v is None else f"{v:+9.4g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=2)
    ap.add_argument("--names", nargs=2, default=None)
    ap.add_argument("--max_step", type=float, default=None,
                    help="truncate both runs to this timestep. Without it, runs "
                         "of different lengths are not comparable — a 'last 20%%' "
                         "window means a different point in training for each.")
    args = ap.parse_args()

    names = args.names or [Path(r).name for r in args.runs]
    data = [load(r) for r in args.runs]

    # Match the ranges. Comparing the tail of a long run against the tail of a
    # short one compares two different points in training, not two treatments.
    cap = args.max_step
    if cap is None:
        cap = min(max((v[-1, 0] for v in d.values()), default=0) for d in data)
    data = [{t: v[v[:, 0] <= cap] for t, v in d.items() if len(v[v[:, 0] <= cap])}
            for d in data]
    print(f"\n[both runs truncated to {cap:,.0f} timesteps]")

    steps = [max((v[-1, 0] for v in d.values()), default=0) for d in data]
    print(f"\n{'':46s}  {names[0]:^24s}  {names[1]:^24s}")
    print(f"{'':46s}  {'first20% -> last20%':^24s}  {'first20% -> last20%':^24s}")
    print("-" * 100)
    for tag in TAGS:
        row = []
        for d in data:
            s = d.get(tag)
            row.append((None, None) if s is None or not len(s)
                       else (window(s, head=True), window(s, head=False)))
        if all(a is None for a, _ in row):
            continue
        cells = "  ".join(f"{fmt(a)} ->{fmt(b)}" for a, b in row)
        print(f"{tag:46s}  {cells}")
    print("-" * 100)
    print(f"{'timesteps':46s}  {steps[0]:^24,}  {steps[1]:^24,}")


if __name__ == "__main__":
    main()

"""
Aggregate the 3-variant x 3-seed sweep into the tables `report.md` reports.

Three tables, selected with `--table`:

  pampjpe   PA-MPJPE per seed, plus the delta against the lifted pose **paired
            per clip-camera** and a one-sample t-test on those deltas. Reads the
            JSON dumps written by `src/evaluate.py --dump_scores`, so run that
            for every checkpoint first (see the report's reproducibility note).
            Pairing matters: `pa_mpjpe_lifted_raw` averages 187 clips while the
            corrected score averages 374 clip-cameras, and although those happen
            to agree to three decimals here, the paired delta is correct by
            construction rather than by coincidence.

  rollout   Training-rollout reprojection, read straight from each run's
            TensorBoard event file. Averaged over the final `--window`
            timesteps rather than read at the last point, because the per-update
            scalar is noisy enough that a single sample is not representative.

  config    Observation/action widths and step counts, read from the checkpoint
            configs — the provenance of the Configuration table.

  diagnostics  The row-oriented Training diagnostics table: the `rollout`
            quantities plus smoothness and the (A) lifted baseline column, as
            3-seed mean ± sd. Pass `--heldout` with the JSON from
            `scripts/heldout_eval.py --dump` to fill the held-out row too;
            without it that row reads "not measured".

Usage:
    python scripts/sweep_stats.py --table rollout
    python scripts/sweep_stats.py --table pampjpe --scores_dir eval_scores
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

POSE_DIM = 156

VARIANTS = {"frozen": "(B1) frozen", "uv": "(B2) du,dv", "notrans": "(B3) pose-only"}

TAGS = {
    "err_corr": "Reprojection / error corrected (px)",
    "err_lift": "Reprojection / error lifted (px)",
    "improve":  "Reprojection / improvement over lifted (px)",
}


def _events(run_dir: Path) -> str:
    hits = glob.glob(str(run_dir / "*" / "events*"))
    if not hits:
        raise FileNotFoundError(f"no TensorBoard event file under {run_dir}")
    return hits[0]


def table_rollout(sweep: Path, seeds: list[int], window: int,
                  runs: list[str] | None = None) -> None:
    """`runs` overrides the `<variant>_s<seed>` naming, for run sets that do not
    use it — e.g. `--sweep_dir checkpoints/biasfix --runs frozen uv notrans`,
    the single-seed runs behind the Training diagnostics table."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    vals: dict = {}
    last_common = None
    # Run sets differ in naming: the sweep uses "<variant>_s<seed>", the
    # single-seed biasfix runs are just "<variant>". Seed 0 stands in for the
    # latter so the rest of the function is shape-agnostic.
    def _split(r: str) -> tuple[str, int]:
        return (r.split("_s")[0], int(r.split("_s")[1])) if "_s" in r else (r, 0)

    pairs = ([_split(r) for r in runs] if runs
             else [(v, s) for v in VARIANTS for s in seeds])
    name = {(v, s): (runs[i] if runs else f"{v}_s{s}")
            for i, (v, s) in enumerate(pairs)}
    for v, s in pairs:
            run = sweep / name[(v, s)]
            if not run.exists():
                continue
            ea = EventAccumulator(_events(run), size_guidance={"scalars": 0})
            ea.Reload()
            series = {k: np.array([(x.step, x.value) for x in ea.Scalars(t)])
                      for k, t in TAGS.items()}
            vals[(v, s)] = series
            end = series["improve"][-1, 0]
            last_common = end if last_common is None else min(last_common, end)

    how = (f"read at the last step common to all runs ({int(last_common):,})"
           if window == 0 else
           f"averaged over the final {window:,} timesteps, ending at the last "
           f"step common to all runs ({int(last_common):,})")
    print(f"# {how}\n")
    print("| variant | error corrected (px) | improvement over lifted (px) | "
          + " / ".join(f"s{s}" for s in seeds) + " |")
    print("|---|---|---|---|")
    seeds = sorted({s for _, s in pairs})
    for v, label in VARIANTS.items():
        def avg(s, key):
            a = vals[(v, s)][key]
            if window == 0:
                # Point read: the last sample at or before the common endpoint.
                # This is what "read at matched N timesteps" means; the windowed
                # form exists because the per-update scalar is noisy.
                return a[a[:, 0] <= last_common][-1, 1]
            m = a[(a[:, 0] >= last_common - window) & (a[:, 0] <= last_common), 1]
            return m.mean()
        have = [s for s in seeds if (v, s) in vals]
        if not have:
            continue
        ec = [avg(s, "err_corr") for s in have]
        im = [avg(s, "improve") for s in have]
        sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else float("nan")
        print(f"| {label} | {np.mean(ec):.2f} ± {sd(ec):.2f} | "
              f"**{np.mean(im):+.3f} ± {sd(im):.3f}** | "
              + " / ".join(f"{x:+.3f}" for x in im) + " |")

    print("\n# lifted error per seed (identical across variants => same clips)")
    for s in seeds:
        have = [(v, s) for v in VARIANTS if (v, s) in vals]
        if not have:
            continue
        def _lift(k):
            a = vals[k]["err_lift"]
            if window == 0:
                return a[a[:, 0] <= last_common][-1, 1]
            return a[(a[:, 0] >= last_common - window) & (a[:, 0] <= last_common), 1].mean()
        e = {round(float(_lift(k)), 4) for k in have}
        print(f"  s{s}: {sorted(e)}" + ("" if len(e) == 1 else "   <- MISMATCH"))


def table_pampjpe(scores_dir: Path, seeds: list[int]) -> None:
    runs = {}
    for v in VARIANTS:
        for s in seeds:
            p = scores_dir / f"{v}_s{s}.json"
            if p.exists():
                runs[(v, s)] = json.load(open(p))
    if not runs:
        raise SystemExit(f"no dumps in {scores_dir}; run src/evaluate.py --dump_scores first")

    any_run = next(iter(runs.values()))
    lifted = dict(zip(any_run["per_cam_keys_lifted"], any_run["per_cam_lifted"]))
    base_374 = np.mean(list(lifted.values())) * 1000
    print(f"# lifted baseline: {any_run['pa_mpjpe_lifted_raw']*1000:.3f} mm over "
          f"{any_run['n_clips_baseline']} clips | {base_374:.3f} mm over "
          f"{len(lifted)} clip-cameras\n")

    print("| variant | " + " | ".join(f"s{s}" for s in seeds)
          + " | mean ± sd | Δ vs lifted | clip-cameras improved |")
    print("|---" * (len(seeds) + 4) + "|")
    print(f"| (A) lifted baseline | " + " | ".join(f"{base_374:.3f}" for _ in seeds)
          + f" | **{base_374:.3f}** | 0 *(by definition)* | — |")

    for v, label in VARIANTS.items():
        have = [s for s in seeds if (v, s) in runs]
        if not have:
            continue
        means, deltas = [], []
        for s in have:
            d = runs[(v, s)]
            corr = np.array(d["per_clip_corrected"])
            base = np.array([lifted[k] for k in d["per_clip_keys"]])  # paired
            means.append(corr.mean() * 1000)
            deltas.append((corr - base) * 1000)
        allde = np.concatenate(deltas)
        t = allde.mean() / (allde.std(ddof=1) / np.sqrt(allde.size))
        per_seed = [d.mean() for d in deltas]
        sd = lambda x: np.std(x, ddof=1) if len(x) > 1 else float("nan")
        cells = " | ".join(f"{m:.3f}" for m in means)
        print(f"| {label} | {cells} | {np.mean(means):.3f} ± {sd(means):.3f} | "
              f"{np.mean(per_seed):+.3f} ± {sd(per_seed):.3f} | "
              f"{(allde < 0).mean()*100:.1f}% |")
        print(f"|   ^ paired t vs 0 over {allde.size:,} clip-camera deltas: "
              f"t = {t:+.1f} |" + " |" * (len(seeds) + 3))


def lifted_smoothness(h5_path: str, split: str = "train", sigma: float = 0.5) -> float:
    """
    The (A) column of the smoothness row: `exp(-a/sigma^2)` on the **untouched
    lifted trajectory**, where `a` is the mean squared acceleration.

    This is the baseline the policy's own smoothness is measured against, and it
    is a pure function of the data — no policy, no rollout — so it is computed
    here rather than read from a run. `GymMoviEnv._lifted_smoothness` computes
    exactly this per step and the env subtracts it from the reward, but it is
    never logged as a scalar, which is why it was missing from the table.

    Replicates the env frame-for-frame: `t` runs 0..T-2 (the env scores before
    incrementing, and truncates at T-1) and the two lookbacks are clamped at 0,
    so t=0 scores a zero acceleration and t=1 a first difference. Frame-weighted,
    matching how skrl averages a per-step scalar over a rollout.
    """
    import h5py

    # One source of truth: check the vectorised form against src.rewards on the
    # first clip. Skipped rather than fatal if src is not importable (the other
    # tables in this file need only numpy + tensorboard).
    try:
        from src.rewards import smoothness_reward
    except Exception:
        smoothness_reward = None

    total, n = 0.0, 0
    with h5py.File(h5_path, "r") as f:
        for clip in f[split].keys():
            grp = f[split][clip]
            for cam in ("pg1", "pg2"):
                if cam not in grp:
                    continue
                p = grp[cam]["poses"][:].astype(np.float32).reshape(-1, POSE_DIM)
                T = p.shape[0]
                if T < 2:
                    continue
                t = np.arange(T)
                acc = p[t] - 2.0 * p[np.maximum(t - 1, 0)] + p[np.maximum(t - 2, 0)]
                r = np.exp(-(acc * acc).reshape(T, -1).mean(1) / (sigma * sigma))[:T - 1]
                if smoothness_reward is not None:
                    for probe in (0, 1, min(7, T - 2)):
                        ref = smoothness_reward(p[probe], p[max(probe - 1, 0)],
                                                p[max(probe - 2, 0)], sigma)
                        assert abs(ref - r[probe]) < 1e-6, "vectorised form drifted"
                    smoothness_reward = None      # one clip is enough
                total += float(r.sum())
                n += r.size
    if not n:
        raise ValueError(f"no lifted poses under {h5_path}:{split}")
    return total / n


def _series(sweep: Path, variant: str, seeds: list[int], tag: str,
            window: int) -> list[float]:
    """Per-seed value of one scalar tag, averaged over the final `window`
    timesteps of each run (see table_rollout for why averaged, not point-read)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = []
    for sd in seeds:
        run = sweep / f"{variant}_s{sd}"
        if not run.exists():
            continue
        ea = EventAccumulator(_events(run), size_guidance={"scalars": 0})
        ea.Reload()
        a = np.array([(x.step, x.value) for x in ea.Scalars(tag)])
        end = a[-1, 0]
        out.append(float(a[(a[:, 0] >= end - window) & (a[:, 0] <= end), 1].mean()))
    return out


def _ms(xs: list[float], fmt: str = "{:.3f}") -> str:
    """mean ± sd, or just the mean when there is only one sample."""
    if not xs:
        return "—"
    m = fmt.format(np.mean(xs))
    if len(xs) < 2:
        return m
    return f"{m} ± {fmt.format(np.std(xs, ddof=1)).lstrip('+')}"


def table_diagnostics(sweep: Path, seeds: list[int], window: int,
                      heldout: Path | None, h5_path: str | None = None) -> None:
    cols = list(VARIANTS)

    corr = {v: _series(sweep, v, seeds, TAGS["err_corr"], window) for v in cols}
    lift = {v: _series(sweep, v, seeds, TAGS["err_lift"], window) for v in cols}
    impr = {v: _series(sweep, v, seeds, TAGS["improve"], window) for v in cols}
    smooth = {v: _series(sweep, v, seeds, "Reward / smoothness", window) for v in cols}

    # (A) is the untouched lifted pose, scored on the same rollouts. It is the
    # same number in every column by construction, so any one of them serves.
    base = lift[cols[0]]

    ho = {}
    if heldout and heldout.exists():
        raw = json.load(open(heldout))
        for v in cols:
            ho[v] = [raw[f"{v}_s{sd}"]["pose/img_improvement_px"]
                     for sd in seeds if f"{v}_s{sd}" in raw]

    print(f"# {len(seeds)} seeds, mean ± sd, averaged over the final "
          f"{window:,} timesteps of each run\n")
    hdr = "| Metric | Unit | (A) | " + " | ".join(VARIANTS[v] for v in cols) + " | (C) |"
    print(hdr)
    print("|---" * (len(cols) + 4) + "|")

    def row(label, unit, a_cell, cells):
        print(f"| {label} | {unit} | {a_cell} | " + " | ".join(cells) + " | |")

    row("Reprojection error, corrected", "px, lower better",
        _ms(base, "{:.2f}"), [_ms(corr[v], "{:.2f}") for v in cols])
    row("Reprojection error, lifted (same rollout)", "px",
        _ms(base, "{:.2f}"), [_ms(lift[v], "{:.2f}") for v in cols])
    row("Improvement over lifted, train rollout", "px, higher better",
        "0 *(by definition)*", [f"**{_ms(impr[v], '{:+.3f}')}**" for v in cols])
    row("Improvement over lifted, held-out val", "px, higher better",
        "0 *(by definition)*",
        [f"**{_ms(ho[v], '{:+.3f}')}**" if ho.get(v) else "not measured"
         for v in cols])
    a_smooth = ("—" if h5_path is None
                else f"{lifted_smoothness(h5_path):.4f}")
    row("Smoothness reward `exp(-a/sigma^2)`", "dimensionless, (0, 1]",
        a_smooth, [_ms(smooth[v], "{:.4f}") for v in cols])
    row("Mean discriminator reward, test", "dimensionless",
        "—", ["—" for _ in cols])
    row("RMSE, corrected vs GT", "normalised pose units",
        "—", ["not measured" for _ in cols])


def table_config(sweep: Path, seeds: list[int],
                 runs: list[str] | None = None) -> None:
    import torch
    names = runs or [f"{v}_s{s}" for v in VARIANTS for s in seeds]
    print("| run | obs width | action width | timesteps | trans_mode | state_trans | viz_interval |")
    print("|---|---|---|---|---|---|---|")
    if True:
        for v_s in names:
            v = v_s
            p = sweep / v_s / "actor_final.pt"
            if not p.exists():
                continue
            ck = torch.load(str(p), map_location="cpu")
            c = ck["config"]
            obs = ck["actor"]["net.0.weight"].shape[1]
            act = ck["actor"]["log_std"].shape[0]
            steps = c.get("total_updates", 0) * c.get("rollouts", 0)
            print(f"| {v} | {obs} | {act} | {steps:,} | {c.get('trans_mode')} | "
                  f"{c.get('state_trans')} | {c.get('viz_interval')} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", choices=("pampjpe", "rollout", "config", "diagnostics"),
                   required=True)
    p.add_argument("--h5_path", default="data/processed_movi.h5",
                   help="processed HDF5, for the (A) lifted-smoothness cell; "
                        "pass '' to skip that (it scans the whole train split)")
    p.add_argument("--heldout", default=None,
                   help="JSON from scripts/heldout_eval.py --dump, to fill the "
                        "held-out row of the diagnostics table")
    p.add_argument("--sweep_dir", default="checkpoints/sweep")
    p.add_argument("--scores_dir", default="eval_scores")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--window", type=int, default=20000)
    p.add_argument("--runs", nargs="+", default=None,
                   help="explicit run directory names, for sets not using "
                        "<variant>_s<seed> (e.g. checkpoints/biasfix)")
    a = p.parse_args()

    if a.table == "diagnostics":
        table_diagnostics(Path(a.sweep_dir), a.seeds, a.window,
                          Path(a.heldout) if a.heldout else None, a.h5_path)
    elif a.table == "rollout":
        table_rollout(Path(a.sweep_dir), a.seeds, a.window, a.runs)
    elif a.table == "pampjpe":
        table_pampjpe(Path(a.scores_dir), a.seeds)
    else:
        table_config(Path(a.sweep_dir), a.seeds, a.runs)


if __name__ == "__main__":
    main()

"""
Aggregate the seed sweep into the report table.

Three conditions x three seeds. Reports mean +/- sample std across seeds, and a
**paired** comparison against the uncorrected baseline.

Why paired. Clip difficulty dominates the spread: PA-MPJPE varies far more
between clips than between conditions, so comparing two means throws away most
of the available power. Every condition is scored on the identical set of
clip-cameras, so the per-clip difference `corrected - lifted` cancels that
variance entirely. A Wilcoxon signed-rank test over ~374 paired differences
answers "did this policy change anything" with far more sensitivity than three
run means ever could -- and it answers it for *this* policy, which is a different
question from "does the method work", which is what the seed spread is for.

    python -m scripts.summarise_sweep
    python -m scripts.summarise_sweep --update_report
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CONDITIONS = [("frozen", "(B1) PPO, trans frozen"),
              ("uv", "(B2) PPO, du,dv image shift"),
              ("notrans", "(B3) PPO, pose-only state")]
MM = 1000.0


def load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def wilcoxon(diff: np.ndarray):
    """(statistic, p) for the signed-rank test; scipy if available, else None."""
    try:
        from scipy.stats import wilcoxon as w
        nz = diff[diff != 0]
        if len(nz) < 10:
            return None
        r = w(nz)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", default="/tmp/sweep")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--update_report", action="store_true")
    ap.add_argument("--report", default=str(REPO / "report.md"))
    args = ap.parse_args()

    d = Path(args.scores_dir)
    base = load(d / "scores_baseline.json")
    if base is None:
        raise SystemExit(f"no baseline scores in {d}")
    base_mean = base["pa_mpjpe_lifted_raw"] * MM
    base_by_key = dict(zip(base.get("per_cam_keys_lifted", []),
                           base.get("per_cam_lifted", [])))
    print(f"(A) baseline, full test split : {base_mean:.2f} mm  "
          f"({base['n_clips_baseline']} clips, {len(base_by_key)} clip-cameras)\n")

    cells = {}
    print(f"{'condition':30s} {'per-seed (mm)':>34s} {'mean +/- std':>16s} {'vs (A)':>10s}  paired test")
    for key, label in CONDITIONS:
        vals, paired_note = [], ""
        for s in args.seeds:
            r = load(d / f"scores_{key}_s{s}.json")
            vals.append(np.nan if r is None else r["pa_mpjpe_corrected"] * MM)
        vals = np.array(vals, dtype=float)
        got = vals[~np.isnan(vals)]
        if len(got) == 0:
            cells[key] = "not measured"
            print(f"{label:30s} {'--':>34s}")
            continue

        # paired difference, seed 42 against the baseline on the same clip-cameras
        r42 = load(d / f"scores_{key}_s{args.seeds[0]}.json")
        if r42 and base_by_key and "per_clip_keys" in r42:
            pairs = [(v, base_by_key[k]) for k, v in
                     zip(r42["per_clip_keys"], r42["per_clip_corrected"])
                     if k in base_by_key]
            if pairs:
                a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
                diff = (a - b) * MM
                res = wilcoxon(diff)
                paired_note = (f"median {np.median(diff):+.3f} mm on {len(diff)} clip-cams"
                               + (f", p={res[1]:.2g}" if res else ""))

        mean, std = got.mean(), got.std(ddof=1) if len(got) > 1 else 0.0
        cells[key] = f"{mean:.2f} ± {std:.2f}"
        per_seed = "  ".join(f"{v:.2f}" if not np.isnan(v) else "--" for v in vals)
        print(f"{label:30s} {per_seed:>34s} {cells[key]:>16s} "
              f"{mean-base_mean:+9.2f}  {paired_note}")

    row = ("| **PA-MPJPE, test (mm), lower is better** | "
           f"**{base_mean:.2f}** | " + " | ".join(cells[k] for k, _ in CONDITIONS) + " | |")
    print(f"\nreport row:\n{row}")

    if args.update_report:
        p = Path(args.report)
        txt = p.read_text()
        new = re.sub(r"^\| \*\*PA-MPJPE, test \(mm\).*$", row, txt, count=1, flags=re.M)
        if new == txt:
            print("!! PA-MPJPE row not found in the report; not modified")
        else:
            p.write_text(new)
            print(f"updated {p}")


if __name__ == "__main__":
    main()

"""
migrate_gt_layout.py
────────────────────
Move GT datasets in processed_movi.h5 from the clip root into a dedicated
'gt' subgroup, matching the layout written by data/norm_upsample.py since
commit 45ba0ec.

    before:  split/clip/{poses,trans,betas}   + split/clip/{pg1,pg2}/...
    after:   split/clip/gt/{poses,trans,betas} + split/clip/{pg1,pg2}/...

Uses HDF5 link moves, so no data is rewritten and the file does not grow.
Clip attributes stay on the clip group; dataset attributes travel with the
datasets. Already-migrated clips are skipped, so the script is resumable.

Usage:
    python scripts/migrate_gt_layout.py data/processed_movi.h5
    python scripts/migrate_gt_layout.py data/processed_movi.h5 --dry_run
"""
from __future__ import annotations

import argparse

import h5py

GT_KEYS = ("poses", "trans", "betas")
GT_GROUP = "gt"


def migrate(h5_path: str, dry_run: bool = False) -> dict[str, int]:
    counts = {"migrated": 0, "already": 0, "skipped": 0}
    mode = "r" if dry_run else "r+"

    with h5py.File(h5_path, mode) as f:
        for split in f:
            for clip_name in f[split]:
                clip = f[split][clip_name]

                if GT_GROUP in clip:
                    counts["already"] += 1
                    continue

                missing = [k for k in GT_KEYS if k not in clip]
                if missing:
                    print(f"  skipping {split}/{clip_name}: no GT at clip root "
                          f"(missing {', '.join(missing)})")
                    counts["skipped"] += 1
                    continue

                if not dry_run:
                    clip.create_group(GT_GROUP)
                    for key in GT_KEYS:
                        f.move(f"{split}/{clip_name}/{key}",
                               f"{split}/{clip_name}/{GT_GROUP}/{key}")
                counts["migrated"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", help="processed_movi.h5 to migrate in place")
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    counts = migrate(args.h5_path, args.dry_run)
    prefix = "[dry run] " if args.dry_run else ""
    print(f"{prefix}migrated={counts['migrated']}  "
          f"already_new_layout={counts['already']}  skipped={counts['skipped']}")


if __name__ == "__main__":
    main()

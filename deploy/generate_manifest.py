"""
Generates a manifest of available data snapshots for the frontend build
step, since a static site has no way to list a directory itself. Also
copies the actual snapshot files into frontend/public/data/, since
that's what Vite's build actually bundles into the deployed site.

Run as part of render.yaml's static site buildCommand, BEFORE `vite
build` — this script itself is never committed to git and produces no
git changes — it only runs at build time, which is exactly why the
snapshot-file approach (fixing the ratings.json overwrite problem)
doesn't reintroduce the same git-friction issue: nothing in the repo
itself needs updating after the fact.
"""

import os
import json
import glob
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_and_copy_snapshots(subdir: str) -> list:
    src_pattern = os.path.join(REPO_ROOT, "data", subdir, "*.json")
    files = sorted(glob.glob(src_pattern))

    dest_dir = os.path.join(REPO_ROOT, "frontend", "public", "data", subdir)
    os.makedirs(dest_dir, exist_ok=True)

    filenames = []
    for f in files:
        filename = os.path.basename(f)
        shutil.copy(f, os.path.join(dest_dir, filename))
        filenames.append(filename)

    return filenames


def copy_single_file(filename: str) -> bool:
    """Copies a single top-level data/ file (e.g. performance.json) into
    the static build, if it exists. The dashboard's Track record tab
    fetches /data/performance.json directly (no manifest entry needed --
    it's a single well-known filename, not a snapshot series), so it just
    needs to be present in frontend/public/data/. Returns whether the
    file existed, purely for the build log."""
    src = os.path.join(REPO_ROOT, "data", filename)
    if not os.path.exists(src):
        return False
    dest_dir = os.path.join(REPO_ROOT, "frontend", "public", "data")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dest_dir, filename))
    return True


def main():
    manifest = {
        "ratings": list_and_copy_snapshots("ratings"),
        "divergence": list_and_copy_snapshots("divergence"),
        "player_grades": list_and_copy_snapshots("player_grades"),
        "cfb_ratings": list_and_copy_snapshots("cfb_ratings"),
        "cfb_divergence": list_and_copy_snapshots("cfb_divergence"),
    }

    output_dir = os.path.join(REPO_ROOT, "frontend", "public", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "manifest.json")

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    has_performance = copy_single_file("performance.json")
    has_cfb_performance = copy_single_file("cfb_performance.json")
    copy_single_file("margin_dist.json")

    print(f"[generate_manifest] wrote {output_path}: "
          f"{len(manifest['ratings'])} ratings snapshots, "
          f"{len(manifest['divergence'])} divergence snapshots, "
          f"{len(manifest['player_grades'])} player grade snapshots, "
          f"{len(manifest['cfb_ratings'])} CFB ratings snapshots, "
          f"performance.json {'copied' if has_performance else 'not present yet (Track record tab shows placeholders)'}, "
          f"cfb_performance.json {'copied' if has_cfb_performance else 'not present yet'}")


if __name__ == "__main__":
    main()

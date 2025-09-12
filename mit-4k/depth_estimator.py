"""Generate depth images from existing RGB images using a pretrained monocular depth network.

This script scans the MIT-4K dataset structure:
    mit-4k/train/img_color/<rgb files>

For every RGB image it predicts a depth map with a MiDaS (or compatible) model and
writes the result to:
    mit-4k/train/depth/<same_name>.png

Depth output is saved as an 8-bit, 3-channel PNG (24-bit colour) to stay consistent
with previous dataset format.

Usage (from project root):
    python mit-4k/generate_depth_from_rgb.py [--device cuda] [--model_type dpt_hybrid]

The first run will automatically download the pretrained weights via torch.hub.
"""
import argparse
import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# --------------------------- Helper functions ---------------------------


def _clone_midas_repo(cache_dir: Path) -> Path:
    """Clone MiDaS repository into torch hub cache if not present."""
    repo_url = "https://github.com/intel-isl/MiDaS"
    print(f"[info] ??? {repo_url} ?? MiDaS ??? {cache_dir} ?")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(cache_dir)])
    except subprocess.CalledProcessError as e:
        print("[error] git ???????????????????")
        raise e
    return cache_dir


def _get_local_repo_path() -> Path:
    hub_dir = Path.home() / ".cache" / "torch" / "hub"
    repo_dir = hub_dir / "MiDaS"
    # ??? intel-isl_MiDaS_master ?????
    if not repo_dir.exists():
        alt = hub_dir / "intel-isl_MiDaS_master"
        if alt.exists():
            repo_dir = alt
    return repo_dir


def load_midas(model_type: str = "dpt_hybrid", device: str = "cpu") -> Tuple[nn.Module, object]:
    """Load a MiDaS model with graceful fallback for offline environments.

    The function first normalises *model_type* to the callable names that appear in
    MiDaS's ``hubconf.py`` (these names are case-sensitive).  For example::

        dpt_hybrid  ->  DPT_Hybrid
        dpt_large   ->  DPT_Large
        midas_v21   ->  MiDaS
        midas_v21_small -> MiDaS_small

    This allows end-users to pass in either the original MiDaS names or the more
    convenient lower-case aliases defined by this script.
    """
    # ------------------------------------------------------------------
    # 1) Canonicalise model_type so that it matches callable names in hubconf
    # ------------------------------------------------------------------
    alias_map = {
        "dpt_hybrid": "DPT_Hybrid",
        "dpt_large": "DPT_Large",
        "midas_v21": "MiDaS",
        "midas_v21_small": "MiDaS_small",
    }
    hub_type = alias_map.get(model_type, model_type)  # fall back to raw value

    repo = "intel-isl/MiDaS"
    try:
        # Try online load first ------------------------------------------------
        model = torch.hub.load(repo, hub_type)
        transforms = torch.hub.load(repo, "transforms")
    except Exception:
        # --------------------------------------------------------------------
        # Offline fallback: search / clone local repo and load from disk
        # --------------------------------------------------------------------
        local_repo = _get_local_repo_path()
        if not local_repo.exists():
            print("[warn] MiDaS repo not found locally ? attempting to clone ?")
            try:
                local_repo = _clone_midas_repo(local_repo)
            except subprocess.CalledProcessError:
                print("[error] Could not clone MiDaS repository. Check your internet"
                      " connection or clone the repo manually to:", local_repo)
                raise
        try:
            model = torch.hub.load(str(local_repo), hub_type, source="local")
            transforms = torch.hub.load(str(local_repo), "transforms", source="local")
        except RuntimeError:
            # hubconf no longer matches ? force re-clone
            import shutil
            print("[warn] Local MiDaS repo is outdated ? refreshing ?")
            shutil.rmtree(local_repo, ignore_errors=True)
            local_repo = _clone_midas_repo(local_repo)
            model = torch.hub.load(str(local_repo), hub_type, source="local")
            transforms = torch.hub.load(str(local_repo), "transforms", source="local")

    # ------------------------------------------------------------------
    # Set device / eval mode ---------------------------------------------------
    # ------------------------------------------------------------------
    model.eval()
    model.to(device)

    # Choose correct transform -------------------------------------------------
    if hub_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return model, transform


def predict_depth(model: nn.Module, transform, img_bgr, device: str = "cpu"):
    """Run monocular depth prediction on a single BGR image => 2D numpy array (float32)."""
    # MiDaS expects RGB PIL or numpy
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth = prediction.cpu().numpy()
    # Normalize to 0-1 range
    depth -= depth.min()
    depth /= depth.max() + 1e-8
    return depth.astype(np.float32)


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Convert normalized depth (0-1) to 8-bit 3-channel RGB image."""
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.merge([depth_u8, depth_u8, depth_u8])


# --------------------------- Main script ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate depth maps from RGB using MiDaS")
    parser.add_argument("--dataset_root", type=str, default=str(Path(__file__).resolve().parent),
                        help="Path to mit-4k directory containing train/img_color and train/depth")
    parser.add_argument("--split", type=str, default="train", help="Dataset split folder (default: train)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"], help="Computation device")
    parser.add_argument("--model_type", type=str, default="dpt_hybrid",
                        help="MiDaS model variant (e.g. dpt_hybrid, dpt_large, midas_v21, midas_v21_small, or original callable names e.g. DPT_Hybrid)")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    img_dir = root / args.split / "img_color"
    depth_dir = root / args.split / "depth"
    if not img_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {img_dir}")
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MiDaS model ({args.model_type}) on {args.device}?")
    model, transform = load_midas(args.model_type, args.device)

    img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.bmp")) + sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        print("No RGB images found in", img_dir)
        return

    for p in img_paths:
        out_path = depth_dir / p.name.replace(p.suffix, ".png")  # save as PNG
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"[warn] Could not read image {p}")
            continue
        depth = predict_depth(model, transform, img_bgr, args.device)
        depth_rgb = depth_to_rgb(depth)
        cv2.imwrite(str(out_path), depth_rgb)
        print(f"[done] {p.name} -> {out_path.relative_to(root)}")

    print("Depth generation complete.")


if __name__ == "__main__":
    main()

import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from rtholo import rtholo
from utils import im2float, resize_keep_aspect, normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained rtholo model on the mit-4k dataset")
    parser.add_argument("--data_path", type=str, default="mit-4k", help="Root path of the mit-4k dataset")
    parser.add_argument("--phase", type=str, default="train", help="Sub-folder inside data_path to use (train / test)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth checkpoint")
    parser.add_argument("--save_dir", type=str, default="save", help="Directory where results will be written")
    parser.add_argument("--img_size", type=int, default=1024, help="Network image resolution")
    parser.add_argument("--layer_num", type=int, default=30,
                        help="Number of discrete depth layers used during training – required for mask computation")
    parser.add_argument("--feature_size", type=float, default=7.48e-6,
                        help="Feature size parameter used at training time")
    parser.add_argument("--img_distance", type=float, default=0.2,
                        help="Image distance parameter used at training time")
    parser.add_argument("--distance_range", type=float, default=0.03,
                        help="Depth range parameter used at training time")
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_filters_per_layer", type=int, default=15)
    parser.add_argument("--CNNPP", action="store_true", help="Whether CNN++ backbone was enabled during training")
    parser.add_argument("--use_alft", action="store_true", help="Whether ALFT module was enabled during training")
    return parser.parse_args()


def prepare_input(amp_img_path: str, depth_img_path: str, img_size: int, device: torch.device):
    """Prepare amplitude + depth tensors given the raw file paths"""
    # amplitude image processing ( copy from dataLoader.py )
    img = cv2.imread(amp_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {amp_img_path}")

    img = img[..., np.newaxis]  # (H, W, 1)
    im = im2float(img, dtype=np.float32)
    low_val = im <= 0.04045
    im[low_val] = 25 / 323 * im[low_val]
    im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11) / 211) ** (12 / 5)
    amp = np.sqrt(im)
    amp = np.transpose(amp, (2, 0, 1))
    amp = resize_keep_aspect(amp, [img_size, img_size])
    amp = np.reshape(amp, (1, img_size, img_size))

    # depth image processing
    depth = cv2.imread(depth_img_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Cannot read {depth_img_path}")
    depth = depth[..., np.newaxis]
    depth = im2float(depth, dtype=np.float32)
    depth = np.transpose(depth, (2, 0, 1))
    depth = resize_keep_aspect(depth, [img_size, img_size])
    depth = np.reshape(depth, (1, img_size, img_size))
    depth = 1 - depth  # keep consistent with training preprocessing

    amp_t = torch.from_numpy(amp).to(device)
    depth_t = torch.from_numpy(depth).to(device)
    return amp_t, depth_t


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = rtholo(
        mode="test",
        size=args.img_size,
        feature_size=args.feature_size,
        distance_range=args.distance_range,
        img_distance=args.img_distance,
        layers_num=args.layer_num,
        num_filters_per_layer=args.num_filters_per_layer,
        num_layers=args.num_layers,
        CNNPP=args.CNNPP,
        use_alft=args.use_alft,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # Prepare IO paths
    img_dir = os.path.join(args.data_path, args.phase, "img_color")
    depth_dir = os.path.join(args.data_path, args.phase, "depth")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"{img_dir} not found")
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_subdirs = ["amp", "depth", "out_amp", "out_amp_mask", "holo"]
    for sd in out_subdirs:
        os.makedirs(os.path.join(args.save_dir, run_id, sd), exist_ok=True)

    # construct numeric-sort list of files
    def _numeric_sort_key(fname):
        base = os.path.splitext(fname)[0]
        return int(base) if base.isdigit() else float('inf')

    img_names = sorted(os.listdir(img_dir), key=_numeric_sort_key)

    target_size = (1920, 1080)

    with torch.no_grad():
        for fname in tqdm(img_names, desc="Processing"):
            amp_path = os.path.join(img_dir, fname)
            depth_path = os.path.join(depth_dir, fname)
            if not os.path.isfile(depth_path):
                print(f"Warning: depth image {depth_path} not found; skipping")
                continue

            amp_t, depth_t = prepare_input(amp_path, depth_path, args.img_size, device)
            source = torch.cat([amp_t, depth_t], dim=-3)
            source = source.unsqueeze(0)  # Add batch dimension
            # dummy ikk not needed; set to 0
            holo, slm_amp, recon_field = model(source, torch.tensor([0], device=device))
            output_amp = recon_field.abs()
            mask = torch.ones_like(output_amp, device=device)  # no mask at inference

            # ----- save images -----
            save_idx = os.path.splitext(fname)[0]

            # output_amp
            out_amp_np = normalize(output_amp[0, 0].cpu().numpy()) * 255
            out_amp_resized = cv2.resize(out_amp_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, run_id, "out_amp", f"{save_idx}.bmp"), out_amp_resized)

            # depth
            depth_np = normalize(depth_t[0, 0].cpu().numpy()) * 255
            depth_resized = cv2.resize(depth_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, run_id, "depth", f"{save_idx}.bmp"), depth_resized)

            # amp original
            amp_np = normalize(amp_t[0, 0].cpu().numpy()) * 255
            amp_resized = cv2.resize(amp_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, run_id, "amp", f"{save_idx}.bmp"), amp_resized)

            # out_amp_mask simply amp * mask here
            out_mask_np = normalize((amp_t * mask)[0, 0].cpu().numpy()) * 255
            out_mask_resized = cv2.resize(out_mask_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, run_id, "out_amp_mask", f"{save_idx}.bmp"), out_mask_resized)

            # Hologram
            holo_np = normalize(holo[0, 0].cpu().numpy()) * 255
            holo_resized = cv2.resize(holo_np, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_dir, run_id, "holo", f"{save_idx}.bmp"), holo_resized)


if __name__ == "__main__":
    main()

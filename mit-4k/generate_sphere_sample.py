import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPTH_DIR = ROOT / "train" / "depth"
COLOR_DIR = ROOT / "train" / "img_color"
DEPTH_DIR.mkdir(parents=True, exist_ok=True)
COLOR_DIR.mkdir(parents=True, exist_ok=True)

# Image size (same as MIT-4K data, typically 4K/1080p). Here we use 1080x1080 for simplicity.
N = 1080  # image height and width
cx, cy = N // 2, N // 2  # center of the image
radius = N // 3  # sphere radius

# Output filename (both color and depth will share the same basename so that the dataloader can match them)
FILENAME = "sphere_color.png"

# Generate depth map (normalized 0~1 where 0 = far, 1 = near)
y, x = np.ogrid[:N, :N]
mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
# Depth of a sphere: z = sqrt(r^2 - (x^2 + y^2)), normalized.
z = np.zeros((N, N), dtype=np.float32)
rel_x = (x - cx) / radius
rel_y = (y - cy) / radius
z_val = np.sqrt(np.clip(1 - (rel_x ** 2 + rel_y ** 2), 0, 1))
z[mask] = z_val[mask]

# Save depth map **using the same filename as the color image**
# Convert to 8-bit and replicate across 3 channels so that the file is 24-bit RGB

depth_img_8 = (z * 255).astype(np.uint8)
depth_rgb = np.stack([depth_img_8] * 3, axis=-1)  # replicate to R,G,B
sphere_depth_path = DEPTH_DIR / FILENAME
Image.fromarray(depth_rgb).save(sphere_depth_path)
print(f"Saved depth map to {sphere_depth_path.relative_to(ROOT)} (24-bit RGB)")

# Generate color image (simple shaded sphere)
color_img = Image.new("RGB", (N, N), (0, 0, 0))
# Simple radial shading based on depth value
for y_coord in range(N):
    for x_coord in range(N):
        if mask[y_coord, x_coord]:
            intensity = int(z[y_coord, x_coord] * 255)
            color_img.putpixel((x_coord, y_coord), (intensity, intensity, intensity))

sphere_color_path = COLOR_DIR / FILENAME
color_img.save(sphere_color_path)
print(f"Saved color image to {sphere_color_path.relative_to(ROOT)}")

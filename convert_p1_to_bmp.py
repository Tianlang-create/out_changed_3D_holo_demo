from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "save"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load P1 from MATLAB file
mat_path = ROOT / "P1.mat"
if not mat_path.exists():
    raise FileNotFoundError(f"{mat_path} not found. Please place P1.mat in the project root.")

mat_data = loadmat(mat_path.as_posix())
# Expecting variable named 'P1' in the .mat file
if "P1" not in mat_data:
    raise KeyError("Variable 'P1' not found inside P1.mat. Make sure the MAT file contains 'P1'.")

P1 = mat_data["P1"].astype(np.float64)

N = 1080
# Create coordinate grids
xn = np.arange(-N / 2, N / 2)
xn, yn = np.meshgrid(xn, xn)

# Generate phase pattern (mirrors MATLAB operations)
P1G = P1 + 2 * np.pi * (xn * 144 + yn * 0) / N
P1G = np.remainder(P1G + 2 * np.pi, 2 * np.pi)
P1G = np.remainder(P1G + 2 * np.pi, 2 * np.pi)

# Scale to 0-255 and convert to uint8
P1GW = (P1G / (2 * np.pi) * 255).astype(np.uint8)

# Save as BMP
bmp_path = SAVE_DIR / "final6-1.bmp"
Image.fromarray(P1GW).save(bmp_path)
print(f"Saved BMP image to {bmp_path}")

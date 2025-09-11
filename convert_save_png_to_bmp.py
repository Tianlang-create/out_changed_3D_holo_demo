import sys
from pathlib import Path

from PIL import Image


def convert_png_to_bmp(base_dir: Path):
    png_files = list(base_dir.rglob("*.png"))
    if not png_files:
        print(f"No PNG files found under {base_dir}.")
        return

    for png_path in png_files:
        bmp_path = png_path.with_suffix(".bmp")
        try:
            with Image.open(png_path) as img:
                # Ensure image is in a mode that BMP supports
                if img.mode in ("RGBA", "LA"):
                    img = img.convert("RGB")
                img.save(bmp_path)
            print(f"Converted {png_path.relative_to(base_dir)} -> {bmp_path.relative_to(base_dir)}")
        except Exception as e:
            print(f"Failed to convert {png_path}: {e}")


if __name__ == "__main__":
    # Default path is project 'save' folder relative to this script
    DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "save"
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAVE_DIR

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Target directory {target_dir} does not exist or is not a directory.")
        sys.exit(1)

    print(f"Converting all PNG files under {target_dir} to BMP format...")
    convert_png_to_bmp(target_dir)
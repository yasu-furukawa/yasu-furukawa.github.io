"""Generate placeholder HP.png (high-frequency) and LP.png (low-frequency) test images.

Replace these with your own 500x500 grayscale photographs before submitting —
see README.md. Left in place only so the rest of the pipeline can run end to end.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy import signal

SIZE = 500
SEED = 42


def make_hp(rng: np.random.Generator) -> np.ndarray:
    noise = rng.integers(0, 256, size=(SIZE, SIZE), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    checker = (((xx // 10) + (yy // 10)) % 2) * 255
    hp = (0.6 * noise + 0.4 * checker).astype(np.uint8)
    return hp


def make_lp(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    lp = np.zeros((SIZE, SIZE), dtype=np.float64)

    # broad smooth gradient (the dominant, near-DC content)
    for _ in range(4):
        cy, cx = rng.uniform(0, SIZE, size=2)
        sigma = rng.uniform(80, 140)
        amp = rng.uniform(0.5, 1.0)
        lp += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    lp -= lp.min()
    lp = lp / lp.max()

    # a few soft-edged blobs so there is *some* boundary content for edge
    # detection to respond to — still low-frequency once blurred, but not
    # perfectly flat everywhere.
    shapes = np.zeros((SIZE, SIZE), dtype=np.float64)
    for _ in range(3):
        cy, cx = rng.uniform(SIZE * 0.2, SIZE * 0.8, size=2)
        radius = rng.uniform(40, 90)
        shapes += ((xx - cx) ** 2 + (yy - cy) ** 2 < radius**2).astype(np.float64) * rng.uniform(0.4, 0.8)
    blur_sigma = 12
    ax = np.arange(2 * int(np.ceil(3 * blur_sigma)) + 1) - int(np.ceil(3 * blur_sigma))
    bx, by = np.meshgrid(ax, ax)
    blur_kernel = np.exp(-(bx**2 + by**2) / (2 * blur_sigma**2))
    blur_kernel /= blur_kernel.sum()
    shapes = signal.fftconvolve(shapes, blur_kernel, mode="same")

    lp = 0.7 * lp + 0.3 * shapes
    lp -= lp.min()
    lp = lp / lp.max() * 255
    return lp.astype(np.uint8)


def main() -> None:
    rng = np.random.default_rng(SEED)
    out_dir = Path(__file__).parent
    cv2.imwrite(str(out_dir / "HP.png"), make_hp(rng))
    cv2.imwrite(str(out_dir / "LP.png"), make_lp(rng))
    print(f"Wrote {out_dir/'HP.png'} and {out_dir/'LP.png'}")


if __name__ == "__main__":
    main()

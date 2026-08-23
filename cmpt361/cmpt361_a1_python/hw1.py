"""CMPT 361 Programming Assignment 1 — Python port.

Runs every task from the assignment spec (frequency analysis, kernel
visualization, spatial/frequency-domain filtering, anti-aliased subsampling,
Canny threshold sweeps) against HP.png and LP.png, writing all results into
output/. Re-run with `python hw1.py` any time the input images change.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy import signal

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "output"
GAUSSIAN_SIGMA = 2.5

# Canny (low, high) thresholds per image. Each non-optimal variant nudges one
# threshold away from the optimal setting while holding the other fixed, so
# the naming reads as <direction>-<threshold moved>: "highlow" = low threshold
# raised, "lowhigh" = high threshold lowered, etc.
CANNY_PARAMS = {
    "HP": {
        "optimal": (50, 150),
        "lowlow": (10, 150),
        "highlow": (110, 150),
        "lowhigh": (50, 60),
        "highhigh": (50, 250),
    },
    "LP": {
        "optimal": (8, 25),
        "lowlow": (2, 25),
        "highlow": (18, 25),
        "lowhigh": (8, 12),
        "highhigh": (8, 60),
    },
}


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img.astype(np.float64)


def save_gray(img: np.ndarray, path: Path) -> None:
    """Min-max normalize to 0-255 and write as 8-bit PNG."""
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    normed = (img - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(img)
    cv2.imwrite(str(path), normed.astype(np.uint8))


def fft_magnitude_image(img: np.ndarray) -> np.ndarray:
    """Log-scaled, fftshifted magnitude spectrum, ready to save with save_gray."""
    spectrum = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(spectrum))


def gaussian_kernel_2d(sigma: float, size: int | None = None) -> np.ndarray:
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def sobel_kernel_horizontal() -> np.ndarray:
    return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)


def dog_kernel_2d(sigma: float, size: int | None = None) -> np.ndarray:
    """Derivative of a 2D Gaussian along x — a single directional edge filter."""
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    dog = -(xx / sigma**2) * g
    return dog - dog.mean()


def save_surface(kernel: np.ndarray, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    size = kernel.shape[0]
    coords = np.arange(size) - size // 2
    xx, yy = np.meshgrid(coords, coords)
    ax.plot_surface(xx, yy, kernel, cmap="viridis")
    ax.set_title(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def freq_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply `kernel` to `img` entirely in the frequency domain."""
    h, w = img.shape
    kh, kw = kernel.shape
    padded = np.zeros((h, w))
    y0, x0 = h // 2 - kh // 2, w // 2 - kw // 2
    padded[y0 : y0 + kh, x0 : x0 + kw] = kernel
    padded = np.fft.ifftshift(padded)  # move kernel center to (0, 0)
    filt_freq = np.fft.fft2(padded)
    img_freq = np.fft.fft2(img)
    return np.real(np.fft.ifft2(img_freq * filt_freq))


def run_frequency_analysis(name: str, img: np.ndarray) -> None:
    save_gray(fft_magnitude_image(img), OUT_DIR / f"{name}-freq.png")


def run_kernel_filtering(name: str, img: np.ndarray, gauss: np.ndarray, dog: np.ndarray) -> None:
    gauss_filt = signal.convolve2d(img, gauss, mode="same", boundary="symm")
    save_gray(gauss_filt, OUT_DIR / f"{name}-filt.png")
    save_gray(fft_magnitude_image(gauss_filt), OUT_DIR / f"{name}-filt-freq.png")

    dog_filt = freq_filter(img, dog)
    save_gray(dog_filt, OUT_DIR / f"{name}-dogfilt.png")
    save_gray(fft_magnitude_image(dog_filt), OUT_DIR / f"{name}-dogfilt-freq.png")


def run_anti_aliasing(name: str, img: np.ndarray) -> None:
    for factor in (2, 4):
        sub = img[::factor, ::factor]
        save_gray(sub, OUT_DIR / f"{name}-sub{factor}.png")
        save_gray(fft_magnitude_image(sub), OUT_DIR / f"{name}-sub{factor}-freq.png")

        aa_sigma = factor / 2.0  # cuts frequencies above the new Nyquist limit
        smoothed = signal.convolve2d(img, gaussian_kernel_2d(aa_sigma), mode="same", boundary="symm")
        sub_aa = smoothed[::factor, ::factor]
        save_gray(sub_aa, OUT_DIR / f"{name}-sub{factor}-aa.png")
        save_gray(fft_magnitude_image(sub_aa), OUT_DIR / f"{name}-sub{factor}-aa-freq.png")


def run_canny(name: str, img: np.ndarray) -> None:
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    for variant, (low, high) in CANNY_PARAMS[name].items():
        edges = cv2.Canny(img_u8, low, high)
        cv2.imwrite(str(OUT_DIR / f"{name}-canny-{variant}.png"), edges)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    gauss = gaussian_kernel_2d(GAUSSIAN_SIGMA)
    dog = dog_kernel_2d(GAUSSIAN_SIGMA)
    sobel_kernel_horizontal()  # defined per spec; not separately visualized/applied below
    save_surface(gauss, OUT_DIR / "gaus-surf.png", f"Gaussian kernel, sigma={GAUSSIAN_SIGMA}")
    save_surface(dog, OUT_DIR / "dog-surf.png", f"Derivative-of-Gaussian kernel, sigma={GAUSSIAN_SIGMA}")

    for name in ("HP", "LP"):
        img = load_gray(ROOT / f"{name}.png")
        run_frequency_analysis(name, img)
        run_kernel_filtering(name, img, gauss, dog)
        run_anti_aliasing(name, img)
        run_canny(name, img)

    print(f"Done. Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()

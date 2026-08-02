# CMPT 361 HW1 — Python port

Python re-implementation of the [CMPT 361 Programming Assignment 1](https://yaksoy.github.io/introvc/assignment1/)
(frequency-domain analysis, Gaussian/DoG/Sobel kernels, spatial- and frequency-domain filtering, anti-aliased
subsampling, Canny threshold sweeps). No MATLAB template code exists for this assignment — only an HTML report
template — so this is a from-scratch Python pipeline built from the assignment spec, not a translation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Replace `HP.png` and `LP.png` with your own 500x500 grayscale photographs — one with lots of high-frequency
   detail/edges, one dominated by low-frequency content. The versions currently in this folder are synthetic
   placeholders (see `generate_images.py`) so the pipeline can be verified end to end; swap them before submitting
   anything for a grade.
2. Run the pipeline:
   ```bash
   python hw1.py
   ```
   This writes every required output image into `output/`.
3. Open `report.html` in a browser to review the assembled report. Fill in the name/date/honor-statement fields at
   the top and the `Discussion:` placeholders throughout — those are your own analysis and shouldn't be
   auto-generated.

## Files

- `generate_images.py` — creates placeholder `HP.png`/`LP.png`. Skip this once you have your own photos.
- `hw1.py` — the full pipeline: FFT analysis, kernel construction/visualization, Gaussian (spatial) and DoG
  (frequency-domain) filtering, anti-aliased subsampling, Canny sweeps. Kernel sigma and Canny threshold values are
  defined as constants near the top of the file — adjust them for your own images.
- `report.html` — static report page referencing `output/*.png`.

## Publishing to GitHub Pages

```bash
git init
git add .
git commit -m "Add CMPT 361 HW1 Python port"
git remote add origin <your-repo-url>
git push -u origin main
```

Then enable GitHub Pages for the repo (Settings → Pages → deploy from the `main` branch), and `report.html` will be
served at `https://<username>.github.io/<repo>/report.html`.

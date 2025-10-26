# Hybrid Images (Computational Photography / CS445 Project 1)

Hybrid images combine the low frequencies of one image with the high
frequencies of another so that the perceived subject changes with viewing
distance and scale. Up close you see the high‑frequency content; from
farther away you see the low‑frequency content.

This repository contains a starter notebook, helper utilities, and sample
images to create and analyze hybrid images.

## Repository Structure

- `CS445_Proj1_Starter.ipynb` — main notebook to run the project.
- `utils.py` — helper functions for alignment, cropping, filtering, and plotting.
- `photos/` — sample images and example results.
- `requirements.txt` — Python dependencies.
- PDFs — reference papers and project report examples.

## Quick Start

Prerequisites: Python 3.9+ recommended. macOS/Linux/Windows supported. Ensure
you can install Python wheels for `opencv-python`.

1) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
# If you need Jupyter:
pip install notebook jupyterlab
```

3) Launch the notebook

```bash
jupyter lab   # or: jupyter notebook
```

Open `CS445_Proj1_Starter.ipynb` and run cells top‑to‑bottom.

## Workflow Overview

1) Choose two images (e.g., from `photos/`). Ideally, align subjects (eyes,
nose, dominant edges) to help the hybrid illusion.
2) Align and/or crop the images so they share the same field of view.
3) Create a low‑pass version of Image A (Gaussian blur).
4) Create a high‑pass version of Image B (subtract its low‑pass version).
5) Combine: `hybrid = low_pass_A + high_pass_B` and clip to [0, 1].
6) Save results and inspect at different scales/distances.

## Utilities (utils.py)

- `align_images(path1, path2, pts1, pts2, save_images=False)`
  Aligns two images given two reference points in each image. Returns two
  aligned `numpy` arrays in BGR order (OpenCV). Set `save_images=True` to
  write aligned versions to disk.

- `prompt_eye_selection(image)`
  Interactive Matplotlib click tool to select two points in an image. Returns
  a `(2, 2)` array of x/y coordinates.

- `interactive_crop(image)` and `crop_image(image, points)`
  Click‑based cropping. Use `interactive_crop` to select two corners; it
  returns the cropped image and crop bounds.

- `gaussian_kernel(sigma, kernel_half_size)`
  Generates a normalized 2D Gaussian kernel for blurring.

- `plot(array, filename=None)` and `plot_spectrum(magnitude_spectrum)`
  Convenience plotting and log‑scaled spectrum visualization.

## Minimal Example (script‑style)

You can adapt the following snippet inside the notebook to produce a hybrid
image. Adjust `sigma_lp`/`sigma_hp` empirically (e.g., 4–10 for faces).

```python
import cv2
import numpy as np
from utils import gaussian_kernel, prompt_eye_selection, align_images

# Load RGB for visualization; keep BGR for OpenCV filtering if preferred
imA = cv2.cvtColor(cv2.imread('photos/vivian.jpg'), cv2.COLOR_BGR2RGB)
imB = cv2.cvtColor(cv2.imread('photos/cat.jpg'), cv2.COLOR_BGR2RGB)

# Select two corresponding points (e.g., eyes) in each image
ptsA = prompt_eye_selection(imA)  # click two points, close the window
ptsB = prompt_eye_selection(imB)

# Align (returns BGR images); convert back to RGB for display
A_aligned_bgr, B_aligned_bgr = align_images('photos/vivian.jpg', 'photos/cat.jpg', ptsA, ptsB)
A = cv2.cvtColor(A_aligned_bgr, cv2.COLOR_BGR2RGB) / 255.0
B = cv2.cvtColor(B_aligned_bgr, cv2.COLOR_BGR2RGB) / 255.0

# Low‑pass A
sigma_lp = 8
k_lp = gaussian_kernel(sigma_lp, kernel_half_size=int(3*sigma_lp))
A_low = cv2.filter2D(A, ddepth=-1, kernel=k_lp, borderType=cv2.BORDER_REFLECT)

# High‑pass B (B minus its low‑pass)
sigma_hp = 4
k_hp = gaussian_kernel(sigma_hp, kernel_half_size=int(3*sigma_hp))
B_low = cv2.filter2D(B, ddepth=-1, kernel=k_hp, borderType=cv2.BORDER_REFLECT)
B_high = B - B_low

# Combine and clip
hybrid = np.clip(A_low + B_high, 0.0, 1.0)

# Save for inspection at different scales
cv2.imwrite('hybrid_result.png', (hybrid[..., ::-1]*255).astype(np.uint8))  # RGB->BGR
```

Tips:
- Ensure both images have the same resolution before filtering/combining.
- Try different `sigma` pairs; the illusion is sensitive to scale and content.
- Normalize to `[0, 1]` for computation; convert back to `[0, 255]` for saving.

## Visualization and Analysis

- Multi‑scale viewing: resize the output smaller to emphasize low frequencies.
- FFT magnitude spectra: use `np.fft.fft2`/`np.fft.fftshift` and
  `utils.plot_spectrum` to inspect frequency content of inputs and result.

## Troubleshooting

- Colors look off: OpenCV loads images as BGR. Convert with
  `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` for display, and RGB->BGR when saving.
- Result looks ghosted/misaligned: use `align_images` and carefully pick
  corresponding points; try cropping to remove borders.
- Weak hybrid effect: increase low‑pass `sigma` for the base image and/or
  adjust the high‑pass `sigma`; choose images with compatible structures.
- Clipping/over‑contrast: consider scaling the high‑pass component (e.g.,
  `hybrid = np.clip(A_low + 0.7*B_high, 0, 1)`).

## Acknowledgments

- Oliva, A., Torralba, A., & Schyns, P. (2006). Hybrid Images. ACM
  SIGGRAPH.  
- Burt, P. J., & Adelson, E. H. (1983). The Laplacian Pyramid as a Compact
  Image Code.

This repository is for educational use in a computational photography course.

# Deinterlace
[![Coverage Status](https://coveralls.io/repos/github/darikoneil/deinterlacing/badge.svg?branch=main)](https://coveralls.io/github/darikoneil/deinterlacing?branch=main)


This python module can be used to correct the misalignment between forward and 
backward-scanned lines collected by resonance-scanning microscopes.

## Features
- **GPU Acceleration**: Optional CuPy backend for increased performance
- **Batch Processing**: Supports block-wise processing to reduce memory constraints.
- **Pooling Noisy Data**: Deinterlacing can be applied to pooled-pixels for improved performance on noisy or sparse images.
- **Handles Instability**: Supports processing individual frames while autocorrection 
  methods applied during acquisition stabilize
- **Sub-Pixel**: Pixel & Sub-Pixel registration available

## Installation
The repository is available on PyPI and can be installed using your
preferred package manager. For example:
pip
```bash
pip install deinterlacing
```
uv
```bash
uv add deinterlacing
```

## Dependencies
- Boltons
- CuPy  (Optional)
- NumPy
- Pydantic
- TQDM

## Example
```python
from deinterlacing import deinterlace
import numpy as np

# Load your images
images = np.load("my_images.npy")

# Deinterlace the images
deinterlace(images)
```

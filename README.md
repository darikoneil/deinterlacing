# Deinterlace Resonant-Scanned Images
This python module can be used to correct the misalignment between forward- and backward-scanned lines using a Fourier-based phase correlation approach.

## Features
- GPU Acceleration: Optional CuPy backend for increased performance
- Batch Processing: Supports block-wise processing to reduce memory constraints.
- Subsampling for Noisy Data: Deinterlacing can be applies to pixel-wise standard deviations for       improved performance on noisy or sparse images.
- Handles Unstability: Supports processing individual frames while autocorrection methods applied      during aquisition stabilize

## Dependencies
- Boltons
- CuPy  (Optional)
- NumPy
- TQDM
  

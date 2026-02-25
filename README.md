Tanuj Tekkale (tt27868), Akshay Gaitonde (ag84839)

__Milestone 1__
- Implemented Whitted-style shading with emissive, ambient, diffuse, and specular terms.
- Added distance attenuation, hard/partial shadows, recursive reflection, and recursive refraction.
- Shadow rays support color filtering through transmissive objects.
- Secondary rays are offset with epsilon to avoid self-intersection artifacts.
- Implemented triangle-ray intersection using barycentric coordinates.
- Implemented Phong normal interpolation for meshes with per-vertex normals.
- Implemented barycentric interpolation of per-vertex material color.
- Degenerate mesh faces are skipped.
- Added supersampling anti-aliasing with 1x1, 2x2, 3x3, and 4x4 patterns.

__Milestone 2__
- Added BVH acceleration for scene objects.
- Added per-trimesh BVH acceleration for mesh faces.
- Added texture mapping with bilinear sampling.
- Added cubemap support from CLI (`-c`) with smart matching of all six faces.
- Mesh backface culling is used for non-refraction rays and disabled for refraction rays.

__Creative Scene (EC)__
- Custom scenes:
  `assets/custom/custom.json`,
  `assets/custom/custom2.json`,
  `assets/custom/custom3.json`,
  `assets/custom/custom4.json`.
- These scenes combine multiple lights, reflective/transmissive materials, and denser scene layouts.
- Example:
  `build/bin/ray -r 5 assets/custom/custom4.json raycheck.out/custom4_preview.png`

__Portals (EC)__
- Added linked circular/rectangular portals that teleport rays between paired openings.
- Portal scenes:
  `assets/custom/portal_rect.json`,
  `assets/custom/portal_circle.json`.
- Example:
  `build/bin/ray -r 5 -w 640 assets/custom/portal_rect.json raycheck.out/portal_preview.png`

__Neural Network Upsampling (EC)__
- `download_data_neuralnet.py` downloads the DIV2K dataset (train + validation splits) from ETH Zurich; 800 training images and 100 validation images, all high-resolution PNG.
- Prints a final summary of how many `.png` files landed in each folder.
- `ec_neuralnet.py` defines `UpsampleNN`, an EDSR/RCAN-style super-resolution network
- it learns only the high-frequency residual on top of a bicubic upscale, with 16 residual blocks, channel attention, and no batch norm
- we cited a couple papers we read about it in the code
- `ImagePairDataset` generates LR/HR pairs on-the-fly by bicubic-downscaling HR crops so no separate LR folder needed; supports random horizontal/vertical flips for augmentation
- `train()` runs the training loop with L1 loss, mixed-precision, and a cosine LR schedule
- `evaluate_and_save_results()` benchmarks the trained model against bicubic and Gaussian baselines using PSNR and SSIM, and saves a 4-panel comparison image per input
- When run as `__main__`, trains on `DIV2K_train_HR` for 15 epochs (or loads existing weights), then evaluates on both `nn_inputs/` and `DIV2K_valid_HR/`
- `nn_infer.py` imports `UpsampleNN` from `ec_neuralnet.py` and has two input modes: `--input` (upscale an existing image/folder) or `--scene` (invoke the ray tracer as a subprocess first, then upscale the output)
- `--scene` mode optionally renders a high-res reference at `width x upscale` alongside the low-res render for comparison
- When a reference is provided, computes and prints a PSNR + SSIM table comparing NN vs. bicubic vs. Gaussian
- Optionally saves a labeled 4-panel side-by-side strip (HR ref / bicubic / gaussian / neural net) via `--save-comparison`
- Supports pass-through flags to the ray binary via `--ray-args`, including `--render-cubemap` for cubemap scenes

The training results for the current out_model.pth is:
```
Evaluating on training domain (nn_inputs)...
Neural Net       PSNR: 41.61, SSIM: 0.9843
Bicubic Baseline PSNR: 29.41, SSIM: 0.9418
Gaussian Baseline PSNR: 30.89, SSIM: 0.9349

Evaluating on DIV2K validation set...
Neural Net       PSNR: 32.40, SSIM: 0.9250
Bicubic Baseline PSNR: 29.35, SSIM: 0.8918
Gaussian Baseline PSNR: 28.16, SSIM: 0.8476
```

__Overlapping Objects (EC)__
- Added overlap-aware refraction by tracking active transmissive media per ray.
- For normal single-object cases, the standard behavior is preserved.
- For overlap/containment transitions, refraction uses the current medium index and next medium index.
- Overlap assumption: in ambiguous overlap regions, the active medium is the most recently entered transmissive object.
- Demo scene: `assets/custom/custom_overlap.json`
- Example:
  `RAY_ENABLE_OVERLAP_REFRACTION=1 build/bin/ray -r 7 -w 640 assets/custom/custom_overlap.json raycheck.out/overlap/custom_overlap.png`

__Harmonic Tracing (EC)__
- Added optional harmonic tracing with conservative Harnack-inspired step sizes and root refinement.
- Includes two demos:
  `assets/custom/harmonic_riemann.json`,
  `assets/custom/harmonic_gyroid.json`.
- Riemann demo:
  `RAY_ENABLE_HARMONIC_TRACING=1 RAY_HARMONIC_MODE=riemann build/bin/ray -r 7 -w 640 assets/custom/harmonic_riemann.json raycheck.out/harmonic/harmonic_riemann.png`
- Gyroid demo:
  `RAY_ENABLE_HARMONIC_TRACING=1 RAY_HARMONIC_MODE=gyroid build/bin/ray -r 7 -w 640 assets/custom/harmonic_gyroid.json raycheck.out/harmonic/harmonic_gyroid.png`

__Build and Test__
- Main script:
  `./build.sh`
- Outputs:
  `raycheck.out/report.csv`
  `raycheck.out/custom/report.csv`
  `raycheck.out/cubemap/report.csv`
  `raycheck.out/aa/s1/report.csv` ... `raycheck.out/aa/s4/report.csv`
  `raycheck.out/portal/portal_rect.png`
  `raycheck.out/portal/portal_circle.png`
  `raycheck.out/overlap/custom_overlap.png`
  `raycheck.out/harmonic/harmonic_riemann.png`
  `raycheck.out/harmonic/harmonic_gyroid.png`

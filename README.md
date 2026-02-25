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
- Training/evaluation script: `ec_neuralnet.py`
- Inference script: `nn_infer.py`
- `nn_infer.py` supports post-processing existing images: `--input <image_or_dir>`.
- `nn_infer.py` also supports end-to-end ray trace + post-process: `--scene <scene_file>`.
- In scene mode, the script runs the ray tracer first, then applies NN upsampling/antialiasing.
- Extra ray-tracer flags can be passed through with `--ray-args ...`.
- Baselines include bicubic and Gaussian filtering, and `--render-reference` prints PSNR comparisons.
- Example (full pipeline, with baseline comparison):
  `python3 nn_infer.py --scene assets/custom/custom.json --output raycheck.out/nn_demo --model our_model.pth --upscale 2 --render-width 320 --render-reference --save-comparison --ray-args -r 5`
- Example (inference-only on existing image):
  `python3 nn_infer.py --input raycheck.out/base/image/simple_box.png --output raycheck.out/nn_demo --model our_model.pth --upscale 2`

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

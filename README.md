Tanuj Tekkale (tt27868), Akshay Gaitonde (ag84839)

__Assignment Part 1__
- We support partial and color-filtered shadow attenuation through transmissive objects by scaling light contribution with transmissive color.
- We determine entering/exiting behavior by checking the sign of the ray-direction dot normal term and handling refraction orientation from that.
- We offset spawned secondary rays by a small epsilon to reduce self-intersection artifacts.
- Triangle meshes use barycentric intersection with Phong interpolation of per-vertex normals.
- We interpolate per-vertex diffuse/material color using barycentric weights (without normalizing material values).
- Degenerate mesh triangles are ignored and not added to the face list.

__Assignment Part 2__
- We use BVH acceleration at the scene level for bounded objects.
- We also build a per-trimesh face BVH to accelerate mesh ray intersection.
- We support cubemap loading from command line (`-c`) with smart matching of the remaining five cubemap face files.
- For triangle meshes, backface culling is enabled for non-refraction rays and disabled for refraction rays.

__Creative Scene (EC)__
- We include custom creative scenes in `assets/custom/custom.json`, `assets/custom/custom2.json`, `assets/custom/custom3.json`, and `assets/custom/custom4.json`.
- These scenes are designed to show combined effects (multi-light shading, reflections/refractions, and complex object/layout composition).
- Example run command: `build/bin/ray -r 5 assets/custom/custom4.json raycheck.out/custom4_preview.png`.

__Portals (EC)__
- We implement portal rendering with paired circular/rectangular openings where rays entering one portal emerge from its linked partner.
- Portal scenes are in `assets/custom/portal_rect.json` and `assets/custom/portal_circle.json`.
- Example run command: `build/bin/ray -r 5 -w 640 assets/custom/portal_rect.json raycheck.out/portal_preview.png`.

__Neural Network (EC)__
- We include a neural-network upsampling/anti-aliasing pipeline with baseline comparisons against bicubic and Gaussian filtering.
- The main training/evaluation script is `ec_neuralnet.py`.
- Inference-only usage is provided in `nn_infer.py`.
- Example inference command: `python3 nn_infer.py --input <input_file_or_dir> --output <output_dir> --model our_model.pth --upscale 2`.

__Overlapping Objects (EC)__
- We add overlap-aware refraction by tracking the active transmissive media along a refraction ray.
- The overlap-aware mode is opt-in, so required-scene behavior remains unchanged unless enabled.
- In the common non-overlap case (air to one object and back to air), we preserve the original transition behavior.
- For overlap/containment transitions, we use current-medium to next-medium indices for Snell refraction.
- Assumption in ambiguous overlap regions: the effective medium is the most recently entered transmissive object.
- Demo scene: `assets/custom/custom_overlap.json`.
- Example run command: `RAY_ENABLE_OVERLAP_REFRACTION=1 build/bin/ray -r 7 -w 640 assets/custom/custom_overlap.json raycheck.out/overlap/custom_overlap.png`.

__Harmonic Tracing (EC)__
- We add an opt-in harmonic tracing path that uses conservative Harnack-inspired step bounds to march rays toward level-set roots.
- The mode is disabled by default, so required Milestone I/II behavior and performance are unaffected unless explicitly enabled.
- We include a harmonic Riemann-surface demo (`assets/custom/harmonic_riemann.json`) and a gyroid demo (`assets/custom/harmonic_gyroid.json`).
- Enable command (Riemann): `RAY_ENABLE_HARMONIC_TRACING=1 RAY_HARMONIC_MODE=riemann build/bin/ray -r 7 -w 640 assets/custom/harmonic_riemann.json raycheck.out/harmonic/harmonic_riemann.png`.
- Enable command (Gyroid): `RAY_ENABLE_HARMONIC_TRACING=1 RAY_HARMONIC_MODE=gyroid build/bin/ray -r 7 -w 640 assets/custom/harmonic_gyroid.json raycheck.out/harmonic/harmonic_gyroid.png`.

__Build and Testing__
- We run `./build.sh` to build in Release mode and run the test sweep.
- The script runs baseline `raycheck` over standard scenes and writes the summary to `raycheck.out/report.csv`.
- The script runs cubemap-specific checks and writes results to `raycheck.out/cubemap/report.csv`.
- The script runs anti-aliasing checks for supersamples 1, 2, 3, and 4 in `raycheck.out/aa/s1` through `raycheck.out/aa/s4`.
- The script renders portal demo outputs to `raycheck.out/portal/portal_rect.png` and `raycheck.out/portal/portal_circle.png`.
- The script also renders the overlap demo output to `raycheck.out/overlap/custom_overlap.png`.
- The script renders harmonic demo outputs to `raycheck.out/harmonic/harmonic_riemann.png` and `raycheck.out/harmonic/harmonic_gyroid.png`.

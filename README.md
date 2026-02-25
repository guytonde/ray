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
- We include custom creative scenes in `assets/scenes/custom.json`, `assets/scenes/custom2.json`, `assets/scenes/custom3.json`, and `assets/scenes/custom4.json`.
- These scenes are designed to show combined effects (multi-light shading, reflections/refractions, and complex object/layout composition).
- Example run command: `build/bin/ray -r 5 assets/scenes/custom4.json raycheck.out/custom4_preview.png`.

__Portals (EC)__
- We implement portal rendering with paired circular/rectangular openings where rays entering one portal emerge from its linked partner.
- Portal scenes are in `assets/portal_scenes/portal_rect.json` and `assets/portal_scenes/portal_circle.json`.
- Example run command: `build/bin/ray -r 5 -w 640 assets/portal_scenes/portal_rect.json raycheck.out/portal_preview.png`.

__Neural Network (EC)__
- We include a neural-network upsampling/anti-aliasing pipeline with baseline comparisons against bicubic and Gaussian filtering.
- The main training/evaluation script is `ec_neuralnet.py`.
- Inference-only usage is provided in `nn_infer.py`.
- Example inference command: `python3 nn_infer.py --input <input_file_or_dir> --output <output_dir> --model our_model.pth --upscale 2`.

__Build and Testing__
- We run `./build.sh` to build in Release mode and run the test sweep.
- The script runs baseline `raycheck` over standard scenes and writes the summary to `raycheck.out/report.csv`.
- The script runs cubemap-specific checks and writes results to `raycheck.out/cubemap/report.csv`.
- The script runs anti-aliasing checks for supersamples 1, 2, 3, and 4 in `raycheck.out/aa/s1` through `raycheck.out/aa/s4`.
- The script renders portal demo outputs to `raycheck.out/portal/portal_rect.png` and `raycheck.out/portal/portal_circle.png`.

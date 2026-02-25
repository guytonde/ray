set -euo pipefail

cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"
cd ..

# cubemaps from assets/cubemaps/
# pass one face per cubemap; raytracer auto-discovers the other 5
declare -A CUBEMAPS
CUBEMAPS[autumn]="assets/cubemaps/cubemap_autumn/autumn_positive_x.png"
CUBEMAPS[brudslojan]="assets/cubemaps/cubemap_brudslojan/$(ls assets/cubemaps/cubemap_brudslojan/ | grep -i 'posx\|pos_x\|positive_x' | head -1)"
CUBEMAPS[iceriver]="assets/cubemaps/cubemap_iceriver/$(ls assets/cubemaps/cubemap_iceriver/ | grep -i 'posx\|pos_x\|positive_x' | head -1)"
CUBEMAPS[nvlobby]="assets/cubemaps/cubemap_nvlobby_new/$(ls assets/cubemaps/cubemap_nvlobby_new/ | grep -i 'posx\|pos_x\|positive_x' | head -1)"
CUBEMAPS[terrain]="assets/cubemaps/cubemap_terrain/$(ls assets/cubemaps/cubemap_terrain/ | grep -i 'posx\|pos_x\|positive_x' | head -1)"
CUBEMAPS[unwrapped]="assets/cubemaps/cubemap_unwrapped/$(ls assets/cubemaps/cubemap_unwrapped/ | grep -i 'posx\|pos_x\|positive_x' | head -1)"
CUBEMAPS[ForbiddenCity]="assets/cubemaps/ForbiddenCity/posx.png"
CUBEMAPS[earth]="assets/cubemaps/earth/$(ls assets/cubemaps/earth/ | grep -i 'posx\|pos_x\|positive_x' | head -1)"
CUBEMAPS[escher]="assets/cubemaps/escher/$(ls assets/cubemaps/escher/ | grep -i 'posx\|pos_x\|positive_x' | head -1)"

#autumn is the default single-cubemap for tests that only need one
DEFAULT_CUBEMAP="${CUBEMAPS[autumn]}"

echo "Running baseline test sweep..."
python3 raycheck.py \
  --refbin ./ray-solution \
  --out raycheck.out/base \
  --i-understand-that-image-metrics-are-not-perfect
cp -f raycheck.out/base/report.csv raycheck.out/report.csv

echo "Running custom scene sweep..."
mkdir -p raycheck.out/custom_scenes
ln -sf ../../assets/custom/custom.json  raycheck.out/custom_scenes/custom.json
ln -sf ../../assets/custom/custom2.json raycheck.out/custom_scenes/custom2.json
ln -sf ../../assets/custom/custom3.json raycheck.out/custom_scenes/custom3.json
ln -sf ../../assets/custom/custom4.json raycheck.out/custom_scenes/custom4.json
python3 raycheck.py \
  --refbin ./ray-solution \
  --scenes raycheck.out/custom_scenes \
  --out    raycheck.out/custom \
  --i-understand-that-image-metrics-are-not-perfect


echo "Running cubemap-specific tests..."
mkdir -p raycheck.out/cubemap_scenes/trimesh
ln -sf ../../assets/scenes/scene_blank.json        raycheck.out/cubemap_scenes/scene_blank.json
ln -sf ../../../assets/scenes/trimesh/dragon4.json raycheck.out/cubemap_scenes/trimesh/dragon4.json

for cm_name in "${!CUBEMAPS[@]}"; do
  cm_face="${CUBEMAPS[$cm_name]}"
  if [[ ! -f "$cm_face" ]]; then
    echo "  Skipping $cm_name — face file not found: $cm_face"
    continue
  fi
  echo "  Testing cubemap: $cm_name"
  python3 raycheck.py \
    --refbin  ./ray-solution \
    --scenes  raycheck.out/cubemap_scenes \
    --cubemap "$cm_face" \
    --out     "raycheck.out/cubemap_${cm_name}" \
    --i-understand-that-image-metrics-are-not-perfect
done

# direct renders of each cubemap for visual comparison
echo "Rendering cubemap demo images..."
mkdir -p raycheck.out/cubemap_renders
for cm_name in "${!CUBEMAPS[@]}"; do
  cm_face="${CUBEMAPS[$cm_name]}"
  [[ ! -f "$cm_face" ]] && continue
  build/bin/ray -r 5 -w 512 \
    -c "$cm_face" \
    assets/scenes/scene_blank.json \
    "raycheck.out/cubemap_renders/${cm_name}_blank.png"
  build/bin/ray -r 5 -w 512 \
    -c "$cm_face" \
    assets/scenes/trimesh/dragon4.json \
    "raycheck.out/cubemap_renders/${cm_name}_dragon4.png"
  echo "  Saved: raycheck.out/cubemap_renders/${cm_name}_*.png"
done



echo "Running anti-aliasing tests (1x1, 2x2, 3x3, 4x4)..."
mkdir -p raycheck.out/aa_scenes/simple raycheck.out/aa_scenes/trimesh raycheck.out/aa
ln -sf ../../../assets/scenes/simple/box.json         raycheck.out/aa_scenes/simple/box.json
ln -sf ../../../assets/scenes/simple/texture_map.json raycheck.out/aa_scenes/simple/texture_map.json
ln -sf ../../../assets/scenes/simple/checkerboard.bmp raycheck.out/aa_scenes/simple/checkerboard.bmp
ln -sf ../../../assets/scenes/trimesh/cube.json       raycheck.out/aa_scenes/trimesh/cube.json

for sppd in 1 2 3 4; do
  cat > "raycheck.out/aa/aa_${sppd}.json" <<EOF
{
  "anti_alias": true,
  "supersamples": ${sppd}
}
EOF
  python3 raycheck.py \
    --refbin ./ray-solution \
    --scenes raycheck.out/aa_scenes \
    --json   "raycheck.out/aa/aa_${sppd}.json" \
    --out    "raycheck.out/aa/s${sppd}" \
    --i-understand-that-image-metrics-are-not-perfect
done


echo "Rendering portal demo scenes..."
mkdir -p raycheck.out/portal
build/bin/ray -r 5 -w 640 assets/custom/portal_rect.json   raycheck.out/portal/portal_rect.png
build/bin/ray -r 5 -w 640 assets/custom/portal_circle.json raycheck.out/portal/portal_circle.png

echo "Rendering overlap demo scene..."
mkdir -p raycheck.out/overlap
RAY_ENABLE_OVERLAP_REFRACTION=1 \
  build/bin/ray -r 7 -w 640 assets/custom/custom_overlap.json raycheck.out/overlap/custom_overlap.png


echo "Rendering harmonic tracing demos..."
mkdir -p raycheck.out/harmonic
RAY_ENABLE_HARMONIC_TRACING=1 RAY_HARMONIC_MODE=riemann \
  build/bin/ray -r 7 -w 640 assets/custom/harmonic_riemann.json raycheck.out/harmonic/harmonic_riemann.png
RAY_ENABLE_HARMONIC_TRACING=1 RAY_HARMONIC_MODE=gyroid \
  build/bin/ray -r 7 -w 640 assets/custom/harmonic_gyroid.json  raycheck.out/harmonic/harmonic_gyroid.png


echo "We done! "
echo "Baseline report:      raycheck.out/report.csv"
echo "Custom report:        raycheck.out/custom/report.csv"
echo "Cubemap reports:      raycheck.out/cubemap_<name>/report.csv (one per cubemap)"
echo "Cubemap renders:      raycheck.out/cubemap_renders/"
echo "AA reports:           raycheck.out/aa/s1/report.csv ... raycheck.out/aa/s4/report.csv"
echo "Portal demos:         raycheck.out/portal/"
echo "Overlap demo:         raycheck.out/overlap/custom_overlap.png"
echo "Harmonic demos:       raycheck.out/harmonic/"

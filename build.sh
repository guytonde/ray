set -euo pipefail

cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"
cd ..

mkdir -p raycheck.out/cubemap_maps
cp -f assets/scenes/skin1.bmp raycheck.out/cubemap_maps/cm_posx.bmp
cp -f assets/scenes/box_map1.bmp raycheck.out/cubemap_maps/cm_negx.bmp
cp -f assets/scenes/simple/checkerboard.bmp raycheck.out/cubemap_maps/cm_posy.bmp
cp -f assets/scenes/textry.png raycheck.out/cubemap_maps/cm_negy.png
cp -f assets/scenes/objmesh/glider.png raycheck.out/cubemap_maps/cm_posz.png
cp -f assets/scenes/objmesh/treelog_diffuse.png raycheck.out/cubemap_maps/cm_negz.png

echo "Running baseline test sweep..."
python3 raycheck.py \
  --refbin ./ray-solution \
  --out raycheck.out/base \
  --i-understand-that-image-metrics-are-not-perfect
cp -f raycheck.out/base/report.csv raycheck.out/report.csv

echo "Running custom scene sweep..."
mkdir -p raycheck.out/custom_scenes
ln -sf ../../assets/custom/custom.json raycheck.out/custom_scenes/custom.json
ln -sf ../../assets/custom/custom2.json raycheck.out/custom_scenes/custom2.json
ln -sf ../../assets/custom/custom3.json raycheck.out/custom_scenes/custom3.json
ln -sf ../../assets/custom/custom4.json raycheck.out/custom_scenes/custom4.json
python3 raycheck.py \
  --refbin ./ray-solution \
  --scenes raycheck.out/custom_scenes \
  --out raycheck.out/custom \
  --i-understand-that-image-metrics-are-not-perfect

echo "Running cubemap-specific tests..."
mkdir -p raycheck.out/cubemap_scenes/trimesh
ln -sf ../../assets/scenes/scene_blank.json raycheck.out/cubemap_scenes/scene_blank.json
ln -sf ../../../assets/scenes/trimesh/dragon4.json raycheck.out/cubemap_scenes/trimesh/dragon4.json
python3 raycheck.py \
  --refbin ./ray-solution \
  --scenes raycheck.out/cubemap_scenes \
  --cubemap raycheck.out/cubemap_maps/cm_posx.bmp \
  --out raycheck.out/cubemap \
  --i-understand-that-image-metrics-are-not-perfect

echo "Running anti-aliasing tests (1x1, 2x2, 3x3, 4x4)..."
mkdir -p raycheck.out/aa_scenes/simple raycheck.out/aa_scenes/trimesh raycheck.out/aa
ln -sf ../../../assets/scenes/simple/box.json raycheck.out/aa_scenes/simple/box.json
ln -sf ../../../assets/scenes/simple/texture_map.json raycheck.out/aa_scenes/simple/texture_map.json
ln -sf ../../../assets/scenes/simple/checkerboard.bmp raycheck.out/aa_scenes/simple/checkerboard.bmp
ln -sf ../../../assets/scenes/trimesh/cube.json raycheck.out/aa_scenes/trimesh/cube.json

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
    --json "raycheck.out/aa/aa_${sppd}.json" \
    --out "raycheck.out/aa/s${sppd}" \
    --i-understand-that-image-metrics-are-not-perfect
done

echo "Rendering portal demo scenes..."
mkdir -p raycheck.out/portal
build/bin/ray -r 5 -w 640 assets/custom/portal_rect.json raycheck.out/portal/portal_rect.png
build/bin/ray -r 5 -w 640 assets/custom/portal_circle.json raycheck.out/portal/portal_circle.png

echo "Rendering overlap demo scene..."
mkdir -p raycheck.out/overlap
RAY_ENABLE_OVERLAP_REFRACTION=1 \
  build/bin/ray -r 7 -w 640 assets/custom/custom_overlap.json raycheck.out/overlap/custom_overlap.png

echo "Done."
echo "Baseline report: raycheck.out/report.csv"
echo "Custom report: raycheck.out/custom/report.csv"
echo "Cubemap report: raycheck.out/cubemap/report.csv"
echo "AA reports: raycheck.out/aa/s1/report.csv ... raycheck.out/aa/s4/report.csv"
echo "Portal demos: raycheck.out/portal/portal_rect.png and raycheck.out/portal/portal_circle.png"
echo "Overlap demo: raycheck.out/overlap/custom_overlap.png"

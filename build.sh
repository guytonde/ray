#!/usr/bin/env bash
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

python3 raycheck.py --refbin ./ray-solution --cubemap raycheck.out/cubemap_maps/cm_posx.bmp

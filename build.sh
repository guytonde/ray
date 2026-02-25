cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
python3 raycheck_mt.py
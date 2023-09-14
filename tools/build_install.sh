cd /workspace
mkdir -p build && cd build
cmake .. -DWITH_GPU=ON -DWITH_PROFILER=ON -DCMAKE_BUILD_TYPE=Debug
make -j 
pip uninstall -y turbo-transformers
pip install  `find . -name *whl`
# pip install --force-reinstall `find ./turbo_transformers -name *whl`

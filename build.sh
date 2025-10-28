#!/bin/bash
# Script to build the ParallelOptimizationEngine project
if [ -d "build" ]; then
echo "Cleaning previous build..."
cd build
make clean || true
rm -f CMakeCache.txt
cd ..
rm -rf build
fi
echo "Creating new build directory..."
mkdir -p build && cd build
echo "Running CMake..."
cmake .. || { echo "CMake failed"; exit 1; }
echo "Building project..."
make -j$(nproc) || { echo "Make failed"; exit 1; }
echo "Copying Python bindings..."
cp src/python_bindings/poe_bindings*.so ../python/ || { echo "Copy failed"; exit 1; }
echo "Build successful! Run from python/ directory."
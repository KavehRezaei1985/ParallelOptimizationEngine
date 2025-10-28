# Dockerfile for ParallelOptimizationEngine
#
# Purpose: Provides a reproducible, containerized environment for building and running the ParallelOptimizationEngine.
# This ensures consistency across development, testing, and deployment, mitigating issues with host-specific dependencies
# (e.g., CUDA versions, library installations). Uses a CUDA devel base for GPU support in simulations.
#
# Key Features:
# - CUDA-enabled base image for GPU acceleration.
# - Installs essential build tools, OpenMP (for parallel CPU strategies), and Python dependencies.
# - Builds the project in-container using CMake and make.
# - Supports volume mounts to persist output files (e.g., scaling.png, CSVs).
#
# Build Command: docker build -t poe:latest .
# Run Command Example: docker run --gpus all --rm -v $(pwd)/python:/app/python poe:latest --method=both --mode=all
# (Use --gpus all for NVIDIA GPU access; volume mount persists outputs like scaling.png.)
#
# Best Practices Applied:
# - Non-interactive mode to avoid build-time prompts.
# - Cache cleanup to minimize image size.
# - Error handling in RUN commands for robust builds.
# - Fallback for PyTorch installation to handle CUDA mismatches.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# Set environment variable for non-interactive package installation to prevent prompts during apt-get.
ENV DEBIAN_FRONTEND=noninteractive
# Install core system dependencies:
# - cmake, g++: For building C++ components.
# - libomp-dev: For OpenMP support in parallel strategies.
# - python3, python3-pip, python3-dev: For Python orchestration and ML.
# - pybind11-dev: For C++/Python bindings.
# - git: For potential repo cloning (if needed in extensions).
# - python3-tk, python3-pil: Optional for interactive visualization (comment out if not needed).
# Cleanup apt cache to reduce image size.
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    libomp-dev \
    python3 \
    python3-pip \
    python3-dev \
    pybind11-dev \
    git \
    # python3-tk \
    # python3-pil \
    && rm -rf /var/lib/apt/lists/*
# Optional: Upgrade CMake to a newer version if the base image's is insufficient.
# Uses Kitware repo for latest stable; || true allows build to continue if already up-to-date.
RUN wget -O - https://apt.kitware.com/kitware-archive.sh | bash && \
    apt-get install -y cmake=3.27.7-0kitware1ubuntu22.04.1 || true
# Set the working directory inside the container for organizing files.
WORKDIR /app
# Copy the entire repository into the container.
# Rationale: Ensures all source files are available for building; use .dockerignore to exclude unnecessary files.
COPY . /app
# Copy and install Python dependencies from requirements.txt.
# Fallback installs PyTorch with specific CUDA version if default fails (e.g., version mismatch).
# --no-cache-dir minimizes pip cache size in the image.
COPY python/requirements.txt /app/python/requirements.txt
RUN pip3 install --no-cache-dir -r /app/python/requirements.txt || \
    pip3 install --no-cache-dir torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# Build the project in-container by inlining build.sh logic.
# Handles cleaning previous builds, CMake configuration, parallel make, and copying bindings.
# Error exits ensure build fails fast on issues; uses all CPU cores with -j$(nproc).
RUN if [ -d "build" ]; then \
        echo "Cleaning previous build..." && \
        cd build && \
        make clean || true && \
        rm -f CMakeCache.txt && \
        cd .. && \
        rm -rf build; \
    fi && \
    echo "Creating new build directory..." && \
    mkdir -p build && cd build && \
    echo "Running CMake..." && \
    cmake .. || { echo "CMake failed"; exit 1; } && \
    echo "Building project..." && \
    make -j$(nproc) || { echo "Make failed"; exit 1; } && \
    echo "Copying Python bindings..." && \
    cp src/python_bindings/poe_bindings*.so ../python/ || { echo "Copy failed"; exit 1; } && \
    echo "Build successful! Run from python/ directory."
# Set ENTRYPOINT to run the main simulation script by default.
# Allows passing arguments (e.g., --N 1000) when running the container.
# CMD provides default args for convenience.
ENTRYPOINT ["python3", "/app/python/run_simulation.py"]
CMD ["--method=both", "--mode=all"]

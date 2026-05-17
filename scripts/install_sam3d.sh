#!/bin/bash

# Installation script for SAM3D backend.
# Works with any CUDA 12.x installation (system-wide, conda, or custom).
# See https://github.com/facebookresearch/sam-3d-objects for more information.

set -euo pipefail

SAM3D_OBJECTS_COMMIT="${SAM3D_OBJECTS_COMMIT:-81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd}"
SAM3_COMMIT="${SAM3_COMMIT:-11dec2936de97f2857c1f76b66d982d5a001155d}"

REPO_ROOT=$(pwd)

# Prefer the project's Python environment when available so header detection and
# package installs target the same interpreter.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
else
    PYTHON_BIN=$(command -v python3)
fi

UV_PIP=(uv pip install --python "$PYTHON_BIN")

echo "========================================="
echo "SAM3D Installation Script"
echo "========================================="
echo ""
echo "Using Python: $PYTHON_BIN"
echo ""

TOTAL_MEM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)
CPU_COUNT=$(nproc)
IS_WSL=false
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    IS_WSL=true
fi

# Check for Python development headers (required for nvdiffrast JIT compilation).
echo "Step 0: Checking system dependencies..."

PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_INCLUDE_DIR=$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or "")')
PYTHON_HEADER="${PYTHON_INCLUDE_DIR}/pyconfig.h"

if [ ! -f "$PYTHON_HEADER" ]; then
    # Fall back to common distro include locations when Python reports an empty or
    # incomplete include path.
    for candidate in \
        "/usr/include/x86_64-linux-gnu/python${PYTHON_VERSION}/pyconfig.h" \
        "/usr/include/python${PYTHON_VERSION}/pyconfig.h"
    do
        if [ -f "$candidate" ]; then
            PYTHON_HEADER="$candidate"
            break
        fi
    done
fi

if [ -f "$PYTHON_HEADER" ]; then
    echo "✓ Python development headers found at $PYTHON_HEADER"
else
    echo "⚠️  Python development headers not found for Python ${PYTHON_VERSION}"
    echo "   These are required for nvdiffrast JIT compilation (texture baking)."
    echo ""

    PYTHON_DEV_PACKAGE=""
    for package in "python${PYTHON_VERSION}-dev" "libpython${PYTHON_VERSION}-dev"; do
        if apt-cache show "$package" >/dev/null 2>&1; then
            PYTHON_DEV_PACKAGE="$package"
            break
        fi
    done

    if [ -n "$PYTHON_DEV_PACKAGE" ]; then
        read -p "Install ${PYTHON_DEV_PACKAGE}? (requires sudo) [Y/n]: " -n 1 -r
        echo ""

        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "Installing Python development headers..."
            sudo apt-get update && sudo apt-get install -y "$PYTHON_DEV_PACKAGE"
            echo "✓ Installed $PYTHON_DEV_PACKAGE"
        else
            echo "⚠️  Skipping. nvdiffrast texture baking may fail without Python headers."
            echo "   Install manually: sudo apt-get install $PYTHON_DEV_PACKAGE"
        fi
    else
        echo "⚠️  No matching apt package found for Python ${PYTHON_VERSION}."
        echo "   If you're using a uv-managed Python, recreate the venv with a bundled interpreter"
        echo "   or install a matching python-dev package from an external repository."
    fi
fi

echo ""

# Auto-detect and validate CUDA installation.
echo "Step 1: Detecting CUDA installation..."

TORCH_CUDA_VERSION=$("$PYTHON_BIN" - <<'PYEOF'
try:
    import torch
    print(torch.version.cuda or "")
except Exception:
    print("")
PYEOF
)

FOUND_IN_PATH=false
PREFERRED_CUDA_HOME=""

if [ -n "$TORCH_CUDA_VERSION" ] && [ -x "/usr/local/cuda-${TORCH_CUDA_VERSION}/bin/nvcc" ]; then
    PREFERRED_CUDA_HOME="/usr/local/cuda-${TORCH_CUDA_VERSION}"
elif [ -n "$TORCH_CUDA_VERSION" ] && [ -x "/usr/local/cuda-${TORCH_CUDA_VERSION%.*}/bin/nvcc" ]; then
    PREFERRED_CUDA_HOME="/usr/local/cuda-${TORCH_CUDA_VERSION%.*}"
fi

if [ -n "$PREFERRED_CUDA_HOME" ]; then
    NVCC_PATH="$PREFERRED_CUDA_HOME/bin/nvcc"
    export PATH="$PREFERRED_CUDA_HOME/bin:$PATH"
    FOUND_IN_PATH=true
    echo "✓ Selected CUDA toolkit matching PyTorch at $PREFERRED_CUDA_HOME"
elif command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    FOUND_IN_PATH=true
else
    # Check common CUDA installation locations.
    for cuda_path in /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-13 ~/miniforge3 ~/miniconda3 ~/anaconda3; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            echo "✓ Found CUDA installation at $cuda_path"
            NVCC_PATH="$cuda_path/bin/nvcc"
            export PATH="$cuda_path/bin:$PATH"
            FOUND_IN_PATH=true
            break
        fi
    done
fi

if [ "$FOUND_IN_PATH" = true ]; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+")
    echo "✓ Found CUDA $CUDA_VERSION"

    if [ -n "$TORCH_CUDA_VERSION" ]; then
        echo "✓ PyTorch expects CUDA $TORCH_CUDA_VERSION"
        TORCH_VERSION=$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)')

        CUDA_MAJOR=${CUDA_VERSION%%.*}
        TORCH_CUDA_MAJOR=${TORCH_CUDA_VERSION%%.*}

        if [ "$CUDA_MAJOR" != "$TORCH_CUDA_MAJOR" ]; then
            echo "✗ Error: nvcc reports CUDA $CUDA_VERSION, but PyTorch was built for CUDA $TORCH_CUDA_VERSION"
            echo ""
            echo "Building CUDA extensions against a different major CUDA runtime can fail or"
            echo "produce incorrect behavior."
            echo ""
            echo "Recommended fixes:"
            echo "  1. Use a CUDA $TORCH_CUDA_VERSION toolkit (best for this environment)"
            echo "  2. Recreate the Python env with a PyTorch build that matches your installed CUDA toolkit"
            echo ""
            TORCH_CUDA_TAG=$(printf '%s' "$TORCH_CUDA_VERSION" | tr -d .)
            echo "For this repo today, using CUDA 12.4 is the safest path because the current env"
            echo "has torch ${TORCH_VERSION} built for cu${TORCH_CUDA_TAG}."
            exit 1
        fi

        if [ "$CUDA_VERSION" != "$TORCH_CUDA_VERSION" ]; then
            echo "⚠️  Toolkit version ($CUDA_VERSION) differs from PyTorch CUDA version ($TORCH_CUDA_VERSION)."
            echo "   Same-major combinations often work, but this is less tested than an exact match."
        fi
    else
        echo "⚠️  Could not detect PyTorch CUDA version from $PYTHON_BIN."
        if [[ ! "$CUDA_VERSION" =~ ^12\. && ! "$CUDA_VERSION" =~ ^13\. ]]; then
            echo "✗ Error: Unsupported CUDA toolkit version $CUDA_VERSION"
            echo ""
            echo "Please install CUDA 12.x or a CUDA 13.x environment that matches your PyTorch build."
            exit 1
        fi
    fi

    if [[ "$CUDA_VERSION" =~ ^13\. ]]; then
        export NVCC_FLAGS="-static-global-template-stub=false${NVCC_FLAGS:+ $NVCC_FLAGS}"
        echo "⚠️  CUDA 13 detected. Applying PyTorch3D NVCC workaround:"
        echo "   NVCC_FLAGS=-static-global-template-stub=false"
        echo "   Note: the SAM3D dependency stack is not fully validated on CUDA 13."
    fi

    # Auto-detect CUDA_HOME from nvcc location.
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    echo "✓ Using CUDA_HOME: $CUDA_HOME"

    # Set LD_LIBRARY_PATH (handle both lib64 and lib for conda compatibility).
    if [ -d "$CUDA_HOME/lib64" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "✓ Added $CUDA_HOME/lib64 to LD_LIBRARY_PATH"
    elif [ -d "$CUDA_HOME/lib" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "✓ Added $CUDA_HOME/lib to LD_LIBRARY_PATH"
    fi

else
    echo "✗ nvcc not found in PATH"
    echo ""
    echo "SAM3D requires CUDA 12.x toolkit to build dependencies (pytorch3d, gsplat, etc.)"
    echo ""
    read -p "Install CUDA 12.4 system-wide? [y/N]: " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading CUDA 12.4 installer..."
        CUDA_INSTALLER="cuda_12.4.0_550.54.14_linux.run"
        CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/${CUDA_INSTALLER}"

        if ! wget -q --show-progress "$CUDA_URL"; then
            echo "✗ Download failed"
            exit 1
        fi

        echo ""
        echo "Installing CUDA 12.4 toolkit (requires sudo)..."
        echo "This will install to /usr/local/cuda-12.4"
        echo ""

        if sudo sh "$CUDA_INSTALLER" --silent --toolkit; then
            echo ""
            echo "✓ CUDA 12.4 installed successfully"

            # Set environment variables for this session.
            export CUDA_HOME="/usr/local/cuda-12.4"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Verify installation.
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+")
                echo "✓ Verified: nvcc $CUDA_VERSION available"
            else
                echo "✗ Installation verification failed"
                exit 1
            fi

            # Clean up installer.
            rm "$CUDA_INSTALLER"
            echo "✓ Cleaned up installer"
        else
            echo "✗ Installation failed"
            rm -f "$CUDA_INSTALLER"
            exit 1
        fi

    else
        echo ""
        echo "Manual installation options:"
        echo "  1. System-wide: https://developer.nvidia.com/cuda-downloads"
        echo "  2. Conda: conda install cuda-toolkit=12.4 -c nvidia"
        echo ""
        echo "After installation, ensure 'nvcc' is in your PATH and re-run this script."
        exit 1
    fi
fi

echo ""
echo "Step 2: Cloning repositories..."

# Limit parallel compilation for heavy CUDA/C++ extensions. PyTorch3D and
# Kaolin can otherwise spawn enough compiler processes to OOM WSL2 even on
# machines with many CPU cores.
if [ -n "${SAM3D_BUILD_JOBS:-}" ]; then
    BUILD_JOBS="$SAM3D_BUILD_JOBS"
else
    BUILD_JOBS=$((TOTAL_MEM_GB / 6))
    if [ "$BUILD_JOBS" -lt 1 ]; then
        BUILD_JOBS=1
    fi
    if [ "$BUILD_JOBS" -gt "$CPU_COUNT" ]; then
        BUILD_JOBS="$CPU_COUNT"
    fi
    if [ "$IS_WSL" = true ] && [ "$BUILD_JOBS" -gt 4 ]; then
        BUILD_JOBS=4
    elif [ "$IS_WSL" = false ] && [ "$BUILD_JOBS" -gt 8 ]; then
        BUILD_JOBS=8
    fi
fi

export MAX_JOBS="$BUILD_JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
export MAKEFLAGS="-j$BUILD_JOBS"
echo "✓ Limiting extension builds to $BUILD_JOBS parallel jobs"
if [ "$IS_WSL" = true ]; then
    echo "  WSL detected; conservative build parallelism helps avoid OOM kills"
fi

# Create external directory if it doesn't exist.
mkdir -p external
cd external

# Clone SAM 3D Objects repository.
if [ ! -d "sam-3d-objects" ]; then
    echo "Cloning SAM 3D Objects repository..."
    git clone https://github.com/facebookresearch/sam-3d-objects.git
    echo "✓ Cloned sam-3d-objects"
else
    echo "✓ sam-3d-objects already exists"
fi
echo "Checking out SAM 3D Objects commit: ${SAM3D_OBJECTS_COMMIT}"
git -C sam-3d-objects fetch origin
git -C sam-3d-objects checkout --detach "${SAM3D_OBJECTS_COMMIT}"

# Clone SAM3 repository.
if [ ! -d "SAM3" ]; then
    echo "Cloning SAM3 repository..."
    git clone https://github.com/facebookresearch/sam3.git SAM3
    echo "✓ Cloned SAM3"
else
    echo "✓ SAM3 already exists"
fi
echo "Checking out SAM3 commit: ${SAM3_COMMIT}"
git -C SAM3 fetch origin
git -C SAM3 checkout --detach "${SAM3_COMMIT}"

echo ""
echo "Step 3: Installing SAM3..."
cd SAM3

# Install SAM3 with notebooks extras (includes inference dependencies).
# This includes: decord, pycocotools, opencv-python, einops, scikit-image, scikit-learn.
echo "Installing SAM3 with inference dependencies..."
"${UV_PIP[@]}" -e ".[notebooks]"
cd ..
echo "✓ SAM3 installed"

echo ""
echo "Step 4: Installing SAM 3D Objects dependencies..."
echo "This will install dependencies from requirements.txt and build CUDA packages."
echo "This may take 10-20 minutes..."
echo ""

cd sam-3d-objects

# First install non-CUDA dependencies from requirements.txt.
# Filter out packages that conflict with our environment or aren't needed.
echo "Installing sam-3d-objects core dependencies..."
grep -v -E "^(torch|torchvision|torchaudio|cuda-python|nvidia-|MoGe|flash_attn|bpy|wandb|jupyter|tensorboard|Flask|webdataset|sagemaker)" requirements.txt > /tmp/filtered_requirements.txt
"${UV_PIP[@]}" -r /tmp/filtered_requirements.txt

# Now install CUDA-dependent packages with --no-build-isolation.
echo ""
echo "Installing gsplat (requires PyTorch at build time)..."
"${UV_PIP[@]}" --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

echo ""
echo "Installing nvdiffrast (requires CUDA)..."
"${UV_PIP[@]}" --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"

echo ""
echo "Pre-compiling nvdiffrast CUDA extensions..."
echo "(This triggers PyTorch JIT compilation - may take 1-2 minutes)"

# Pre-compilation script - ensures nvdiffrast is ready to use.
if "$PYTHON_BIN" << 'PYEOF'
import sys
import os

try:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available - pre-compilation will happen on first use")
        sys.exit(0)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print("Compiling nvdiffrast CUDA kernels...")

    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()

    # Verify compilation.
    import torch.utils.cpp_extension as cpp_ext
    build_dir = cpp_ext._get_build_directory("nvdiffrast_plugin", False)
    so_path = os.path.join(build_dir, "nvdiffrast_plugin.so")

    if os.path.exists(so_path):
        size_mb = os.path.getsize(so_path) / (1024 * 1024)
        print(f"SUCCESS: {so_path} ({size_mb:.1f} MB)")
    else:
        print("WARNING: .so file not found, compilation may have failed")
        sys.exit(1)

except Exception as e:
    print(f"Pre-compilation failed: {e}")
    print("NOTE: nvdiffrast will compile on first SAM3D use (~2-5 min delay)")
    sys.exit(0)  # Non-fatal
PYEOF
then
    echo "✓ nvdiffrast pre-compiled successfully"
else
    echo "⚠️  nvdiffrast pre-compilation skipped (will compile on first use)"
fi

echo ""
echo "Installing kaolin 0.17.0 (requires CUDA, building from source)..."
"${UV_PIP[@]}" --no-build-isolation \
    "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0"

echo ""
echo "Installing pytorch3d from source..."
"${UV_PIP[@]}" --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

# Install inference-specific requirements.
echo ""
echo "Installing inference dependencies..."
"${UV_PIP[@]}" seaborn==0.13.2 gradio==5.49.0 imageio utils3d

# Install MoGe (depth model used by SAM 3D Objects).
echo ""
echo "Installing MoGe depth model..."
"${UV_PIP[@]}" "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

cd ..

echo ""
echo "✓ All dependencies installed"

echo ""
echo "Step 5: Downloading model checkpoints..."
echo ""

# Create checkpoints directory.
mkdir -p checkpoints

# Choose download source.
CHECKPOINT_SOURCE=${SCENESMITH_CHECKPOINT_SOURCE:-}
if [ -z "$CHECKPOINT_SOURCE" ]; then
    echo "Choose checkpoint download source:"
    echo "  1. HuggingFace (official source, requires access approval + login)"
    echo "  2. ModelScope (domestic mirror, no HuggingFace CLI needed)"
    read -p "Select source [1/2, default: 1]: " CHECKPOINT_SOURCE_CHOICE
    case "${CHECKPOINT_SOURCE_CHOICE:-1}" in
        2) CHECKPOINT_SOURCE="modelscope" ;;
        *) CHECKPOINT_SOURCE="huggingface" ;;
    esac
fi

if [ "$CHECKPOINT_SOURCE" = "huggingface" ]; then
    echo ""
    echo "⚠️  Important: HuggingFace authentication required!"
    echo "    1. Request access: https://huggingface.co/facebook/sam3"
    echo "    2. Request access: https://huggingface.co/facebook/sam-3d-objects"
    echo "    3. Login: hf auth login (or huggingface-cli login)"
    echo ""
    read -p "Have you requested access and logged in? [y/N]: " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please complete authentication steps above and re-run the script."
        exit 1
    fi

    # Download SAM3 checkpoint.
    if [ ! -f "checkpoints/sam3.pt" ]; then
        echo "Downloading SAM3 checkpoint (sam3.pt) from HuggingFace..."
        hf download facebook/sam3 sam3.pt --local-dir checkpoints
        echo "✓ Downloaded sam3.pt"
    else
        echo "✓ sam3.pt already exists"
    fi

    # Download SAM 3D Objects checkpoints (entire checkpoints folder).
    if [ ! -f "checkpoints/.sam3d_objects_downloaded" ]; then
        echo "Downloading SAM 3D Objects checkpoints from HuggingFace..."
        hf download facebook/sam-3d-objects \
            --repo-type model \
            --local-dir checkpoints/sam-3d-objects-download \
            --include "checkpoints/*"

        # Move checkpoints to correct location.
        mv checkpoints/sam-3d-objects-download/checkpoints/* checkpoints/
        rm -rf checkpoints/sam-3d-objects-download
        touch checkpoints/.sam3d_objects_downloaded
        echo "✓ Downloaded SAM 3D Objects checkpoints"
    else
        echo "✓ SAM 3D Objects checkpoints already exist"
    fi
elif [ "$CHECKPOINT_SOURCE" = "modelscope" ]; then
    echo "Using ModelScope mirror for checkpoint download..."
    echo "  SAM3: https://www.modelscope.cn/models/facebook/sam3/summary"
    echo "  SAM 3D Objects: https://www.modelscope.cn/models/facebook/sam-3d-objects/summary"
    echo ""
    echo "Installing ModelScope client..."
    "${UV_PIP[@]}" modelscope
    echo "Downloading checkpoints from ModelScope with resume support..."
    "$PYTHON_BIN" "$REPO_ROOT/scripts/download_modelscope_checkpoints.py" \
        --output-dir "$PWD/checkpoints"
else
    echo "✗ Unknown checkpoint source: $CHECKPOINT_SOURCE"
    echo "  Supported values: huggingface, modelscope"
    exit 1
fi

cd ..

echo ""
echo "========================================="
echo "SAM3D Installation Complete!"
echo "========================================="
echo ""
echo "Checkpoints located in: external/checkpoints/"
echo "  SAM3: external/checkpoints/sam3.pt"
echo "  SAM 3D Objects: external/checkpoints/*.{ckpt,pt,yaml}"
echo ""
echo "To use SAM3D backend, update your config:"
echo "  asset_manager:"
echo "    backend: \"sam3d\""
echo "    sam3d:"
echo "      sam3_checkpoint: \"external/checkpoints/sam3.pt\""
echo "      sam3d_checkpoint: \"external/checkpoints/pipeline.yaml\""
echo ""
echo "Note: SAM 3D Objects uses pipeline.yaml which references other checkpoints."
echo ""

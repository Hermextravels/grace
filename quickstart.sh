#!/bin/bash
# Quick start script for Hybrid Bitcoin Puzzle Solver

set -e

echo "=== Hybrid Solver Quick Start ==="
echo ""

# Check dependencies
echo "[1/5] Checking dependencies..."

if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found. Please install build tools."
    exit 1
fi

if ! command -v pkg-config &> /dev/null; then
    echo "WARNING: pkg-config not found. Will try default paths."
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    echo "  ✓ CUDA detected: $(nvcc --version | grep release | cut -d' ' -f5)"
    BUILD_TYPE="GPU"
else
    echo "  ⚠ CUDA not found - will build CPU-only version"
    BUILD_TYPE="CPU"
fi

# Clean previous build
echo ""
echo "[2/5] Cleaning previous build..."
make clean 2>/dev/null || true

# Build
echo ""
echo "[3/5] Building hybrid_solver ($BUILD_TYPE mode)..."
if [ "$BUILD_TYPE" = "GPU" ]; then
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    make cpu-only -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

if [ ! -f "./hybrid_solver" ]; then
    echo "ERROR: Build failed. Check errors above."
    exit 1
fi

echo "  ✓ Build successful"

# Check target file
echo ""
echo "[4/5] Checking puzzle data..."
if [ ! -f "data/puzzle71_target.txt" ]; then
    echo "ERROR: data/puzzle71_target.txt not found"
    exit 1
fi
echo "  ✓ Target file ready"

# Show usage
echo ""
echo "[5/5] Ready to run!"
echo ""
echo "─────────────────────────────────────────────────"
echo "Example commands:"
echo ""
echo "# Test run (small range, 2 threads):"
echo "./hybrid_solver 1000000 2000000 data/test_addresses.txt 2"
echo ""
echo "# Puzzle 71 (full range, 8 threads):"
echo "./hybrid_solver 20000000000000000 3ffffffffffffffffff data/puzzle71_target.txt 8"
echo ""
echo "# Resume from checkpoint (same command):"
echo "./hybrid_solver 20000000000000000 3ffffffffffffffffff data/puzzle71_target.txt 8"
echo ""
echo "─────────────────────────────────────────────────"
echo ""
echo "Tips:"
echo "  • Press Ctrl+C to stop (progress auto-saves)"
echo "  • Monitor with: watch -n5 cat hybrid_solver.checkpoint"
echo "  • Check speed: look for 'MKeys/s' in output"
echo "  • GPU usage: nvidia-smi -l 1"
echo ""
echo "Good luck! 🍀"

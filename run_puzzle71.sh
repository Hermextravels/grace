#!/bin/bash
# Quick launcher for Puzzle 71 (most feasible unsolved puzzle)

echo "================================"
echo "  Hybrid Solver - Puzzle #71"
echo "  Range: 71 bits"
echo "  Target: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
echo "  Prize: 7.1 BTC"
echo "================================"
echo ""

# Check if compiled
if [ ! -f "./hybrid_solver" ]; then
    echo "[!] Executable not found. Building..."
    make clean && make
    if [ $? -ne 0 ]; then
        echo "[!] Build failed. Please check errors above."
        exit 1
    fi
fi

# Run with optimal settings for puzzle 71
# - 8 threads (adjust based on your CPU)
# - GPU enabled if available
# - Checkpoint every 60 seconds
# - Stride: 100M for good coverage

# Run with correct paths
./hybrid_solver \
    400000000000000000 \
    7fffffffffffffffff \
    data/puzzle71_target.txt \
    8

echo ""
echo "[*] Solver stopped. Checkpoint saved."
echo "[*] To resume: ./run_puzzle71.sh"

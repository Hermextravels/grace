#!/bin/bash
# Run solver on ALL feasible puzzles (71-99) simultaneously
# Each puzzle gets a portion of GPU/CPU resources

echo "========================================"
echo "  Hybrid Solver - ALL FEASIBLE PUZZLES"
echo "  Target: Puzzles 71-99 (no pubkeys)"
echo "  Total Prize: ~200+ BTC"
echo "========================================"
echo ""

# Check if compiled
if [ ! -f "./hybrid_solver" ]; then
    echo "[!] Building solver..."
    make clean && make
    if [ $? -ne 0 ]; then
        echo "[!] Build failed."
        exit 1
    fi
fi

# Load all addresses into bloom filter for multi-target search
# This is MUCH more efficient than running separate processes

echo "[+] Loading all target addresses (puzzles 71-99)..."
echo "[+] Starting multi-target solver..."
echo ""

# Run with all puzzles loaded (create combined addresses file)
if [ ! -f data/all_targets.txt ]; then
    echo "[*] Creating combined target file..."
    grep -v "^#" data/unsolved_71_99.txt | cut -d',' -f5 > data/all_targets.txt
fi

./hybrid_solver \
    400000000000000000 \
    7ffffffffffffffffffffffff \
    data/all_targets.txt \
    16

echo ""
echo "[*] Multi-target solver stopped."
echo "[*] Check hybrid_solver.checkpoint for progress"

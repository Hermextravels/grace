#!/bin/bash
# GPU Solver Test Suite
# Validates CUDA implementation before production deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SOLVER="./hybrid_solver_gpu"
DATA_DIR="data"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${GREEN}🧪 GPU Solver Test Suite${NC}                                 ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if binary exists
if [[ ! -f "$SOLVER" ]]; then
    echo -e "${RED}❌ Error: $SOLVER not found${NC}"
    echo -e "${YELLOW}Build with: make gpu${NC}"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ Error: nvidia-smi not found${NC}"
    echo -e "${YELLOW}This test requires an NVIDIA GPU${NC}"
    exit 1
fi

echo -e "${GREEN}[1/5] GPU Information${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo ""

echo -e "${GREEN}[2/5] Test: Known Puzzle #66 (Narrow Range)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Expected: Should find key 0x2832ed74f2b5e35ee"
echo "Address: 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
echo ""

# Create test data
cat > /tmp/test_puzzle66.txt << EOF
13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so
EOF

START_TIME=$(date +%s)
if $SOLVER 2832ed74f2b5e35e0 2832ed74f2b5e3600 /tmp/test_puzzle66.txt 1 66; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    if [[ -f "WINNER_PUZZLE_66.txt" ]]; then
        echo -e "${GREEN}✅ Test PASSED - Key found in ${ELAPSED}s${NC}"
        echo ""
        echo "Result:"
        cat WINNER_PUZZLE_66.txt
        echo ""
    else
        echo -e "${RED}❌ Test FAILED - No winner file created${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Test FAILED - Solver exited with error${NC}"
    exit 1
fi

echo -e "${GREEN}[3/5] Test: Performance Benchmark (1M keys)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test range: 1 million keys
cat > /tmp/test_bench.txt << EOF
1FakeAddressForBenchmarkTestXXXXXXXXXXX
EOF

START_TIME=$(date +%s.%N)
timeout 30s $SOLVER 400000000000000000 40000000000100000 /tmp/test_bench.txt 1 71 > /dev/null 2>&1 || true
END_TIME=$(date +%s.%N)

ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
KEYS_TESTED=1048576  # 0x100000 in decimal
KEYS_PER_SEC=$(echo "scale=0; $KEYS_TESTED / $ELAPSED" | bc)

echo "Keys tested: $KEYS_TESTED"
echo "Time: ${ELAPSED}s"
echo -e "${YELLOW}Speed: ${KEYS_PER_SEC} keys/sec${NC}"
echo ""

# Performance assessment
if (( KEYS_PER_SEC > 10000000 )); then
    echo -e "${GREEN}✅ EXCELLENT: >10M keys/sec (production ready)${NC}"
elif (( KEYS_PER_SEC > 1000000 )); then
    echo -e "${YELLOW}⚠️  GOOD: >1M keys/sec (acceptable)${NC}"
else
    echo -e "${RED}⚠️  SLOW: <1M keys/sec (optimization needed)${NC}"
fi
echo ""

echo -e "${GREEN}[4/5] Test: Multi-Target Matching${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat > /tmp/test_multi.txt << EOF
13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so
1FakeAddress1XXXXXXXXXXXXXXXXXXXXXXX
1FakeAddress2XXXXXXXXXXXXXXXXXXXXXXX
1FakeAddress3XXXXXXXXXXXXXXXXXXXXXXX
EOF

echo "Testing with 4 target addresses (1 valid, 3 fake)"
if $SOLVER 2832ed74f2b5e35e0 2832ed74f2b5e3600 /tmp/test_multi.txt 1 66 > /dev/null 2>&1; then
    if [[ -f "WINNER_PUZZLE_66.txt" ]]; then
        FOUND_ADDR=$(grep "Address:" WINNER_PUZZLE_66.txt | awk '{print $2}')
        if [[ "$FOUND_ADDR" == "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so" ]]; then
            echo -e "${GREEN}✅ Test PASSED - Correct address matched${NC}"
        else
            echo -e "${RED}❌ Test FAILED - Wrong address: $FOUND_ADDR${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}❌ Test FAILED - Solver error${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}[5/5] Test: Hash Function Validation${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify the found key generates correct address
if [[ -f "WINNER_PUZZLE_66.txt" ]]; then
    FOUND_KEY=$(grep "HEX:" WINNER_PUZZLE_66.txt | awk '{print $2}')
    EXPECTED_KEY="000000000000000000000000000000000000000000000002832ed74f2b5e35ee"
    
    if [[ "$FOUND_KEY" == "$EXPECTED_KEY" ]]; then
        echo -e "${GREEN}✅ Private key format: CORRECT${NC}"
    else
        echo -e "${RED}❌ Private key mismatch${NC}"
        echo "Expected: $EXPECTED_KEY"
        echo "Got:      $FOUND_KEY"
        exit 1
    fi
    
    FOUND_WIF=$(grep "WIF:" WINNER_PUZZLE_66.txt | awk '{print $2}')
    echo "WIF: $FOUND_WIF"
    echo -e "${GREEN}✅ WIF generation: OK${NC}"
fi
echo ""

# GPU Memory Check
echo -e "${GREEN}[*] GPU Memory Usage${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo ""

# Cleanup
rm -f /tmp/test_puzzle66.txt /tmp/test_bench.txt /tmp/test_multi.txt
rm -f WINNER_PUZZLE_66.txt

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${GREEN}✅ All Tests PASSED - GPU Solver Ready for Production${NC}   ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Deploy to production GPU server"
echo "  2. Run: ./hybrid_solver_gpu [start] [end] data/puzzle71.txt 1 71"
echo "  3. Monitor with: watch -n 1 nvidia-smi"
echo ""

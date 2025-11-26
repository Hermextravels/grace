# Hybrid Bitcoin Puzzle Solver - Makefile
# Supports: CPU-only build or CPU+GPU (CUDA) build

CXX = g++
NVCC = nvcc
TARGET = hybrid_solver
TARGET_GMP = hybrid_solver_gmp
TARGET_GPU = hybrid_solver_gpu

# Directories
SRC_DIR = src
INC_DIR = include
DATA_DIR = data
BUILD_DIR = build

# Detect CUDA
CUDA_PATH ?= /usr/local/cuda
CUDA_AVAILABLE := $(shell command -v nvcc 2> /dev/null)

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -march=native -pthread -Wall -Wextra
CXXFLAGS += -I$(INC_DIR)
LDFLAGS = -pthread

# CUDA flags (for GPU build)
NVCCFLAGS = -O3 -std=c++14 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 \
	-gencode=arch=compute_89,code=sm_89 \
	-gencode=arch=compute_90,code=sm_90
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

# OpenSSL for hash functions
OPENSSL_FLAGS = $(shell pkg-config --cflags openssl 2>/dev/null || echo "-I/usr/local/opt/openssl/include")
OPENSSL_LIBS = $(shell pkg-config --libs openssl 2>/dev/null || echo "-L/usr/local/opt/openssl/lib -lssl -lcrypto")

CXXFLAGS += $(OPENSSL_FLAGS)
LDFLAGS += $(OPENSSL_LIBS)

# GMP for big integer math
GMP_FLAGS = $(shell pkg-config --cflags gmp 2>/dev/null || echo "")
GMP_LIBS = $(shell pkg-config --libs gmp 2>/dev/null || echo "-lgmp")

# libsecp256k1 (Bitcoin Core's official EC library)
SECP256K1_FLAGS = -I/usr/local/include
SECP256K1_LIBS = -L/usr/local/lib -lsecp256k1

# Source files (original 64-bit version)
CPP_SOURCES = $(SRC_DIR)/main.cpp \
              $(SRC_DIR)/secp256k1_tiny.cpp \
              $(SRC_DIR)/bloom_filter.cpp \
              $(SRC_DIR)/base58.cpp

# GMP version sources (128-bit+ support - custom EC)
CPP_SOURCES_GMP = $(SRC_DIR)/main_gmp.cpp \
                  $(SRC_DIR)/bloom_filter.cpp \
                  $(SRC_DIR)/base58.cpp

# GMP + libsecp256k1 version (SOTA - fastest & most accurate)
CPP_SOURCES_GMP_SECP = $(SRC_DIR)/main_gmp_secp.cpp \
                       $(SRC_DIR)/bloom_filter.cpp \
                       $(SRC_DIR)/base58.cpp

CU_SOURCES = $(SRC_DIR)/gpu_kernel.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CPP_OBJECTS_GMP = $(CPP_SOURCES_GMP:.cpp=_gmp.o)
# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CPP_OBJECTS_GMP = $(CPP_SOURCES_GMP:.cpp=_gmp.o)
CPP_OBJECTS_GMP_SECP = $(CPP_SOURCES_GMP_SECP:.cpp=_secp.o)
CU_OBJECTS = $(SRC_DIR)/gpu_secp256k1.o

# Default target builds CPU version (SOTA)
.PHONY: all
all: hybrid_solver_secp

.PHONY: help
help:
	@echo "Bitcoin Puzzle Solver - Build Targets:"
	@echo ""
	@echo "  make hybrid        - Build Hybrid GPU+CPU multi-puzzle solver [RECOMMENDED]"
	@echo "  make secp          - Build SOTA CPU solver (GMP + libsecp256k1)"
	@echo "  make gpu           - Build GPU solver (CUDA + GMP + libsecp256k1)"
	@echo "  make all           - Build SOTA CPU solver (default)"
	@echo "  make clean         - Remove all build artifacts"
	@echo ""
	@echo "Hybrid Solver Features:"
	@echo "  - GPU speed (500M-1B keys/s) + CPU accuracy (100% correct)"
	@echo "  - Multi-puzzle: Solves multiple puzzles simultaneously"
	@echo "  - Independent: Each thread stops only when it finds its key"
	@echo "  - Safe: Append-only key storage with timestamps"
	@echo "  - Maximum discovery: Can find multiple keys in one run"
	@echo ""
	@echo "GPU Requirements:"
	@echo "  - NVIDIA GPU with Compute Capability 7.5+ (Tesla T4, RTX 20xx+)"
	@echo "  - CUDA Toolkit 11.0+"
	@echo "  - nvcc compiler"

# SOTA version: GMP + libsecp256k1 (CPU - FASTEST & MOST ACCURATE)
.PHONY: secp
secp: hybrid_solver_secp

hybrid_solver_secp: $(CPP_OBJECTS_GMP_SECP)
	@echo "[+] Linking hybrid_solver_secp with GMP + libsecp256k1..."
	$(CXX) $(CPP_OBJECTS_GMP_SECP) -o hybrid_solver_secp $(LDFLAGS) $(GMP_LIBS) $(SECP256K1_LIBS)
	@echo "[+] Build complete: ./hybrid_solver_secp"
	@echo "[✨] SOTA CPU VERSION - Uses Bitcoin Core's secp256k1 (fastest & 100% accurate)"
	@echo "[i] Supports puzzles 65-256 bits with multi-threading"

# GPU version: CUDA + GMP + libsecp256k1
.PHONY: gpu
gpu: check_cuda $(TARGET_GPU)

.PHONY: check_cuda
check_cuda:
ifndef CUDA_AVAILABLE
	@echo "[!] ERROR: CUDA not found. GPU build requires:"
	@echo "    - NVIDIA CUDA Toolkit (11.0+)"
	@echo "    - nvcc compiler in PATH"
	@echo "    - NVIDIA GPU with Compute Capability 7.5+"
	@echo ""
	@echo "Install: https://developer.nvidia.com/cuda-downloads"
	@exit 1
else
	@echo "[✓] CUDA detected: $(CUDA_AVAILABLE)"
	@echo "[i] Building for GPU architectures: SM 75, 80, 86, 89, 90"
endif

$(TARGET_GPU): $(CU_OBJECTS) $(CPP_OBJECTS_GMP_SECP)
	@echo "[+] Linking $(TARGET_GPU) with CUDA support..."
	$(NVCC) $(NVCCFLAGS) $(CU_OBJECTS) $(CPP_OBJECTS_GMP_SECP) -o $(TARGET_GPU) \
		-Xcompiler -pthread $(GMP_LIBS) $(SECP256K1_LIBS) $(CUDA_LDFLAGS) $(OPENSSL_LIBS)
	@echo "[+] Build complete: ./$(TARGET_GPU)"
	@echo "[🚀] GPU VERSION - Massive parallelization on CUDA"
	@echo "[i] Target: 100M-1B keys/sec on RTX 4090 / A100"

# Hybrid GPU+CPU multi-puzzle solver (RECOMMENDED)
.PHONY: hybrid
hybrid: check_cuda $(CU_OBJECTS) $(SRC_DIR)/hybrid_gpu_cpu.o
	@echo "[+] Linking Hybrid GPU+CPU multi-puzzle solver..."
	$(NVCC) $(NVCCFLAGS) $(CU_OBJECTS) $(SRC_DIR)/hybrid_gpu_cpu.o -o hybrid_multi_solver \
		-Xcompiler -pthread $(GMP_LIBS) $(SECP256K1_LIBS) $(CUDA_LDFLAGS) $(OPENSSL_LIBS)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ Hybrid GPU+CPU Solver Ready: ./hybrid_multi_solver      ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Features:"
	@echo "  🚀 GPU: Fast candidate generation (500M-1B keys/s)"
	@echo "  ✅ CPU: Validates all GPU hits with libsecp256k1"
	@echo "  🎯 Multi-puzzle: Solves multiple puzzles simultaneously"
	@echo "  💾 Safe: Append-only key storage (never overwrites)"
	@echo "  🔄 Independent: Each thread runs until it finds its key"
	@echo "  🏆 Maximum discovery: Can find multiple keys per run"
	@echo ""
	@echo "Usage: ./hybrid_multi_solver puzzles_71-99.csv"
	@echo ""

$(SRC_DIR)/hybrid_gpu_cpu.o: $(SRC_DIR)/hybrid_gpu_cpu.cpp
	@echo "[+] Compiling hybrid GPU+CPU solver..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CUDA kernel
$(SRC_DIR)/gpu_secp256k1.o: $(SRC_DIR)/gpu_secp256k1.cu
	@echo "[+] Compiling CUDA kernel: $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 64-bit version (original - for puzzles ≤64 bits)
$(TARGET): $(OBJECTS)
	@echo "[+] Linking $(TARGET)..."
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "[+] Build complete: ./$(TARGET)"

# 128-bit+ GMP version (custom EC - for puzzles 65-256 bits)
$(TARGET_GMP): $(CPP_OBJECTS_GMP)
	@echo "[+] Linking $(TARGET_GMP) with GMP support..."
	$(CXX) $(CPP_OBJECTS_GMP) -o $(TARGET_GMP) $(LDFLAGS) $(GMP_LIBS)
	@echo "[+] Build complete: ./$(TARGET_GMP)"
	@echo "[i] This version supports puzzles up to 256 bits"

# Compile C++ sources (original version)
%.o: %.cpp
	@echo "[+] Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C++ sources for GMP version (prevents name collision)
%_gmp.o: %.cpp
	@echo "[+] Compiling $< (GMP version)..."
	$(CXX) $(CXXFLAGS) $(GMP_FLAGS) -c $< -o $@

# Compile C++ sources for GMP + secp256k1 version (SOTA)
%_secp.o: %.cpp
	@echo "[+] Compiling $< (SOTA: GMP + secp256k1)..."
	$(CXX) $(CXXFLAGS) $(GMP_FLAGS) $(SECP256K1_FLAGS) -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	@echo "[+] Cleaning build artifacts..."
	rm -f $(SRC_DIR)/*.o
	rm -f $(TARGET) $(TARGET_GMP) $(TARGET_GPU) hybrid_solver_secp
	rm -f WINNER_*.txt
	@echo "[+] Clean complete"

# CPU-only build (without CUDA even if available)
cpu-only: CXXFLAGS := $(filter-out -DUSE_CUDA,$(CXXFLAGS))
cpu-only: OBJECTS = $(CPP_OBJECTS)
cpu-only: $(TARGET)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "[+] Cleaning..."
	rm -f $(SRC_DIR)/*.o $(TARGET) $(TARGET_GMP) hybrid_solver_secp
	rm -f *.checkpoint *.work *.bin
	@echo "[+] Clean complete"

# Clean only GMP version
.PHONY: clean-gmp
clean-gmp:
	@echo "[+] Cleaning GMP build artifacts..."
	rm -f src/main_gmp.o src/bloom_filter_gmp.o src/base58_gmp.o $(TARGET_GMP)
	@echo "[+] GMP clean complete"

# Clean only SOTA version
.PHONY: clean-secp
clean-secp:
	@echo "[+] Cleaning SOTA build artifacts..."
	rm -f src/main_gmp_secp_secp.o src/bloom_filter_secp.o src/base58_secp.o hybrid_solver_secp
	@echo "[+] SOTA clean complete"

# Build only GMP version (for puzzles 65-256 bits)
.PHONY: gmp
gmp: $(TARGET_GMP)
	@echo "[i] GMP solver ready for puzzles 65-256 bits"

# Build only SOTA version (RECOMMENDED)
.PHONY: secp
secp: hybrid_solver_secp
	@echo "[i] ✨ SOTA solver ready - uses Bitcoin Core's secp256k1"

# Install dependencies (macOS with Homebrew)
install-deps-mac:
	@echo "[+] Installing dependencies (macOS)..."
	brew install openssl gmp

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	@echo "[+] Installing dependencies (Ubuntu/Debian)..."
	sudo apt-get update
	sudo apt-get install -y build-essential libssl-dev libgmp-dev

# Test build
test: $(TARGET)
	@echo "[+] Running test (small range)..."
	./$(TARGET) 1000000 2000000 data/test_addresses.txt 2

# Help
.PHONY: help
help:
	@echo "Hybrid Bitcoin Puzzle Solver - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build both 64-bit and GMP versions (default)"
	@echo "  $(TARGET)        - Build 64-bit version (puzzles ≤64 bits)"
	@echo "  $(TARGET_GMP)    - Build GMP version (puzzles 65-256 bits)"
	@echo "  gmp              - Alias for $(TARGET_GMP)"
	@echo "  cpu-only         - Build 64-bit without CUDA even if available"
	@echo "  clean            - Remove all build artifacts"
	@echo "  clean-gmp        - Remove only GMP build artifacts"
	@echo "  install-deps-mac - Install dependencies on macOS (Homebrew)"
	@echo "  install-deps-ubuntu - Install dependencies on Ubuntu/Debian"
	@echo "  test             - Run test with small range (64-bit version)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make gmp                    # Build GMP solver for puzzles 71+"
	@echo "  make all                    # Build both versions"
	@echo "  make clean && make gmp      # Clean rebuild of GMP version"
	@echo ""
	@echo "Running GMP Solver:"
	@echo "  ./$(TARGET_GMP) <range_start> <range_end> <target_file> <threads> <puzzle_num>"
	@echo "  Example: ./$(TARGET_GMP) 400000000000000000 400001000000000000 data/puzzle71.txt 8 71"
	@echo "  cpu-only         - Build CPU-only version"
	@echo "  clean            - Remove build artifacts"
	@echo "  test             - Build and run quick test"
	@echo "  install-deps-mac - Install dependencies on macOS"
	@echo "  install-deps-ubuntu - Install dependencies on Ubuntu/Debian"
	@echo ""
	@echo "Usage:"
	@echo "  make                  # Build with auto-detection"
	@echo "  make cpu-only         # Force CPU-only build"
	@echo "  make CUDA_PATH=/path  # Custom CUDA path"

.PHONY: all clean cpu-only test help install-deps-mac install-deps-ubuntu

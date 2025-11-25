# Hybrid Bitcoin Puzzle Solver - Makefile
# Supports: CPU-only build or CPU+GPU (CUDA) build

CXX = g++
NVCC = nvcc
TARGET = hybrid_solver

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

# OpenSSL for hash functions
OPENSSL_FLAGS = $(shell pkg-config --cflags openssl 2>/dev/null || echo "-I/usr/local/opt/openssl/include")
OPENSSL_LIBS = $(shell pkg-config --libs openssl 2>/dev/null || echo "-L/usr/local/opt/openssl/lib -lssl -lcrypto")

CXXFLAGS += $(OPENSSL_FLAGS)
LDFLAGS += $(OPENSSL_LIBS)

# GMP for big integer math (optional, can use built-in)
GMP_FLAGS = $(shell pkg-config --cflags gmp 2>/dev/null || echo "")
GMP_LIBS = $(shell pkg-config --libs gmp 2>/dev/null || echo "-lgmp")

CXXFLAGS += $(GMP_FLAGS)
LDFLAGS += $(GMP_LIBS)

# Source files
CPP_SOURCES = $(SRC_DIR)/main.cpp \
              $(SRC_DIR)/secp256k1_tiny.cpp \
              $(SRC_DIR)/bloom_filter.cpp

CU_SOURCES = $(SRC_DIR)/gpu_kernel.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# CUDA flags (if available)
ifdef CUDA_AVAILABLE
    CXXFLAGS += -DUSE_CUDA -I$(CUDA_PATH)/include
    NVCCFLAGS = -O3 -arch=sm_75 -I$(INC_DIR) --compiler-options -fPIC
    LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
    OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)
    $(info [*] CUDA detected - building with GPU support)
else
    OBJECTS = $(CPP_OBJECTS)
    $(info [*] CUDA not found - building CPU-only version)
endif

# Default target
all: $(TARGET)

# Main executable
$(TARGET): $(OBJECTS)
	@echo "[+] Linking $(TARGET)..."
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "[+] Build complete: ./$(TARGET)"

# Compile C++ sources
%.o: %.cpp
	@echo "[+] Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
ifdef CUDA_AVAILABLE
%.o: %.cu
	@echo "[+] Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# CPU-only build (without CUDA even if available)
cpu-only: CXXFLAGS := $(filter-out -DUSE_CUDA,$(CXXFLAGS))
cpu-only: OBJECTS = $(CPP_OBJECTS)
cpu-only: $(TARGET)

# Clean build artifacts
clean:
	@echo "[+] Cleaning..."
	rm -f $(SRC_DIR)/*.o $(TARGET)
	rm -f *.checkpoint *.work *.bin

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
help:
	@echo "Hybrid Bitcoin Puzzle Solver - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build with GPU support if CUDA available"
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

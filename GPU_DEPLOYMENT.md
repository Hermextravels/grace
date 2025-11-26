# GPU Deployment Guide - Bitcoin Puzzle Solver

## 🎯 Overview

This guide explains how to deploy the **hybrid_solver GPU version** to cloud GPU servers for maximum solving speed.

**Target Performance**: 100M-1B keys/sec on RTX 4090 / A100

## 📋 Prerequisites

### GPU Requirements
- **NVIDIA GPU**: Compute Capability 7.5+ (Tesla T4, RTX 20xx, A100, RTX 4090)
- **VRAM**: Minimum 8GB (16GB+ recommended for large ranges)
- **CUDA**: Version 11.0+ installed

### Software Requirements
```bash
# CUDA Toolkit
nvidia-smi  # Verify driver
nvcc --version  # Verify CUDA compiler

# Build dependencies
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libgmp-dev pkg-config git
```

### Install libsecp256k1
```bash
# Clone Bitcoin Core's secp256k1
git clone https://github.com/bitcoin-core/secp256k1.git
cd secp256k1
./autogen.sh
./configure --enable-module-recovery
make
sudo make install
sudo ldconfig
```

## 🚀 Build Instructions

### 1. Clone Repository
```bash
git clone https://github.com/[your-repo]/puzzle71.git
cd puzzle71/hybrid_solver
```

### 2. Build GPU Version
```bash
# Check CUDA is detected
make help

# Build GPU solver
make gpu

# Verify build
./hybrid_solver_gpu --help
```

Expected output:
```
[✓] CUDA detected: /usr/bin/nvcc
[i] Building for GPU architectures: SM 75, 80, 86, 89, 90
[+] Compiling CUDA kernel: src/gpu_secp256k1.cu...
[+] Linking hybrid_solver_gpu with CUDA support...
[+] Build complete: ./hybrid_solver_gpu
[🚀] GPU VERSION - Massive parallelization on CUDA
[i] Target: 100M-1B keys/sec on RTX 4090 / A100
```

## 🧪 Testing

### Test 1: Verify GPU Initialization
```bash
# Test with puzzle 66 (known solution)
./hybrid_solver_gpu 2832ed74f2b5e35e0 2832ed74f2b5e3600 data/puzzle66.txt 1 66

# Expected: Should find key in seconds
# Address: 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so
# Private key: 0x2832ed74f2b5e35ee
```

### Test 2: Performance Benchmark
```bash
# Narrow range (1 million keys)
time ./hybrid_solver_gpu 400000000000000000 40000000000100000 data/puzzle71.txt 1 71

# Calculate keys/sec from output
```

### Test 3: Multi-GPU (if available)
```bash
# GPU 0: Puzzle 71
CUDA_VISIBLE_DEVICES=0 ./hybrid_solver_gpu [puzzle71_range] data/puzzle71.txt 1 71 &

# GPU 1: Puzzle 72
CUDA_VISIBLE_DEVICES=1 ./hybrid_solver_gpu [puzzle72_range] data/puzzle72.txt 1 72 &
```

## ☁️ Cloud GPU Providers

### Option 1: Lambda Labs (Recommended)
**Pros**: High-performance GPUs, simple setup, competitive pricing  
**GPUs**: A100 (80GB), RTX 6000 Ada, H100  
**Cost**: $1.10-$2.49/hr  

```bash
# 1. Sign up: https://lambdalabs.com/service/gpu-cloud
# 2. Launch instance (Ubuntu 22.04 + CUDA pre-installed)
# 3. SSH into instance
# 4. Clone repo and build
```

### Option 2: Vast.ai (Budget)
**Pros**: Cheapest GPU rentals, many options  
**GPUs**: RTX 3090, 4090, A100, etc.  
**Cost**: $0.20-$1.50/hr  

```bash
# 1. Search: https://vast.ai/console/create/
# 2. Filter: CUDA 11.8+, 16GB+ VRAM, score >0.95
# 3. Rent instance
# 4. Use provided SSH command
```

### Option 3: AWS EC2 (Enterprise)
**Pros**: Reliable, scalable, on-demand + spot instances  
**GPUs**: p3 (V100), p4 (A100), g5 (A10G)  
**Cost**: $3-$32/hr (on-demand), 70% less (spot)  

```bash
# Launch p3.2xlarge (Tesla V100 16GB)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-groups gpu-access

# Or use p4d.24xlarge for 8x A100 (production)
```

## 📊 Expected Performance

| GPU Model | VRAM | CUDA Cores | Est. Speed | Puzzle 71 Time |
|-----------|------|------------|------------|----------------|
| Tesla T4 | 16GB | 2560 | 50-100 MKey/s | Hours |
| RTX 3090 | 24GB | 10496 | 200-500 MKey/s | Minutes |
| RTX 4090 | 24GB | 16384 | 500M-1B Key/s | Seconds-Minutes |
| A100 (40GB) | 40GB | 6912 | 300-600 MKey/s | Minutes |
| A100 (80GB) | 80GB | 6912 | 300-600 MKey/s | Minutes |
| H100 | 80GB | 16896 | 1-2 BKey/s | Seconds |

*Note: Actual speed depends on optimization and CUDA kernel efficiency*

## 🔧 Optimization Tips

### 1. Tune Block/Thread Count
Edit `gpu_secp256k1.cu`:
```cpp
int num_blocks = 512;      // Increase for more parallelism
int threads_per_block = 256; // Optimal: 256-512 for most GPUs
```

### 2. Enable Persistent Threads
```cpp
// Keep threads alive across kernel launches
// Reduces overhead for repeated searches
```

### 3. Use Pinned Memory
```cpp
// Host-side allocations
uint8_t* h_targets;
cudaMallocHost(&h_targets, size);  // Faster than malloc
```

### 4. Async Kernel Launches
```cpp
// Overlap computation with data transfer
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<blocks, threads, 0, stream>>>(...);
```

## 📝 Production Deployment Checklist

### Pre-Launch
- [ ] Test with known puzzle (66) to verify correctness
- [ ] Benchmark performance (keys/sec) on target GPU
- [ ] Setup monitoring (GPU utilization, temperature)
- [ ] Configure automatic restart on completion

### During Run
- [ ] Monitor GPU temperature (< 80°C ideal)
- [ ] Watch for CUDA errors in logs
- [ ] Track progress (keys checked vs total)
- [ ] Check for found keys periodically

### Post-Run
- [ ] Verify found private keys
- [ ] Test WIF import in Bitcoin Core
- [ ] Securely store keys
- [ ] Clean up GPU instance

## 🚨 Troubleshooting

### "CUDA not found" Error
```bash
# Check CUDA installation
which nvcc
ls /usr/local/cuda

# Add to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "Out of Memory" Error
```cpp
// Reduce batch size in kernel launch
int num_blocks = 128;  // Reduce from 256
```

### Slow Performance
```bash
# Check GPU is actually being used
nvidia-smi dmon -s u -c 1

# Profile with nvprof
nvprof ./hybrid_solver_gpu [args]
```

### Wrong Addresses Generated
```bash
# Test with CPU version first (verified correct)
./hybrid_solver_secp [same args]

# Compare outputs - GPU should match CPU exactly
```

## 📧 Support

If you encounter issues:
1. Check CUDA version: `nvcc --version` >= 11.0
2. Verify GPU compatibility: Compute Capability >= 7.5
3. Test CPU version first to isolate CUDA issues
4. Review kernel output for error messages

## 🎁 Success Criteria

✅ Build completes without errors  
✅ Test with puzzle 66 finds correct key  
✅ Performance: >10M keys/sec minimum  
✅ GPU utilization: >90%  
✅ No CUDA errors in logs  

Good luck solving! 🚀

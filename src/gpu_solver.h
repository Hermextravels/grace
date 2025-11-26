// GPU solver wrapper - integrates CUDA kernel with main solver
#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize GPU and CUDA constants
void gpu_init_constants();

// Search for private keys on GPU
// Returns 1 if found, 0 if not found
int gpu_search_keys(
    const uint8_t* start_key_bytes,    // 32 bytes: starting private key
    uint64_t total_keys,                // Total number of keys to search
    const uint8_t* target_hash160s,    // Array of target hash160s (20 bytes each)
    uint32_t num_targets,               // Number of target addresses
    uint8_t* found_key_out,            // Output: 32 bytes private key if found
    uint8_t* found_hash160_out         // Output: 20 bytes hash160 if found
);

#ifdef __cplusplus
}
#endif

#endif // GPU_SOLVER_H

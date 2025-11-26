#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// BSGS main entry point
// Returns 1 if key found, 0 otherwise
int gpu_bsgs_solve(const char* address, const char* start_hex, const char* end_hex, int bits, unsigned char* found_key);

#ifdef __cplusplus
}
#endif

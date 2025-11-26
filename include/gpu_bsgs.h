
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Main BSGS solver entry point
// Returns 0 if key found, -1 otherwise
int gpu_bsgs_solve(
	const char* address,
	const char* start_hex,
	const char* end_hex,
	int bits,
	uint64_t* found_key
);

#ifdef __cplusplus
}
#endif

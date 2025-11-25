#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H

#include <stdint.h>
#include <stdbool.h>

#define BLOOM_BITS_PER_ELEMENT 20  // Optimal for ~1% false positive rate

typedef struct {
    uint8_t* bits;
    uint64_t size;        // Size in bits
    uint32_t num_hashes;  // Number of hash functions
    uint64_t count;       // Number of elements added
} bloom_filter;

// Initialize bloom filter for expected number of elements
bloom_filter* bloom_create(uint64_t expected_elements);

// Add hash160 to bloom filter
void bloom_add(bloom_filter* bf, const uint8_t* hash160);

// Check if hash160 is possibly in the filter
bool bloom_check(const bloom_filter* bf, const uint8_t* hash160);

// Free bloom filter
void bloom_free(bloom_filter* bf);

// Get memory usage in MB
double bloom_memory_mb(const bloom_filter* bf);

#endif

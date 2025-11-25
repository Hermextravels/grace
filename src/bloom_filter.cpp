/*
 * Simple Bloom Filter Implementation
 * Optimized for Bitcoin address hash160 matching
 */

#include "../include/bloom_filter.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// MurmurHash3 32-bit (simplified)
static uint32_t murmur3_32(const uint8_t* key, size_t len, uint32_t seed) {
    uint32_t h = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    
    // Body
    const uint32_t* blocks = (const uint32_t*)(data);
    for (int i = 0; i < nblocks; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        
        h ^= k1;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }
    
    // Tail
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = (k1 << 15) | (k1 >> 17); k1 *= c2; h ^= k1;
    }
    
    // Finalization
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    
    return h;
}

bloom_filter* bloom_create(uint64_t expected_elements) {
    bloom_filter* bf = (bloom_filter*)malloc(sizeof(bloom_filter));
    if (!bf) return NULL;
    
    // Calculate optimal size: m = -n*ln(p) / (ln(2)^2)
    // For p=0.01 (1% false positive): m ≈ 9.6 * n ≈ 10*n bits
    bf->size = expected_elements * BLOOM_BITS_PER_ELEMENT;
    
    // Calculate optimal number of hash functions: k = (m/n) * ln(2)
    // For our parameters: k ≈ 13.8 ≈ 14
    bf->num_hashes = (uint32_t)((double)BLOOM_BITS_PER_ELEMENT * 0.693);
    if (bf->num_hashes < 1) bf->num_hashes = 1;
    if (bf->num_hashes > 20) bf->num_hashes = 20;  // Cap at 20
    
    // Allocate bit array
    uint64_t bytes = (bf->size + 7) / 8;
    bf->bits = (uint8_t*)calloc(bytes, 1);
    if (!bf->bits) {
        free(bf);
        return NULL;
    }
    
    bf->count = 0;
    
    printf("[Bloom] Created filter: %llu bits (%.2f MB), %u hash functions\n",
           bf->size, (double)bytes / (1024*1024), bf->num_hashes);
    
    return bf;
}

void bloom_add(bloom_filter* bf, const uint8_t* hash160) {
    for (uint32_t i = 0; i < bf->num_hashes; i++) {
        uint64_t hash = murmur3_32(hash160, 20, i);
        uint64_t bit_pos = hash % bf->size;
        bf->bits[bit_pos / 8] |= (1 << (bit_pos % 8));
    }
    bf->count++;
}

bool bloom_check(const bloom_filter* bf, const uint8_t* hash160) {
    for (uint32_t i = 0; i < bf->num_hashes; i++) {
        uint64_t hash = murmur3_32(hash160, 20, i);
        uint64_t bit_pos = hash % bf->size;
        if (!(bf->bits[bit_pos / 8] & (1 << (bit_pos % 8)))) {
            return false;  // Definitely not in set
        }
    }
    return true;  // Possibly in set (may be false positive)
}

void bloom_free(bloom_filter* bf) {
    if (bf) {
        free(bf->bits);
        free(bf);
    }
}

double bloom_memory_mb(const bloom_filter* bf) {
    if (!bf) return 0.0;
    uint64_t bytes = (bf->size + 7) / 8;
    return (double)bytes / (1024.0 * 1024.0);
}

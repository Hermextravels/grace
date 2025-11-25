#ifndef SECP256K1_TINY_H
#define SECP256K1_TINY_H

#include <stdint.h>
#include <string.h>

// Minimal secp256k1 implementation for address generation
// Optimized for speed on puzzle solving

typedef struct {
    uint64_t d[4];  // 256-bit integer (little-endian)
} u256;

typedef struct {
    u256 x;
    u256 y;
} point;

// secp256k1 curve parameters
extern const u256 SECP256K1_P;
extern const u256 SECP256K1_N;
extern const point SECP256K1_G;

// Endomorphism constants (lambda and beta for faster point multiplication)
extern const u256 LAMBDA;
extern const u256 BETA;

// Core operations
void u256_set_u64(u256* out, uint64_t val);
void u256_add_mod(u256* result, const u256* a, const u256* b, const u256* mod);
void u256_sub_mod(u256* result, const u256* a, const u256* b, const u256* mod);
void u256_mul_mod(u256* result, const u256* a, const u256* b, const u256* mod);
void u256_inv_mod(u256* result, const u256* a, const u256* mod);

// Point operations
void point_add(point* result, const point* p1, const point* p2);
void point_double(point* result, const point* p);
void point_mul(point* result, const point* p, const u256* scalar);
void point_mul_g(point* result, const u256* scalar);

// Fast endomorphism-based point multiplication
void point_mul_endo(point* result, const u256* scalar);

// Address generation from point
void point_to_address_compressed(const point* p, char* address);
void point_to_hash160_compressed(const point* p, uint8_t* hash160);

#endif

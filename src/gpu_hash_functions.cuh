// Device hash functions for CUDA kernels
// This file must be included in every .cu file that uses sha256 or ripemd160
#ifndef GPU_HASH_FUNCTIONS_CUH
#define GPU_HASH_FUNCTIONS_CUH

#include <stdint.h>

// SHA256 constants and device function
__device__ __constant__ uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define EP1(x) (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define SIG0(x) (ROTR32(x, 7) ^ ROTR32(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR32(x, 17) ^ ROTR32(x, 19) ^ ((x) >> 10))

__device__ void sha256(uint8_t* hash, const uint8_t* data, uint32_t len) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint32_t w[64];
    uint8_t block[64];
    uint32_t bitlen = len * 8;
    uint32_t i, j;
    uint32_t blocks = (len + 9 + 63) / 64;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        uint32_t block_start = blk * 64;
        for (i = 0; i < 64; i++) {
            if (block_start + i < len) {
                block[i] = data[block_start + i];
            } else if (block_start + i == len) {
                block[i] = 0x80;
            } else {
                block[i] = 0x00;
            }
        }
        if (blk == blocks - 1) {
            block[63] = (uint8_t)(bitlen);
            block[62] = (uint8_t)(bitlen >> 8);
            block[61] = (uint8_t)(bitlen >> 16);
            block[60] = (uint8_t)(bitlen >> 24);
        }
        for (i = 0; i < 16; i++) {
            w[i] = ((uint32_t)block[i * 4] << 24) |
                   ((uint32_t)block[i * 4 + 1] << 16) |
                   ((uint32_t)block[i * 4 + 2] << 8) |
                   ((uint32_t)block[i * 4 + 3]);
        }
        for (i = 16; i < 64; i++) {
            w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (i = 0; i < 64; i++) {
            uint32_t t1 = h + EP1(e) + CH(e, f, g) + K256[i] + w[i];
            uint32_t t2 = EP0(a) + MAJ(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
    for (i = 0; i < 8; i++) {
        hash[i * 4] = (state[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
        hash[i * 4 + 3] = state[i] & 0xff;
    }
}

#define ROTL32_RMD(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
__device__ __constant__ uint32_t RMD_K_LEFT[5] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
};
__device__ __constant__ uint32_t RMD_K_RIGHT[5] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
};
__device__ __forceinline__ uint32_t rmd_f(uint32_t x, uint32_t y, uint32_t z, int round) {
    if (round < 16) return x ^ y ^ z;
    if (round < 32) return (x & y) | (~x & z);
    if (round < 48) return (x | ~y) ^ z;
    if (round < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}
__device__ void ripemd160(uint8_t* hash, const uint8_t* data, uint32_t len) {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint8_t block[64];
    uint32_t w[16];
    uint64_t bitlen = (uint64_t)len * 8;
    const int r_left[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
    };
    const int r_right[80] = {
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
    };
    const int s_left[80] = {
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
    };
    const int s_right[80] = {
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
    };
    for (uint32_t blk = 0; blk < (len + 8 + 63) / 64; blk++) {
        uint32_t block_start = blk * 64;
        for (int i = 0; i < 64; i++) {
            if (block_start + i < len) {
                block[i] = data[block_start + i];
            } else if (block_start + i == len) {
                block[i] = 0x80;
            } else {
                block[i] = 0x00;
            }
        }
        if (blk == ((len + 8 + 63) / 64) - 1) {
            block[56] = (uint8_t)(bitlen);
            block[57] = (uint8_t)(bitlen >> 8);
            block[58] = (uint8_t)(bitlen >> 16);
            block[59] = (uint8_t)(bitlen >> 24);
            block[60] = (uint8_t)(bitlen >> 32);
            block[61] = (uint8_t)(bitlen >> 40);
            block[62] = (uint8_t)(bitlen >> 48);
            block[63] = (uint8_t)(bitlen >> 56);
        }
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)block[i * 4]) |
                   ((uint32_t)block[i * 4 + 1] << 8) |
                   ((uint32_t)block[i * 4 + 2] << 16) |
                   ((uint32_t)block[i * 4 + 3] << 24);
        }
        uint32_t al = h[0], bl = h[1], cl = h[2], dl = h[3], el = h[4];
        uint32_t ar = h[0], br = h[1], cr = h[2], dr = h[3], er = h[4];
        for (int j = 0; j < 80; j++) {
            uint32_t tl = ROTL32_RMD(al + rmd_f(bl, cl, dl, j) + w[r_left[j]] + RMD_K_LEFT[j / 16], s_left[j]) + el;
            al = el; el = dl; dl = ROTL32_RMD(cl, 10); cl = bl; bl = tl;
            uint32_t tr = ROTL32_RMD(ar + rmd_f(br, cr, dr, 79 - j) + w[r_right[j]] + RMD_K_RIGHT[j / 16], s_right[j]) + er;
            ar = er; er = dr; dr = ROTL32_RMD(cr, 10); cr = br; br = tr;
        }
        uint32_t tmp = h[1] + cl + dr;
        h[1] = h[2] + dl + er;
        h[2] = h[3] + el + ar;
        h[3] = h[4] + al + br;
        h[4] = h[0] + bl + cr;
        h[0] = tmp;
    }
    for (int i = 0; i < 5; i++) {
        hash[i * 4] = (h[i] >> 0) & 0xff;
        hash[i * 4 + 1] = (h[i] >> 8) & 0xff;
        hash[i * 4 + 2] = (h[i] >> 16) & 0xff;
        hash[i * 4 + 3] = (h[i] >> 24) & 0xff;
    }
}

#endif // GPU_HASH_FUNCTIONS_CUH

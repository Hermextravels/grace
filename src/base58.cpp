// Bitcoin Base58 encoding
// Adapted from Bitcoin Core base58.cpp
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

bool base58_encode(const unsigned char* pbegin, const unsigned char* pend, char* output) {
    // Skip & count leading zeroes
    int zeroes = 0;
    int length = 0;
    while (pbegin != pend && *pbegin == 0) {
        pbegin++;
        zeroes++;
    }
    
    // Allocate enough space in big-endian base58 representation
    int size = (pend - pbegin) * 138 / 100 + 1; // log(256) / log(58), rounded up
    unsigned char* b58 = (unsigned char*)calloc(size, sizeof(unsigned char));
    
    // Process the bytes
    while (pbegin != pend) {
        int carry = *pbegin;
        int i = 0;
        // Apply "b58 = b58 * 256 + ch"
        for (int j = size - 1; j >= 0; j--, i++) {
            if (carry == 0 && i >= length)
                break;
            carry += 256 * b58[j];
            b58[j] = carry % 58;
            carry /= 58;
        }
        
        length = i;
        pbegin++;
    }
    
    // Skip leading zeroes in base58 result
    int start = size - length;
    while (start < size && b58[start] == 0)
        start++;
    
    // Translate the result into a string
    int out_idx = 0;
    for (int i = 0; i < zeroes; i++)
        output[out_idx++] = '1';
    for (int i = start; i < size; i++)
        output[out_idx++] = pszBase58[b58[i]];
    output[out_idx] = '\0';
    
    free(b58);
    return true;
}

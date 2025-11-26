// Address utility functions for Bitcoin
#include "../include/address_utils.h"
#include <string.h>
#include <stdint.h>

// Decode Base58 Bitcoin address to extract hash160
bool address_to_hash160(const char* address, uint8_t* hash160_out) {
    const char* b58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    uint8_t bin[32] = {0};
    int bin_len = 0;
    // Base58 decode
    for (const char* p = address; *p; p++) {
        const char* digit = strchr(b58, *p);
        if (!digit) return false;
        int carry = (int)(digit - b58);
        for (int i = bin_len - 1; i >= 0; i--) {
            carry += 58 * bin[i];
            bin[i] = carry % 256;
            carry /= 256;
        }
        while (carry > 0) {
            memmove(bin + 1, bin, bin_len);
            bin[0] = carry % 256;
            bin_len++;
            carry /= 256;
        }
    }
    // Count leading '1's (represent leading zeros)
    int leading_ones = 0;
    for (const char* p = address; *p == '1'; p++) leading_ones++;
    // Bitcoin address = version(1) + hash160(20) + checksum(4) = 25 bytes
    int total_len = bin_len + leading_ones;
    if (total_len != 25) return false;
    // Extract hash160 (skip version byte at position 0+leading_ones, take next 20 bytes)
    if (bin_len >= 21) {
        memcpy(hash160_out, bin + 1, 20);
    } else {
        // Handle case where leading zeros are implied
        memset(hash160_out, 0, 20);
        if (bin_len > 1) {
            memcpy(hash160_out + (20 - (bin_len - 1)), bin + 1, bin_len - 1);
        }
    }
    return true;
}

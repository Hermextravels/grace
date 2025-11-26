// Quick test for WIF encoding
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>
#include "base58.h"

void private_key_to_wif(const uint8_t* key_bytes, char* wif_out) {
    uint8_t extended[38];
    uint8_t hash1[SHA256_DIGEST_LENGTH];
    uint8_t hash2[SHA256_DIGEST_LENGTH];
    
    extended[0] = 0x80;
    memcpy(extended + 1, key_bytes, 32);
    extended[33] = 0x01;
    
    SHA256(extended, 34, hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);
    
    for (int i = 0; i < 4; i++) {
        extended[34 + i] = hash2[i];
    }
    
    // Use pointer-based base58_encode from Bitcoin Core
    base58_encode(extended, extended + 37, wif_out);
}

int main() {
    // Test with a known private key
    // Private key: 0x0000000000000000000000000000000000000000000000000000000000000001
    // Expected WIF (compressed): KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn
    
    uint8_t test_key[32] = {0};
    test_key[31] = 0x01;
    
    char wif[64];
    private_key_to_wif(test_key, wif);
    
    // Debug: print the extended bytes
    uint8_t extended[38];
    uint8_t hash1[SHA256_DIGEST_LENGTH];
    uint8_t hash2[SHA256_DIGEST_LENGTH];
    
    extended[0] = 0x80;
    memcpy(extended + 1, test_key, 32);
    extended[33] = 0x01;
    
    SHA256(extended, 34, hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);
    
    for (int i = 0; i < 4; i++) {
        extended[34 + i] = hash2[i];
    }
    
    printf("Extended bytes (%d): ", 37);
    for (int i = 0; i < 38; i++) {
        printf("%02x", extended[i]);
    }
    printf("\n");
    
    printf("Test Private Key: 0000000000000000000000000000000000000000000000000000000000000001\n");
    printf("Generated WIF: %s\n", wif);
    printf("Expected WIF:  KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn\n");
    
    if (strcmp(wif, "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn") == 0) {
        printf("\n✅ WIF encoding is CORRECT!\n");
        return 0;
    } else {
        printf("\n❌ WIF encoding mismatch\n");
        return 1;
    }
}

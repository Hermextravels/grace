#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>
#include "base58.h"

int main() {
    uint8_t test_key[32] = {0};
    test_key[31] = 0x01;
    
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
    
    printf("All 38 bytes of extended array:\n");
    for (int i = 0; i < 38; i++) {
        printf("%02x ", extended[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
    
    printf("\nFirst 37 bytes (what we encode):\n");
    for (int i = 0; i < 37; i++) {
        printf("%02x", extended[i]);
    }
    printf("\n");
    
    char wif[64];
    base58_encode(extended, extended + 37, wif);
    
    printf("\nGenerated WIF: %s (len=%lu)\n", wif, strlen(wif));
    printf("Expected  WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn (len=52)\n");
    
    return 0;
}

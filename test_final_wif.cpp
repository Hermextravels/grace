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
    
    base58_encode(extended, extended + 38, wif_out);
}

int main() {
    uint8_t test_key[32] = {0};
    test_key[31] = 0x01;
    
    char wif[64];
    private_key_to_wif(test_key, wif);
    
    printf("Test key: 0000000000000000000000000000000000000000000000000000000000000001\n");
    printf("WIF:      %s\n", wif);
    printf("Expected: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn\n");
    
    if (strcmp(wif, "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn") == 0) {
        printf("\n✅ SUCCESS! WIF encoding is correct.\n");
        return 0;
    } else {
        printf("\n❌ FAILED\n");
        return 1;
    }
}

// Address utility functions for Bitcoin
#ifndef ADDRESS_UTILS_H
#define ADDRESS_UTILS_H

#include <stdint.h>
#include <stdbool.h>

// Decode Base58 Bitcoin address to extract hash160
bool address_to_hash160(const char* address, uint8_t* hash160_out);

#endif // ADDRESS_UTILS_H
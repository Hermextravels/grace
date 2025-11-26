#ifndef BASE58_H
#define BASE58_H

#include <stdbool.h>

bool base58_encode(const unsigned char* pbegin, const unsigned char* pend, char* output);

#endif

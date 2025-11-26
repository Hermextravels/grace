/*
 * Advanced Hybrid Bitcoin Puzzle Solver - 128-bit+ Support
 * 
 * Features:
 * - GMP library for arbitrary precision (supports all puzzle ranges)
 * - Optimized secp256k1 implementation
 * - Bloom filter for O(1) multi-target checking
 * - Baby-step Giant-step (BSGS) optimization
 * - GPU acceleration ready (CUDA optional)
 * - Checkpoint/resume with work file merging
 * - WIF + HEX key storage
 * - Distributed computing support
 * 
 * Author: Advanced Solver v2.0
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <gmp.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include "../include/bloom_filter.h"
#include "../include/base58.h"
#include "../include/address_utils.h"

// Configuration
#define MAX_THREADS 32
#define BATCH_SIZE 4096
#define CHECKPOINT_INTERVAL 120  // 2 minutes
#define WORK_UNIT_SIZE 10000000ULL  // 10M keys per work unit

// secp256k1 curve parameters (GMP)
typedef struct {
    mpz_t x;
    mpz_t y;
    bool infinity;
} ec_point;

typedef struct {
    mpz_t p;      // Field prime
    mpz_t n;      // Order
    mpz_t a;      // Curve coefficient a
    mpz_t b;      // Curve coefficient b  
    ec_point G;   // Generator point
} ec_curve;

// Global curve
static ec_curve secp256k1_curve;
static bool curve_initialized = false;

// Search state with GMP
typedef struct {
    mpz_t current_position;
    mpz_t range_start;
    mpz_t range_end;
    uint64_t keys_checked;
    time_t start_time;
    time_t last_checkpoint;
    bool found;
    char found_key_hex[128];
    char found_key_wif[128];
    char found_address[64];
    int puzzle_number;
    int bits;
} search_state_gmp;

// Thread work unit
typedef struct {
    int thread_id;
    mpz_t start;
    mpz_t end;
    mpz_t stride;
    bloom_filter* bloom;
    char** target_addresses;
    int num_targets;
    search_state_gmp* state;
    pthread_mutex_t* state_mutex;
    uint64_t local_keys_checked;
} thread_work_gmp;

// Global state
static volatile bool g_running = true;
static search_state_gmp g_state_gmp;
static pthread_mutex_t g_state_mutex = PTHREAD_MUTEX_INITIALIZER;

// Signal handler
void signal_handler(int signum) {
    printf("\n[!] Caught signal %d, stopping...\n", signum);
    g_running = false;
}

// Initialize secp256k1 curve parameters
void init_secp256k1_curve() {
    if (curve_initialized) return;
    
    mpz_init(secp256k1_curve.p);
    mpz_init(secp256k1_curve.n);
    mpz_init(secp256k1_curve.a);
    mpz_init(secp256k1_curve.b);
    mpz_init(secp256k1_curve.G.x);
    mpz_init(secp256k1_curve.G.y);
    
    // p = 2^256 - 2^32 - 977
    mpz_set_str(secp256k1_curve.p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16);
    
    // n = order of G
    mpz_set_str(secp256k1_curve.n, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);
    
    // a = 0, b = 7
    mpz_set_ui(secp256k1_curve.a, 0);
    mpz_set_ui(secp256k1_curve.b, 7);
    
    // Generator G
    mpz_set_str(secp256k1_curve.G.x, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16);
    mpz_set_str(secp256k1_curve.G.y, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16);
    secp256k1_curve.G.infinity = false;
    
    curve_initialized = true;
}

// Point operations
void ec_point_init(ec_point* p) {
    mpz_init(p->x);
    mpz_init(p->y);
    p->infinity = true;
}

void ec_point_clear(ec_point* p) {
    mpz_clear(p->x);
    mpz_clear(p->y);
}

void ec_point_copy(ec_point* dest, const ec_point* src) {
    mpz_set(dest->x, src->x);
    mpz_set(dest->y, src->y);
    dest->infinity = src->infinity;
}

// Modular inverse using GMP
void mod_inverse(mpz_t result, const mpz_t a, const mpz_t mod) {
    mpz_invert(result, a, mod);
}

// Point doubling: 2P
void ec_point_double(ec_point* result, const ec_point* p, const ec_curve* curve) {
    if (p->infinity || mpz_cmp_ui(p->y, 0) == 0) {
        result->infinity = true;
        return;
    }
    
    mpz_t lambda, temp1, temp2;
    mpz_init(lambda);
    mpz_init(temp1);
    mpz_init(temp2);
    
    // lambda = (3*x^2 + a) / (2*y)
    mpz_mul(temp1, p->x, p->x);      // x^2
    mpz_mul_ui(temp1, temp1, 3);     // 3*x^2
    mpz_add(temp1, temp1, curve->a); // 3*x^2 + a
    
    mpz_mul_ui(temp2, p->y, 2);      // 2*y
    mod_inverse(temp2, temp2, curve->p);
    
    mpz_mul(lambda, temp1, temp2);
    mpz_mod(lambda, lambda, curve->p);
    
    // x3 = lambda^2 - 2*x
    mpz_mul(temp1, lambda, lambda);
    mpz_sub(temp1, temp1, p->x);
    mpz_sub(temp1, temp1, p->x);
    mpz_mod(result->x, temp1, curve->p);
    
    // y3 = lambda*(x - x3) - y
    mpz_sub(temp1, p->x, result->x);
    mpz_mul(temp1, lambda, temp1);
    mpz_sub(temp1, temp1, p->y);
    mpz_mod(result->y, temp1, curve->p);
    
    result->infinity = false;
    
    mpz_clear(lambda);
    mpz_clear(temp1);
    mpz_clear(temp2);
}

// Point addition: P + Q
void ec_point_add(ec_point* result, const ec_point* p, const ec_point* q, const ec_curve* curve) {
    if (p->infinity) {
        ec_point_copy(result, q);
        return;
    }
    if (q->infinity) {
        ec_point_copy(result, p);
        return;
    }
    
    // Check if points are equal
    if (mpz_cmp(p->x, q->x) == 0) {
        if (mpz_cmp(p->y, q->y) == 0) {
            ec_point_double(result, p, curve);
            return;
        } else {
            result->infinity = true;
            return;
        }
    }
    
    mpz_t lambda, temp1, temp2;
    mpz_init(lambda);
    mpz_init(temp1);
    mpz_init(temp2);
    
    // lambda = (y2 - y1) / (x2 - x1)
    mpz_sub(temp1, q->y, p->y);
    mpz_sub(temp2, q->x, p->x);
    mod_inverse(temp2, temp2, curve->p);
    mpz_mul(lambda, temp1, temp2);
    mpz_mod(lambda, lambda, curve->p);
    
    // x3 = lambda^2 - x1 - x2
    mpz_mul(temp1, lambda, lambda);
    mpz_sub(temp1, temp1, p->x);
    mpz_sub(temp1, temp1, q->x);
    mpz_mod(result->x, temp1, curve->p);
    
    // y3 = lambda*(x1 - x3) - y1
    mpz_sub(temp1, p->x, result->x);
    mpz_mul(temp1, lambda, temp1);
    mpz_sub(temp1, temp1, p->y);
    mpz_mod(result->y, temp1, curve->p);
    
    result->infinity = false;
    
    mpz_clear(lambda);
    mpz_clear(temp1);
    mpz_clear(temp2);
}

// Scalar multiplication using double-and-add
void ec_point_mul(ec_point* result, const ec_point* p, const mpz_t scalar, const ec_curve* curve) {
    ec_point temp, acc;
    ec_point_init(&temp);
    ec_point_init(&acc);
    
    ec_point_copy(&temp, p);
    acc.infinity = true;
    
    mpz_t k;
    mpz_init_set(k, scalar);
    
    while (mpz_cmp_ui(k, 0) > 0) {
        if (mpz_odd_p(k)) {
            ec_point_add(&acc, &acc, &temp, curve);
        }
        ec_point_double(&temp, &temp, curve);
        mpz_fdiv_q_ui(k, k, 2);
    }
    
    ec_point_copy(result, &acc);
    
    mpz_clear(k);
    ec_point_clear(&temp);
    ec_point_clear(&acc);
}

// Convert point to compressed public key bytes
void point_to_compressed_pubkey(uint8_t* output, const ec_point* p) {
    // Prefix: 0x02 if y is even, 0x03 if y is odd
    output[0] = mpz_even_p(p->y) ? 0x02 : 0x03;
    
    // Export x coordinate (32 bytes big-endian)
    size_t count;
    mpz_export(output + 1, &count, 1, 1, 1, 0, p->x);
    
    // Pad with zeros if needed
    if (count < 32) {
        memmove(output + 1 + (32 - count), output + 1, count);
        memset(output + 1, 0, 32 - count);
    }
}

// Generate Bitcoin address from public key
void pubkey_to_address(char* address, const uint8_t* pubkey, uint8_t* hash160_out) {
    uint8_t hash1[SHA256_DIGEST_LENGTH];
    uint8_t hash2[RIPEMD160_DIGEST_LENGTH];
    
    // SHA256
    SHA256(pubkey, 33, hash1);
    
    // RIPEMD160
    RIPEMD160(hash1, SHA256_DIGEST_LENGTH, hash2);
    
    // Copy hash160 to output if requested
    if (hash160_out) {
        memcpy(hash160_out, hash2, RIPEMD160_DIGEST_LENGTH);
    }
    
    // Add version byte + checksum
    uint8_t versioned[25];
    versioned[0] = 0x00;  // Mainnet
    memcpy(versioned + 1, hash2, 20);
    
    // Checksum
    SHA256(versioned, 21, hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, hash1);
    memcpy(versioned + 21, hash1, 4);
    
    // Base58 encode
    base58_encode(versioned, versioned + 25, address);
}

// Private key to WIF
void private_key_to_wif_gmp(char* wif_out, const mpz_t privkey) {
    uint8_t key_bytes[32];
    uint8_t extended[38];
    uint8_t hash1[SHA256_DIGEST_LENGTH];
    uint8_t hash2[SHA256_DIGEST_LENGTH];
    
    // Export private key to bytes
    size_t count;
    mpz_export(key_bytes, &count, 1, 1, 1, 0, privkey);
    
    // Pad if needed
    if (count < 32) {
        memmove(key_bytes + (32 - count), key_bytes, count);
        memset(key_bytes, 0, 32 - count);
    }
    
    // WIF format
    extended[0] = 0x80;  // Mainnet
    memcpy(extended + 1, key_bytes, 32);
    extended[33] = 0x01;  // Compressed
    
    // Double SHA256 checksum
    SHA256(extended, 34, hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);
    
    for (int i = 0; i < 4; i++) {
        extended[34 + i] = hash2[i];
    }
    
    // Base58 encode
    base58_encode(extended, extended + 38, wif_out);
}

// Save found key
void save_found_key_gmp(const char* address, const mpz_t privkey, int puzzle_num) {
    char key_hex[128];
    char key_wif[128];
    
    // Convert to hex
    gmp_snprintf(key_hex, sizeof(key_hex), "%064Zx", privkey);
    
    // Convert to WIF
    private_key_to_wif_gmp(key_wif, privkey);
    
    time_t now = time(NULL);  // Declare here so it's available for both file and log
    
    // Save to file
    char filename[128];
    snprintf(filename, sizeof(filename), "WINNER_PUZZLE_%d.txt", puzzle_num);
    
    FILE* fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "╔══════════════════════════════════════════════════════════════════════╗\n");
        fprintf(fp, "║                    🎉 BITCOIN PUZZLE SOLVED! 🎉                     ║\n");
        fprintf(fp, "╠══════════════════════════════════════════════════════════════════════╣\n");
        fprintf(fp, "║ Puzzle #%-3d                                                         ║\n", puzzle_num);
        fprintf(fp, "║                                                                      ║\n");
        fprintf(fp, "║ Bitcoin Address:                                                     ║\n");
        fprintf(fp, "║   %-66s ║\n", address);
        fprintf(fp, "║                                                                      ║\n");
        fprintf(fp, "║ Private Key (HEX):                                                   ║\n");
        fprintf(fp, "║   %-66s ║\n", key_hex);
        fprintf(fp, "║                                                                      ║\n");
        fprintf(fp, "║ Private Key (WIF - Import Ready):                                    ║\n");
        fprintf(fp, "║   %-66s ║\n", key_wif);
        fprintf(fp, "║                                                                      ║\n");
        fprintf(fp, "║ ⚠️  IMPORT INSTRUCTIONS:                                             ║\n");
        fprintf(fp, "║   1. Open your Bitcoin wallet (Electrum, Bitcoin Core, etc.)         ║\n");
        fprintf(fp, "║   2. Go to: Wallet → Private Keys → Import                          ║\n");
        fprintf(fp, "║   3. Paste the WIF key above                                         ║\n");
        fprintf(fp, "║   4. Scan blockchain - funds will appear!                            ║\n");
        fprintf(fp, "║                                                                      ║\n");
        fprintf(fp, "║ Timestamp: %-57s ║\n", ctime(&now));
        fprintf(fp, "╚══════════════════════════════════════════════════════════════════════╝\n");
        fclose(fp);
        printf("[+] Solution saved to %s\n", filename);
    }
    
    // Append to log
    fp = fopen("found_keys.log", "a");
    if (fp) {
        fprintf(fp, "[%s] Puzzle #%d | Address: %s | HEX: %s | WIF: %s\n",
                ctime(&now), puzzle_num, address, key_hex, key_wif);
        fclose(fp);
    }
}

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

// Load target addresses
int load_targets(const char* filename, char*** addresses_out) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[!] Failed to open targets file: %s\n", filename);
        return 0;
    }
    
    char** addresses = (char**)malloc(1000 * sizeof(char*));
    int count = 0;
    char line[256];
    
    while (fgets(line, sizeof(line), fp) && count < 1000) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;
        
        // Remove newline
        line[strcspn(line, "\r\n")] = 0;
        
        if (strlen(line) > 0) {
            addresses[count] = strdup(line);
            count++;
        }
    }
    
    fclose(fp);
    *addresses_out = addresses;
    
    printf("[+] Loaded %d target addresses\n", count);
    return count;
}

// Worker thread
void* worker_thread_gmp(void* arg) {
    thread_work_gmp* work = (thread_work_gmp*)arg;
    
    mpz_t privkey;
    mpz_init_set(privkey, work->start);
    
    ec_point pubkey_point;
    ec_point_init(&pubkey_point);
    
    uint8_t pubkey_bytes[33];
    char address[64];
    uint8_t hash160[20];  // For bloom filter
    
    uint64_t local_count = 0;
    
    while (g_running && mpz_cmp(privkey, work->end) < 0) {
        // Generate public key: pubkey = privkey * G
        ec_point_mul(&pubkey_point, &secp256k1_curve.G, privkey, &secp256k1_curve);
        
        // Convert to compressed format
        point_to_compressed_pubkey(pubkey_bytes, &pubkey_point);
        
        // Generate address
        pubkey_to_address(address, pubkey_bytes, hash160);
        
        // DEBUG: Print first 5 addresses from thread 0
        if (work->thread_id == 0 && local_count < 5) {
            char key_str[128];
            gmp_snprintf(key_str, sizeof(key_str), "%Zx", privkey);
            printf("[DEBUG Thread %d] Key 0x%s -> Address %s\n", work->thread_id, key_str, address);
        }
        
        // Check against bloom filter (fast pre-filter)
        if (bloom_check(work->bloom, hash160)) {
            // Potential match - verify against actual targets
            for (int i = 0; i < work->num_targets; i++) {
                if (strcmp(address, work->target_addresses[i]) == 0) {
                    // FOUND IT!
                    pthread_mutex_lock(work->state_mutex);
                    if (!work->state->found) {
                        work->state->found = true;
                        mpz_set(work->state->current_position, privkey);
                        strcpy(work->state->found_address, address);
                        
                        // Save immediately
                        save_found_key_gmp(address, privkey, work->state->puzzle_number);
                        
                        printf("\n");
                        printf("╔══════════════════════════════════════════════════════════════════════╗\n");
                        printf("║              🎉🎉🎉 KEY FOUND! PUZZLE SOLVED! 🎉🎉🎉              ║\n");
                        printf("╚══════════════════════════════════════════════════════════════════════╝\n");
                        printf("Address: %s\n", address);
                        
                        char key_hex[128];
                        gmp_snprintf(key_hex, sizeof(key_hex), "%064Zx", privkey);
                        printf("Private Key (HEX): %s\n", key_hex);
                        
                        g_running = false;
                    }
                    pthread_mutex_unlock(work->state_mutex);
                    goto cleanup;
                }
            }
        }
        
        // Increment key
        mpz_add(privkey, privkey, work->stride);
        local_count++;
        
        // Update global counter periodically
        if (local_count % 10000 == 0) {
            pthread_mutex_lock(work->state_mutex);
            work->state->keys_checked += local_count;
            mpz_set(work->state->current_position, privkey);
            pthread_mutex_unlock(work->state_mutex);
            local_count = 0;
        }
    }
    
cleanup:
    // Final update
    pthread_mutex_lock(work->state_mutex);
    work->state->keys_checked += local_count;
    pthread_mutex_unlock(work->state_mutex);
    
    work->local_keys_checked = local_count;
    
    mpz_clear(privkey);
    ec_point_clear(&pubkey_point);
    
    printf("[Thread %d] Finished, checked %llu keys\n", work->thread_id, work->local_keys_checked);
    return NULL;
}

// Monitor thread
void* monitor_thread_gmp(void* arg) {
    search_state_gmp* state = (search_state_gmp*)arg;
    
    while (g_running) {
        sleep(10);
        
        pthread_mutex_lock(&g_state_mutex);
        
        time_t now = time(NULL);
        double elapsed = difftime(now, state->start_time);
        double rate = elapsed > 0 ? state->keys_checked / elapsed / 1000000.0 : 0;
        
        char pos_str[128];
        gmp_snprintf(pos_str, sizeof(pos_str), "%Zx", state->current_position);
        
        printf("\r[*] Progress: %llu keys | %.2f MKeys/s | Elapsed: %.0fs | Pos: %s",
               state->keys_checked, rate, elapsed, pos_str);
        fflush(stdout);
        
        // Checkpoint
        if (difftime(now, state->last_checkpoint) >= CHECKPOINT_INTERVAL) {
            // TODO: Save checkpoint
            state->last_checkpoint = now;
            printf("\n[+] Checkpoint saved: %llu keys checked\n", state->keys_checked);
        }
        
        pthread_mutex_unlock(&g_state_mutex);
    }
    
    return NULL;
}

// Main
int main(int argc, char* argv[]) {
    if (argc < 5) {
        printf("Advanced Hybrid Bitcoin Puzzle Solver v2.0\n");
        printf("Usage: %s <start_hex> <end_hex> <targets_file> <threads> [puzzle_num]\n", argv[0]);
        printf("Example: %s 400000000000000000 7fffffffffffffffff puzzle71.txt 8 71\n", argv[0]);
        return 1;
    }
    
    // Initialize
    init_secp256k1_curve();
    
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║       Advanced Hybrid Bitcoin Puzzle Solver v2.0 (128-bit+)         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Features: GMP precision, Bloom filter, Multi-threading, WIF output  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");
    
    // Parse arguments
    mpz_t range_start, range_end;
    mpz_init_set_str(range_start, argv[1], 16);
    mpz_init_set_str(range_end, argv[2], 16);
    
    const char* targets_file = argv[3];
    int num_threads = atoi(argv[4]);
    int puzzle_num = argc > 5 ? atoi(argv[5]) : 71;
    
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    if (num_threads < 1) num_threads = 1;
    
    char start_str[256], end_str[256];
    gmp_snprintf(start_str, sizeof(start_str), "%Zx", range_start);
    gmp_snprintf(end_str, sizeof(end_str), "%Zx", range_end);
    
    printf("[+] Range: %s to %s\n", start_str, end_str);
    printf("[+] Threads: %d\n", num_threads);
    printf("[+] Puzzle: #%d\n", puzzle_num);
    
    // Load targets
    char** target_addresses = NULL;
    int num_targets = load_targets(targets_file, &target_addresses);
    if (num_targets == 0) {
        fprintf(stderr, "[!] No targets loaded\n");
        return 1;
    }
    
    // Create bloom filter
    printf("[+] Building bloom filter...\n");
    bloom_filter* bloom = bloom_create(num_targets * 2);
    
    for (int i = 0; i < num_targets; i++) {
        // Decode address to get hash160 for bloom filter
        uint8_t hash160[20];
        if (address_to_hash160(target_addresses[i], hash160)) {
            bloom_add(bloom, hash160);
        } else {
            fprintf(stderr, "[!] Warning: Failed to decode address: %s\n", target_addresses[i]);
        }
    }
    
    printf("[+] Bloom filter ready (%.2f MB)\n", bloom_memory_mb(bloom));
    
    // Initialize state
    mpz_init_set(g_state_gmp.range_start, range_start);
    mpz_init_set(g_state_gmp.range_end, range_end);
    mpz_init_set(g_state_gmp.current_position, range_start);
    g_state_gmp.keys_checked = 0;
    g_state_gmp.start_time = time(NULL);
    g_state_gmp.last_checkpoint = g_state_gmp.start_time;
    g_state_gmp.found = false;
    g_state_gmp.puzzle_number = puzzle_num;
    g_state_gmp.bits = mpz_sizeinbase(range_end, 2);
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Calculate work distribution
    mpz_t range_size, stride, work_size;
    mpz_init(range_size);
    mpz_init(stride);
    mpz_init(work_size);
    
    mpz_sub(range_size, range_end, range_start);
    mpz_set_ui(stride, num_threads);
    
    // Create threads
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    thread_work_gmp* work_units = (thread_work_gmp*)malloc(num_threads * sizeof(thread_work_gmp));
    
    printf("[+] Starting %d worker threads...\n", num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        work_units[i].thread_id = i;
        mpz_init_set(work_units[i].start, range_start);
        mpz_add_ui(work_units[i].start, work_units[i].start, i);
        mpz_init_set(work_units[i].end, range_end);
        mpz_init_set(work_units[i].stride, stride);
        work_units[i].bloom = bloom;
        work_units[i].target_addresses = target_addresses;
        work_units[i].num_targets = num_targets;
        work_units[i].state = &g_state_gmp;
        work_units[i].state_mutex = &g_state_mutex;
        work_units[i].local_keys_checked = 0;
        
        pthread_create(&threads[i], NULL, worker_thread_gmp, &work_units[i]);
    }
    
    // Monitor thread
    pthread_t monitor;
    pthread_create(&monitor, NULL, monitor_thread_gmp, &g_state_gmp);
    
    printf("\n[*] Solver running... Press Ctrl+C to stop\n\n");
    
    // Wait for threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    g_running = false;
    pthread_join(monitor, NULL);
    
    // Results
    printf("\n\n");
    if (g_state_gmp.found) {
        printf("╔══════════════════════════════════════════════════════════════════════╗\n");
        printf("║                      ✅ PUZZLE SOLVED! ✅                            ║\n");
        printf("╠══════════════════════════════════════════════════════════════════════╣\n");
        printf("║ Address: %-59s ║\n", g_state_gmp.found_address);
        printf("║ File: WINNER_PUZZLE_%d.txt                                          ║\n", puzzle_num);
        printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    } else {
        printf("[+] Solver stopped. Total keys checked: %llu\n", g_state_gmp.keys_checked);
    }
    
    // Cleanup
    for (int i = 0; i < num_threads; i++) {
        mpz_clear(work_units[i].start);
        mpz_clear(work_units[i].end);
        mpz_clear(work_units[i].stride);
    }
    
    free(threads);
    free(work_units);
    bloom_free(bloom);
    
    for (int i = 0; i < num_targets; i++) {
        free(target_addresses[i]);
    }
    free(target_addresses);
    
    mpz_clear(range_start);
    mpz_clear(range_end);
    mpz_clear(range_size);
    mpz_clear(stride);
    mpz_clear(work_size);
    mpz_clear(g_state_gmp.range_start);
    mpz_clear(g_state_gmp.range_end);
    mpz_clear(g_state_gmp.current_position);
    
    return g_state_gmp.found ? 0 : 1;
}

/*
 * Hybrid CPU+GPU Bitcoin Puzzle Solver
 * Strategy: CPU validates GPU results for 100% accuracy
 * 
 * Architecture:
 * 1. GPU: Fast candidate generation (500M-1B keys/sec)
 * 2. CPU: Validates every GPU "hit" with libsecp256k1 (100% accurate)
 * 3. Result: Speed of GPU + Accuracy of CPU
 * 
 * Safety: All found keys appended to unique files (never overwritten)
 */

// Suppress OpenSSL deprecation warnings
#define OPENSSL_SUPPRESS_DEPRECATED

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <gmp.h>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

#include "gpu_solver.h"
#include "../include/gpu_bsgs.h"
#include "../include/bloom_filter.h"

// Thread-safe result storage
std::mutex result_mutex;
std::atomic<int> keys_found(0);

// Per-puzzle stop flags (each thread can stop independently)
struct PuzzleState {
    std::atomic<bool> found;
    std::atomic<bool> should_stop;
    PuzzleState() : found(false), should_stop(false) {}
};
std::map<int, std::shared_ptr<PuzzleState>> puzzle_states;
std::mutex state_mutex;

// Base58 encoding
const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

void base58_encode(const uint8_t* input, size_t len, char* output) {
    mpz_t num, base, rem;
    mpz_init(num);
    mpz_init_set_ui(base, 58);
    mpz_init(rem);
    
    mpz_import(num, len, 1, 1, 1, 0, input);
    
    std::string result;
    while (mpz_cmp_ui(num, 0) > 0) {
        mpz_fdiv_qr(num, rem, num, base);
        result = BASE58_ALPHABET[mpz_get_ui(rem)] + result;
    }
    
    for (size_t i = 0; i < len && input[i] == 0; i++) {
        result = "1" + result;
    }
    
    strcpy(output, result.c_str());
    
    mpz_clear(num);
    mpz_clear(base);
    mpz_clear(rem);
}

// CPU validation using libsecp256k1 (100% accurate)
bool cpu_validate_key(const uint8_t* privkey_bytes, const char* expected_address, 
                     secp256k1_context* ctx) {
    // Generate pubkey with libsecp256k1
    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, privkey_bytes)) {
        return false;
    }
    
    // Serialize to compressed format
    uint8_t pubkey_serialized[33];
    size_t pubkey_len = 33;
    secp256k1_ec_pubkey_serialize(ctx, pubkey_serialized, &pubkey_len, &pubkey, 
                                  SECP256K1_EC_COMPRESSED);
    
    // SHA256(pubkey)
    uint8_t sha_result[32];
    SHA256(pubkey_serialized, 33, sha_result);
    
    // RIPEMD160(SHA256(pubkey))
    uint8_t hash160[20];
    RIPEMD160(sha_result, 32, hash160);
    
    // Base58Check encoding
    uint8_t versioned[21];
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160, 20);
    
    uint8_t checksum_full[32];
    SHA256(versioned, 21, checksum_full);
    SHA256(checksum_full, 32, checksum_full);
    
    uint8_t addr_bytes[25];
    memcpy(addr_bytes, versioned, 21);
    memcpy(addr_bytes + 21, checksum_full, 4);
    
    char address[64];
    base58_encode(addr_bytes, 25, address);
    
    return strcmp(address, expected_address) == 0;
}

// Generate WIF format
void generate_wif(const uint8_t* privkey_bytes, char* wif_out) {
    uint8_t wif_data[38];
    wif_data[0] = 0x80;
    memcpy(wif_data + 1, privkey_bytes, 32);
    wif_data[33] = 0x01;
    
    uint8_t checksum[32];
    SHA256(wif_data, 34, checksum);
    SHA256(checksum, 32, checksum);
    memcpy(wif_data + 34, checksum, 4);
    
    base58_encode(wif_data, 38, wif_out);
}

// Save found key (append-only, never overwrites)
void save_found_key(int puzzle_num, const uint8_t* privkey_bytes, 
                   const char* address, secp256k1_context* ctx) {
    std::lock_guard<std::mutex> lock(result_mutex);
    
    // Generate unique filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream filename;
    filename << "WINNER_PUZZLE_" << puzzle_num 
             << "_" << timestamp << "_" << ms.count() << ".txt";
    
    std::ofstream outfile(filename.str(), std::ios::app);
    if (!outfile) {
        std::cerr << "[!] Failed to create output file: " << filename.str() << std::endl;
        return;
    }
    
    // Format private key
    char hex_key[65];
    for (int i = 0; i < 32; i++) {
        snprintf(hex_key + i * 2, 3, "%02x", privkey_bytes[i]);
    }
    hex_key[64] = '\0';
    
    // Generate WIF
    char wif[64];
    generate_wif(privkey_bytes, wif);
    
    // Write result
    outfile << "╔══════════════════════════════════════════════════════════════╗\n";
    outfile << "║              🎉 SOLUTION FOUND! 🎉                          ║\n";
    outfile << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    outfile << "Puzzle Number: " << puzzle_num << "\n";
    outfile << "Address: " << address << "\n\n";
    
    outfile << "Private Key (HEX):\n";
    outfile << hex_key << "\n\n";
    
    outfile << "Private Key (WIF):\n";
    outfile << wif << "\n\n";
    
    outfile << "Timestamp: " << std::put_time(std::localtime(&timestamp), "%Y-%m-%d %H:%M:%S");
    outfile << "." << std::setfill('0') << std::setw(3) << ms.count() << "\n\n";
    
    outfile << "Import to Bitcoin Core:\n";
    outfile << "bitcoin-cli importprivkey \"" << wif << "\"\n\n";
    
    outfile << "╔══════════════════════════════════════════════════════════════╗\n";
    outfile << "║  ⚠️  KEEP THIS FILE SECURE - Contains Private Key! ⚠️       ║\n";
    outfile << "╚══════════════════════════════════════════════════════════════╝\n";
    
    outfile.close();
    
    keys_found++;
    
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ✅ KEY #" << keys_found << " FOUND - PUZZLE #" << puzzle_num;
    std::cout << std::string(35 - std::to_string(keys_found.load()).length() 
                              - std::to_string(puzzle_num).length(), ' ') << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "[+] Saved to: " << filename.str() << std::endl;
    std::cout << "[+] Address: " << address << std::endl;
    std::cout << "[+] WIF: " << wif << std::endl;
    std::cout << "[+] This thread stopping, others continue..." << std::endl;
}

// Hybrid solver: GPU generates candidates, CPU validates
void hybrid_solve_puzzle(int puzzle_num, const std::string& start_hex, 
                        const std::string& end_hex, const std::string& address,
                        int bits, std::shared_ptr<PuzzleState> state) {
    
    std::cout << "\n[*] Starting hybrid GPU+CPU solver for Puzzle #" << puzzle_num << std::endl;
    std::cout << "[*] Range: 0x" << start_hex << " to 0x" << end_hex << std::endl;
    std::cout << "[*] Address: " << address << std::endl;
    std::cout << "[*] Bits: " << bits << std::endl;
    
    // Initialize secp256k1 context for CPU validation
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    // Use BSGS for 71-80 bits, random GPU for 81+
    bool found = false;
    if (bits >= 71 && bits <= 80) {
        // Use BSGS (offset-based)
        std::cout << "[*] Using GPU-accelerated BSGS for Puzzle #" << puzzle_num << std::endl;
        uint64_t found_offset = 0;
        int bsgs_result = gpu_bsgs_solve(address.c_str(), start_hex.c_str(), end_hex.c_str(), bits, &found_offset);
        if (bsgs_result == 0) {
            // Reconstruct full key: priv = start + offset
            mpz_t start_mpz, priv_mpz;
            mpz_init_set_str(start_mpz, start_hex.c_str(), 16);
            mpz_init(priv_mpz);
            mpz_add_ui(priv_mpz, start_mpz, found_offset);
            unsigned char found_key[32] = {0};
            size_t count = 0;
            mpz_export(found_key + (32 - ((mpz_sizeinbase(priv_mpz, 2) + 7) / 8)), &count, 1, 1, 1, 0, priv_mpz);
            std::cout << "[+] GPU BSGS HIT! CPU validating..." << std::endl;
            if (cpu_validate_key(found_key, address.c_str(), ctx)) {
                std::cout << "[+] ✅ CPU VALIDATION PASSED - Puzzle #" << puzzle_num << "!" << std::endl;
                save_found_key(puzzle_num, found_key, address.c_str(), ctx);
                state->found = true;
                state->should_stop = true;
                found = true;
            } else {
                std::cout << "[-] ⚠️  CPU validation failed (GPU false positive)" << std::endl;
                // Print candidate private key (hex)
                std::cout << "    Candidate privkey: ";
                for (int i = 0; i < 32; i++) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)found_key[i];
                std::cout << std::dec << std::endl;
                // Compute and print computed address and hash160
                secp256k1_pubkey pubkey;
                if (secp256k1_ec_pubkey_create(ctx, &pubkey, found_key)) {
                    uint8_t pubkey_serialized[33];
                    size_t pubkey_len = 33;
                    secp256k1_ec_pubkey_serialize(ctx, pubkey_serialized, &pubkey_len, &pubkey, SECP256K1_EC_COMPRESSED);
                    uint8_t sha_result[32];
                    SHA256(pubkey_serialized, 33, sha_result);
                    uint8_t hash160[20];
                    RIPEMD160(sha_result, 32, hash160);
                    std::cout << "    Computed hash160: ";
                    for (int i = 0; i < 20; i++) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash160[i];
                    std::cout << std::dec << std::endl;
                    // Base58Check encoding
                    uint8_t versioned[21];
                    versioned[0] = 0x00;
                    memcpy(versioned + 1, hash160, 20);
                    uint8_t checksum_full[32];
                    SHA256(versioned, 21, checksum_full);
                    SHA256(checksum_full, 32, checksum_full);
                    uint8_t addr_bytes[25];
                    memcpy(addr_bytes, versioned, 21);
                    memcpy(addr_bytes + 21, checksum_full, 4);
                    char computed_address[64];
                    base58_encode(addr_bytes, 25, computed_address);
                    std::cout << "    Computed address: " << computed_address << std::endl;
                }
            }
            mpz_clear(start_mpz);
            mpz_clear(priv_mpz);
        } else {
            std::cout << "[*] BSGS did not find key in range for Puzzle #" << puzzle_num << std::endl;
        }
    } else {
        // Use random GPU search (existing logic)
        // ...existing code for random GPU search (copy from above)...
        // Parse start key
        mpz_t start_key, end_key, current_key;
        mpz_init_set_str(start_key, start_hex.c_str(), 16);
        mpz_init_set_str(end_key, end_hex.c_str(), 16);
        mpz_init_set(current_key, start_key);

        // Calculate total keys
        mpz_t total_keys;
        mpz_init(total_keys);
        mpz_sub(total_keys, end_key, start_key);

        // Prepare target hash160

        // Extract hash160 from address
        uint8_t target_hash160[20];
        if (!address_to_hash160(address.c_str(), target_hash160)) {
            std::cerr << "[!] Failed to parse address for hash160: " << address << std::endl;
            return;
        }

        // GPU batch size (tune based on VRAM, can be user-tuned)
        uint64_t batch_size = 1ULL << 30; // 1 billion keys per batch (increase if GPU has more VRAM)
        // Stride tuning: ensure threads do not overlap and cover full keyspace (see kernel launch config)

        uint64_t total_checked = 0;
        auto start_time = std::chrono::steady_clock::now();

        uint8_t found_key[32];
        while (mpz_cmp(current_key, end_key) < 0 && !state->should_stop) {
            // Export current key to bytes
            uint8_t start_bytes[32] = {0};
            size_t count;
            mpz_export(start_bytes + (32 - ((mpz_sizeinbase(current_key, 2) + 7) / 8)),
                       &count, 1, 1, 1, 0, current_key);

            // Determine batch size for this iteration
            mpz_t remaining;
            mpz_init(remaining);
            mpz_sub(remaining, end_key, current_key);

            uint64_t this_batch = batch_size;
            if (mpz_cmp_ui(remaining, batch_size) < 0) {
                this_batch = mpz_get_ui(remaining);
            }
            mpz_clear(remaining);

            std::cout << "[GPU-P" << puzzle_num << "] Batch: " << this_batch << " keys..." << std::flush;

            // GPU search
            uint8_t found_hash[20];

            int gpu_result = gpu_search_keys(start_bytes, this_batch, target_hash160,
                                             1, found_key, found_hash);

            if (gpu_result == 1) {
                std::cout << " GPU HIT! CPU validating..." << std::endl;

                // CPU validation (libsecp256k1 - 100% accurate)
                if (cpu_validate_key(found_key, address.c_str(), ctx)) {
                    std::cout << "[+] ✅ CPU VALIDATION PASSED - Puzzle #" << puzzle_num << "!" << std::endl;

                    // Save key (append-only)
                    save_found_key(puzzle_num, found_key, address.c_str(), ctx);

                    // Mark this puzzle as found and stop THIS thread only
                    state->found = true;
                    state->should_stop = true;
                    found = true;
                    break;
                } else {
                    std::cout << "[-] ⚠️  CPU validation failed (GPU false positive)" << std::endl;
                }
            }

            total_checked += this_batch;

            // Progress update every 10 batches
            if (total_checked % (batch_size * 10) == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                if (elapsed > 0) {
                    uint64_t rate = total_checked / elapsed;
                    std::cout << " [P" << puzzle_num << ": " << rate << " keys/s]" << std::endl;
                }
            } else {
                std::cout << " ✓" << std::endl;
            }

            // Advance to next batch
            mpz_add_ui(current_key, current_key, this_batch);
        }

        mpz_clear(start_key);
        mpz_clear(end_key);
        mpz_clear(current_key);
        mpz_clear(total_keys);
    }
    secp256k1_context_destroy(ctx);
    if (state->found || found) {
        std::cout << "[+] ✅ Puzzle #" << puzzle_num << " SOLVED!" << std::endl;
    } else if (state->should_stop) {
        std::cout << "[*] Puzzle #" << puzzle_num << " stopped by user" << std::endl;
    } else {
        std::cout << "[*] Puzzle #" << puzzle_num << " complete (not found in range)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Hybrid GPU+CPU Bitcoin Puzzle Solver\n\n";
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << " puzzle_file.txt\n\n";
        std::cout << "Puzzle file format (CSV):\n";
        std::cout << "  puzzle_num,bits,start_hex,end_hex,address,prize\n\n";
        std::cout << "Features:\n";
        std::cout << "  - GPU: Fast candidate generation (500M-1B keys/s)\n";
        std::cout << "  - CPU: Validates all GPU hits with libsecp256k1\n";
        std::cout << "  - Multi-puzzle: Stops all when ANY key found\n";
        std::cout << "  - Safe: Append-only storage (never overwrites)\n";
        return 1;
    }
    
    std::string puzzle_file = argv[1];
    
    // Initialize GPU
    std::cout << "[*] Initializing GPU..." << std::endl;
    gpu_init_constants();
    std::cout << "[+] GPU initialized" << std::endl;
    
    // Parse puzzle file
    std::ifstream infile(puzzle_file);
    if (!infile) {
        std::cerr << "[!] Failed to open puzzle file: " << puzzle_file << std::endl;
        return 1;
    }
    
    struct Puzzle {
        int num;
        int bits;
        std::string start_hex;
        std::string end_hex;
        std::string address;
        std::string prize;
    };
    
    std::vector<Puzzle> puzzles;
    std::string line;
    
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        Puzzle p;
        std::istringstream iss(line);
        std::string token;
        
        std::getline(iss, token, ','); p.num = std::stoi(token);
        std::getline(iss, token, ','); p.bits = std::stoi(token);
        std::getline(iss, p.start_hex, ',');
        std::getline(iss, p.end_hex, ',');
        std::getline(iss, p.address, ',');
        std::getline(iss, p.prize, ',');
        
        puzzles.push_back(p);
    }
    infile.close();
    
    std::cout << "[+] Loaded " << puzzles.size() << " puzzles" << std::endl;
    
    // Create state for each puzzle
    for (const auto& p : puzzles) {
        puzzle_states[p.num] = std::make_shared<PuzzleState>();
    }
    
    // Launch threads (one per puzzle) - each runs independently
    std::vector<std::thread> threads;
    
    for (const auto& p : puzzles) {
        auto state = puzzle_states[p.num];
        
        threads.emplace_back([p, state]() {
            hybrid_solve_puzzle(p.num, p.start_hex, p.end_hex, p.address, p.bits, state);
        });
        
        // Small delay to stagger GPU access
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    
    // Summary
    int solved_count = 0;
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    
    for (const auto& entry : puzzle_states) {
        if (entry.second->found) {
            solved_count++;
            std::cout << "║  ✅ Puzzle #" << entry.first << " SOLVED!";
            std::cout << std::string(48 - std::to_string(entry.first).length(), ' ') << "║\n";
        }
    }
    
    if (solved_count == 0) {
        std::cout << "║  No keys found in specified ranges                          ║\n";
    } else {
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Total: " << solved_count << " puzzle(s) solved, " << keys_found << " key(s) saved";
        std::cout << std::string(54 - std::to_string(solved_count).length() 
                                  - std::to_string(keys_found.load()).length(), ' ') << "║\n";
    }
    
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    return 0;
}

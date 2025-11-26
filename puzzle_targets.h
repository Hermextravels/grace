#ifndef PUZZLE_TARGETS_H
#define PUZZLE_TARGETS_H

#include <string>
#include <vector>

struct PuzzleTarget {
    int number;
    int bits;
    std::string start_hex;
    std::string end_hex;
    std::string address;
    double btc;
    std::string pubkey; // empty if not available
};

// Unsolved puzzles 71-99 (most feasible for single GPU solving)
const std::vector<PuzzleTarget> PRIORITY_PUZZLES = {
    {71, 71, "400000000000000000", "7fffffffffffffffff", "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", 7.1, ""},
    {72, 72, "800000000000000000", "ffffffffffffffffff", "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR", 7.2, ""},
    {73, 73, "1000000000000000000", "1ffffffffffffffffff", "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", 7.3, ""},
    {74, 74, "2000000000000000000", "3ffffffffffffffffff", "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv", 7.4, ""},
    {76, 76, "8000000000000000000", "fffffffffffffffffff", "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF", 7.6, ""},
    {77, 77, "10000000000000000000", "1fffffffffffffffffff", "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE", 7.7, ""},
    {78, 78, "20000000000000000000", "3fffffffffffffffffff", "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb", 7.8, ""},
    {79, 79, "40000000000000000000", "7fffffffffffffffffff", "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8", 7.9, ""},
    {81, 81, "100000000000000000000", "1ffffffffffffffffffff", "15qsCm78whspNQFydGJQk5rexzxTQopnHZ", 8.1, ""},
    {82, 82, "200000000000000000000", "3ffffffffffffffffffff", "13zYrYhhJxp6Ui1VV7pqa5WDhNWM45ARAC", 8.2, ""},
    {83, 83, "400000000000000000000", "7ffffffffffffffffffff", "14MdEb4eFcT3MVG5sPFG4jGLuHJSnt1Dk2", 8.3, ""},
    {84, 84, "800000000000000000000", "fffffffffffffffffffff", "1CMq3SvFcVEcpLMuuH8PUcNiqsK1oicG2D", 8.4, ""},
    {86, 86, "2000000000000000000000", "3fffffffffffffffffffff", "1K3x5L6G57Y494fDqBfrojD28UJv4s5JcK", 8.6, ""},
    {87, 87, "4000000000000000000000", "7fffffffffffffffffffff", "1PxH3K1Shdjb7gSEoTX7UPDZ6SH4qGPrvq", 8.7, ""},
    {88, 88, "8000000000000000000000", "ffffffffffffffffffffff", "16AbnZjZZipwHMkYKBSfswGWKDmXHjEpSf", 8.8, ""},
    {89, 89, "10000000000000000000000", "1ffffffffffffffffffffff", "19QciEHbGVNY4hrhfKXmcBBCrJSBZ6TaVt", 8.9, ""},
    {91, 91, "40000000000000000000000", "7ffffffffffffffffffffff", "1EzVHtmbN4fs4MiNk3ppEnKKhsmXYJ4s74", 9.1, ""},
    {92, 92, "80000000000000000000000", "fffffffffffffffffffffff", "1AE8NzzgKE7Yhz7BWtAcAAxiFMbPo82NB5", 9.2, ""},
    {93, 93, "100000000000000000000000", "1fffffffffffffffffffffff", "17Q7tuG2JwFFU9rXVj3uZqRtioH3mx2Jad", 9.3, ""},
    {94, 94, "200000000000000000000000", "3fffffffffffffffffffffff", "1K6xGMUbs6ZTXBnhw1pippqwK6wjBWtNpL", 9.4, ""},
    {96, 96, "800000000000000000000000", "ffffffffffffffffffffffff", "15ANYzzCp5BFHcCnVFzXqyibpzgPLWaD8b", 9.6, ""},
    {97, 97, "1000000000000000000000000", "1ffffffffffffffffffffffff", "18ywPwj39nGjqBrQJSzZVq2izR12MDpDr8", 9.7, ""},
    {98, 98, "2000000000000000000000000", "3ffffffffffffffffffffffff", "1CaBVPrwUxbQYYswu32w7Mj4HR4maNoJSX", 9.8, ""},
    {99, 99, "4000000000000000000000000", "7ffffffffffffffffffffffff", "1JWnE6p6UN7ZJBN7TtcbNDoRcjFtuDWoNL", 9.9, ""}
};

// Puzzles with public keys (can use kangaroo methods)
const std::vector<PuzzleTarget> PUBKEY_PUZZLES = {
    {135, 135, "4000000000000000000000000000000000", "7fffffffffffffffffffffffffffffffff", "16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v", 13.5, "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"},
    {140, 140, "80000000000000000000000000000000000", "fffffffffffffffffffffffffffffffffff", "1QKBaU6WAeycb3DbKbLBkX7vJiaS8r42Xo", 14.0, "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640"},
    {145, 145, "1000000000000000000000000000000000000", "1ffffffffffffffffffffffffffffffffffff", "19GpszRNUej5yYqxXoLnbZWKew3KdVLkXg", 14.5, "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5"},
    {150, 150, "20000000000000000000000000000000000000", "3fffffffffffffffffffffffffffffffffffff", "1MUJSJYtGPVGkBCTqGspnxyHahpt5Te8jy", 15.0, "03137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49"},
    {155, 155, "400000000000000000000000000000000000000", "7ffffffffffffffffffffffffffffffffffffff", "1AoeP37TmHdFh8uN72fu9AqgtLrUwcv2wJ", 15.5, "035cd1854cae45391ca4ec428cc7e6c7d9984424b954209a8eea197b9e364c05f6"},
    {160, 160, "8000000000000000000000000000000000000000", "ffffffffffffffffffffffffffffffffffffffff", "1NBC8uXJy1GiJ6drkiZa1WuKn51ps7EPTv", 16.0, "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673"}
};

#endif // PUZZLE_TARGETS_H

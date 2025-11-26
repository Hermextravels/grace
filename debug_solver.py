#!/usr/bin/env python3
"""Debug the GMP solver by manually checking if it should find puzzle #66"""

import hashlib
from ecdsa import SECP256k1, SigningKey
import base58

def private_to_address_hash160(priv_hex):
    """Convert private key to Bitcoin address and hash160"""
    priv_bytes = bytes.fromhex(priv_hex.zfill(64))
    sk = SigningKey.from_string(priv_bytes, curve=SECP256k1)
    vk = sk.verifying_key
    
    # Compressed public key
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    
    if y % 2 == 0:
        pubkey = b'\x02' + x.to_bytes(32, 'big')
    else:
        pubkey = b'\x03' + x.to_bytes(32, 'big')
    
    # SHA256 then RIPEMD160
    sha256_hash = hashlib.sha256(pubkey).digest()
    ripemd160 = hashlib.new('ripemd160', sha256_hash).digest()
    
    # Add version byte
    versioned = b'\x00' + ripemd160
    checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
    address = base58.b58encode(versioned + checksum).decode()
    
    return address, pubkey.hex(), ripemd160.hex()

# Test the range the solver is checking
print("=" * 80)
print("DEBUGGING GMP SOLVER - Puzzle #66")
print("=" * 80)

target_key = int("2832ed74f2b5e35ee", 16)
range_start = int("2832ed74f2b5e35e0", 16)
range_end = int("2832ed74f2b5e3600", 16)

print(f"\nTarget key:   0x{target_key:x} ({target_key})")
print(f"Range start:  0x{range_start:x} ({range_start})")
print(f"Range end:    0x{range_end:x} ({range_end})")
print(f"Key in range: {range_start <= target_key < range_end}")

print(f"\nKeys in range: {range_end - range_start}")

# Check several keys around the target
print("\n" + "=" * 80)
print("Checking keys in the range:")
print("=" * 80)

for offset in range(-2, 5):
    key_val = range_start + offset
    if key_val < 0:
        continue
    
    key_hex = f"{key_val:x}"
    addr, pubkey, hash160 = private_to_address_hash160(key_hex)
    
    marker = " ⭐ TARGET" if key_val == target_key else ""
    print(f"\nKey #{offset}: 0x{key_hex}{marker}")
    print(f"  Address: {addr}")
    print(f"  Hash160: {hash160}")
    
    if addr == "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so":
        print(f"  ✅ MATCH! This is puzzle #66!")

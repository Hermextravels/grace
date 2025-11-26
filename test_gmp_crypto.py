#!/usr/bin/env python3
"""Test GMP solver's crypto implementation against known puzzle #66"""

import hashlib
from ecdsa import SECP256k1, SigningKey
import base58

def private_to_address(priv_hex):
    """Convert private key to Bitcoin address"""
    # Parse private key
    priv_bytes = bytes.fromhex(priv_hex.zfill(64))
    
    # Generate public key
    sk = SigningKey.from_string(priv_bytes, curve=SECP256k1)
    vk = sk.verifying_key
    
    # Compressed public key (0x02 or 0x03 prefix)
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    
    if y % 2 == 0:
        pubkey = b'\x02' + x.to_bytes(32, 'big')
    else:
        pubkey = b'\x03' + x.to_bytes(32, 'big')
    
    # SHA256 then RIPEMD160
    sha256_hash = hashlib.sha256(pubkey).digest()
    ripemd160 = hashlib.new('ripemd160', sha256_hash).digest()
    
    # Add version byte (0x00 for mainnet)
    versioned = b'\x00' + ripemd160
    
    # Double SHA256 for checksum
    checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
    
    # Base58 encode
    address = base58.b58encode(versioned + checksum).decode()
    
    return address, pubkey.hex(), ripemd160.hex()

# Test puzzle #66
priv_key = "2832ed74f2b5e35ee"
expected_address = "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"

address, pubkey, hash160 = private_to_address(priv_key)

print("=" * 70)
print("PUZZLE #66 VERIFICATION")
print("=" * 70)
print(f"Private Key (HEX): {priv_key.zfill(64)}")
print(f"Public Key (compressed): {pubkey}")
print(f"Hash160: {hash160}")
print(f"Address Generated: {address}")
print(f"Expected Address:  {expected_address}")
print(f"Match: {'✅ YES' if address == expected_address else '❌ NO'}")
print("=" * 70)

# Also print what the bloom filter should be checking
print(f"\nBloom filter should check hash160: {hash160}")
print(f"Hash160 bytes: {bytes.fromhex(hash160)}")

#!/usr/bin/env python3
"""
Quick test to verify puzzle #56 solution generates correct address
"""
import hashlib
from ecdsa import SECP256k1, SigningKey
import base58

def private_key_to_address(privkey_hex):
    """Convert private key to Bitcoin address (compressed)"""
    sk = SigningKey.from_string(bytes.fromhex(privkey_hex), curve=SECP256k1)
    vk = sk.verifying_key
    
    # Compressed public key
    x = int.from_bytes(vk.to_string()[:32], 'big')
    y = int.from_bytes(vk.to_string()[32:], 'big')
    prefix = b'\x02' if y % 2 == 0 else b'\x03'
    pubkey_compressed = prefix + x.to_bytes(32, 'big')
    
    # Generate address
    sha = hashlib.sha256(pubkey_compressed).digest()
    ripe = hashlib.new('ripemd160', sha).digest()
    versioned = b'\x00' + ripe
    checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
    address = base58.b58encode(versioned + checksum).decode()
    
    return address, pubkey_compressed.hex()

def private_key_to_wif(privkey_hex):
    """Convert private key to WIF format (compressed)"""
    privkey_bytes = bytes.fromhex(privkey_hex)
    extended = b'\x80' + privkey_bytes + b'\x01'
    hash1 = hashlib.sha256(extended).digest()
    hash2 = hashlib.sha256(hash1).digest()
    checksum = hash2[:4]
    wif = base58.b58encode(extended + checksum).decode()
    return wif

# Test with puzzle #56
print("=" * 70)
print("PUZZLE #56 VERIFICATION TEST")
print("=" * 70)

privkey_hex = "000000000000000000000000000000000000000000000000009d18b63ac4ffdf"
expected_address = "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi"
expected_pubkey = "033f2db2074e3217b3e5ee305301eeebb1160c4fa1e993ee280112f6348637999a"

address, pubkey = private_key_to_address(privkey_hex)
wif = private_key_to_wif(privkey_hex)

print(f"\nPrivate Key (HEX):")
print(f"  {privkey_hex}")
print(f"\nPrivate Key (WIF):")
print(f"  {wif}")
print(f"\nPublic Key (compressed):")
print(f"  Generated: {pubkey}")
print(f"  Expected:  {expected_pubkey}")
print(f"  Match: {'✅' if pubkey == expected_pubkey else '❌'}")
print(f"\nBitcoin Address:")
print(f"  Generated: {address}")
print(f"  Expected:  {expected_address}")
print(f"  Match: {'✅' if address == expected_address else '❌'}")

if address == expected_address and pubkey == expected_pubkey:
    print(f"\n{'='*70}")
    print("✅ SUCCESS! All verifications passed.")
    print("The solver's crypto implementation should work correctly.")
    print(f"{'='*70}")
else:
    print(f"\n{'='*70}")
    print("❌ FAILED! Verification mismatch.")
    print(f"{'='*70}")

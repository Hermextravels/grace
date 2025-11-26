#!/bin/bash
# Fix libsecp256k1 library path issue on Linux GPU servers

echo "[*] Fixing library paths..."

# Update library cache
sudo ldconfig 2>/dev/null

# Add library path to environment
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Make permanent
if ! grep -q "LD_LIBRARY_PATH=/usr/local/lib" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "[+] Added LD_LIBRARY_PATH to ~/.bashrc"
fi

# Verify secp256k1 is found
echo ""
echo "[*] Checking libsecp256k1..."
if ldconfig -p 2>/dev/null | grep -q secp256k1; then
    echo "[✓] libsecp256k1 found in library cache"
    ldconfig -p | grep secp256k1
elif ls /usr/local/lib/libsecp256k1.so* >/dev/null 2>&1; then
    echo "[✓] libsecp256k1 found in /usr/local/lib"
    ls -lh /usr/local/lib/libsecp256k1.so*
else
    echo "[!] libsecp256k1 NOT FOUND - install it first!"
    exit 1
fi

echo ""
echo "[*] Testing solver..."
if ./hybrid_solver_secp --help >/dev/null 2>&1; then
    echo "[✓] CPU solver works!"
else
    echo "[!] CPU solver still has library issues"
    echo "[!] Run: source ~/.bashrc  OR  logout and login again"
fi

echo ""
echo "Run this to apply changes to current shell:"
echo "  source ~/.bashrc"

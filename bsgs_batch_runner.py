

import csv
import json
import os
import subprocess
import tempfile
import multiprocessing
import time

# --- CONFIG ---
CSV_FILE = 'puzzles_71-99.csv'
CHECKPOINT_FILE = 'bsgs_progress.json'
SOLVER_CMD = './hybrid_multi_solver'  # Use your batch solver
BATCH_SIZE = 2**56  # Number of keys per BSGS pass (matches ~12GB table)

# --- LOAD CHECKPOINT ---
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# --- HEX UTILS ---
def hex_to_int(h):
    return int(h, 16)

def int_to_hex(i):
    return format(i, 'x')


# --- PARALLEL PASS WORKER ---
def run_pass(args):
    puzzle_num, bits, chunk_start, chunk_end, address, btc_prize, pass_idx, passes, puzzle_key = args
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.csv') as tmpcsv:
        writer = csv.writer(tmpcsv)
        writer.writerow(['# Format: puzzle_number,bits,start_hex,end_hex,address,btc_prize'])
        writer.writerow([puzzle_num, bits, int_to_hex(chunk_start), int_to_hex(chunk_end), address, btc_prize])
        tmpcsv_path = tmpcsv.name
    cmd = [SOLVER_CMD, tmpcsv_path]
    print(f'  [*] Pass {pass_idx+1}/{passes}: {int_to_hex(chunk_start)} to {int_to_hex(chunk_end)}')
    print(f'      Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    found = 'KEY FOUND' in result.stdout
    os.unlink(tmpcsv_path)
    return (puzzle_key, pass_idx+1, found)

# --- MAIN LOOP ---
def main():
    checkpoint = load_checkpoint()
    max_parallel = 1  # Tesla T4: use only one process for best GPU efficiency
    print(f"[*] Running {max_parallel} pass at a time (Tesla T4 optimal).")
    with open(CSV_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            puzzle_num, bits, start_hex, end_hex, address, btc_prize = row
            try:
                puzzle_num_int = int(puzzle_num)
            except Exception:
                continue
            if puzzle_num_int < 71 or puzzle_num_int > 80:
                continue
            bits = int(bits)
            start = hex_to_int(start_hex)
            end = hex_to_int(end_hex)
            total_keys = end - start + 1
            passes = (total_keys + BATCH_SIZE - 1) // BATCH_SIZE
            print(f'\n=== Puzzle #{puzzle_num} ({bits} bits) ===')
            print(f'Address: {address}')
            print(f'Range: {start_hex} to {end_hex} ({total_keys} keys, {passes} passes)')
            puzzle_key = f'puzzle_{puzzle_num}'
            last_pass = checkpoint.get(puzzle_key, 0)
            found = False
            pass_args = []
            for p in range(last_pass, passes):
                chunk_start = start + p * BATCH_SIZE
                chunk_end = min(chunk_start + BATCH_SIZE - 1, end)
                pass_args.append((puzzle_num, bits, chunk_start, chunk_end, address, btc_prize, p, passes, puzzle_key))
            with multiprocessing.Pool(processes=max_parallel) as pool:
                for result in pool.imap_unordered(run_pass, pass_args):
                    key, pass_idx, pass_found = result
                    checkpoint[key] = pass_idx
                    save_checkpoint(checkpoint)
                    if pass_found:
                        print(f'  [+] Key found for puzzle {puzzle_num}!')
                        found = True
                        break
            if not found:
                print(f'  [-] No key found for puzzle {puzzle_num} in searched range.')

if __name__ == '__main__':
    main()

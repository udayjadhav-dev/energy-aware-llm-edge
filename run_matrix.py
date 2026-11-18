#!/usr/bin/env python3
"""
run_matrix.py - run the whole experiment matrix unattended.

Matrix: quantization {q4, q8} x parallel requests {1, 2, 4} x 3 repeats = 18 runs,
in a fixed random order (so heat/drift does not bias one config).
Finished runs are skipped, so you can stop it and start it again later.

Usage (in the folder with energy_log.py and the .gguf files):
  python3 run_matrix.py --dry          # only print the plan
  sudo python3 run_matrix.py           # run everything (about 45-60 min)
"""
import argparse
import os
import random
import subprocess
import sys
import time

MODELS = {"q4": "3b-q4.gguf", "q8": "3b-q8.gguf"}
NPLS = [1, 2, 4]
REPS = 3
NPP, NTG, CTX = 128, 512, 4096   # same context size for every config
COOLDOWN = 90                    # seconds between runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print the plan only")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN)
    args = ap.parse_args()

    if not args.dry and os.geteuid() != 0:
        sys.exit("Run with sudo:  sudo python3 run_matrix.py")
    for f in list(MODELS.values()) + ["energy_log.py"]:
        if not os.path.exists(f):
            sys.exit(f"Missing file in this folder: {f}")

    runs = [(q, n, r) for q in MODELS for n in NPLS for r in range(1, REPS + 1)]
    random.Random(42).shuffle(runs)

    for i, (q, n, r) in enumerate(runs, 1):
        name = f"3b_{q}_np{n}_r{r}"
        if os.path.exists(f"{name}_raw.csv"):
            print(f"[{i}/{len(runs)}] {name}: already done, skipping")
            continue
        cmd = ["caffeinate", "-di", "python3", "energy_log.py",
               "--name", name, "--tokens", str(NTG * n), "--",
               "llama-batched-bench", "-m", MODELS[q], "-c", str(CTX), "-ngl", "99",
               "-npp", str(NPP), "-ntg", str(NTG), "-npl", str(n)]
        print(f"\n=== run {i}/{len(runs)}: {name} ===", flush=True)
        if args.dry:
            print("  " + " ".join(cmd))
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  WARNING: {name} exited with code {rc}; continuing")
        if i < len(runs):
            print(f"  cooling down {args.cooldown}s...", flush=True)
            time.sleep(args.cooldown)

    print("\nAll done. Results are in results.csv")


if __name__ == "__main__":
    main()

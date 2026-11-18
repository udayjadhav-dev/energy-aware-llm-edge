# Energy-aware LLM inference on a memory-constrained edge device

Preliminary measurements of energy per generated token for Llama-3.2-3B on a MacBook with an Apple M2 chip and 8 GB of unified memory, under 1, 2 and 4 concurrent requests, with 4-bit and 8-bit weights. Power is split into package (CPU + GPU + Neural Engine), DRAM and whole system.

This is a small preliminary study, not a finished benchmark. See Limitations.

## Results

Means of three runs per configuration (spread between repeats under 5% of the mean). The 8-bit, 2-request row leaves out one run (`3b_q8_np2_r2`) whose idle baseline was contaminated.

| Precision | Concurrent requests | Throughput (tokens/s) | Package (J/token) | Package + DRAM (J/token) | Whole system (J/token) |
| --- | --- | --- | --- | --- | --- |
| 4-bit (Q4_K_M) | 1 | 39 | 0.340 | 0.410 | 0.511 |
| 4-bit (Q4_K_M) | 2 | 48 | 0.310 | 0.350 | 0.464 |
| 4-bit (Q4_K_M) | 4 | 85 | 0.147 | 0.173 | 0.237 |
| 8-bit (Q8_0) | 1 | 24 | 0.401 | 0.522 | 0.738 |
| 8-bit (Q8_0) | 2 | 46 | 0.218 | 0.292 | 0.427 |
| 8-bit (Q8_0) | 4 | 85 | 0.121 | 0.160 | 0.236 |

Package = CPU + GPU + Neural Engine. J/token = net energy over the GPU-active window (idle baseline subtracted) divided by generated tokens.

Main observations:

- Going from one to four requests lowers package-plus-DRAM energy per token by 58% (4-bit) and 69% (8-bit), while per-request speed falls (about 39 to about 21 tokens/s for 4-bit).
- DRAM is 12% to 17% of package-plus-DRAM power with 4-bit weights and 23% to 25% with 8-bit weights.
- With one request, 4-bit is clearly more efficient than 8-bit. With four requests the two are equal at whole-system level (0.237 vs 0.236 J/token).

## Repository contents

| File | What it is |
| --- | --- |
| `energy_log.py` | Runs one workload and logs power; appends a summary row to `results.csv` |
| `run_matrix.py` | Runs the full matrix (2 precisions x 3 concurrency levels x 3 repeats) in random order |
| `results.csv` | One summary row per run (18 rows) |
| `*_raw.csv` | Per-second power samples for each run |

## Reproducing

Requirements: a Mac with Apple Silicon (tested on M2, macOS 26.5), [Homebrew](https://brew.sh), and about 6 GB of free disk space for the models.

```
brew install llama.cpp macmon
```

Download two GGUF files from [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) on Hugging Face (`Llama-3.2-3B-Instruct-Q4_K_M.gguf` and `Llama-3.2-3B-Instruct-Q8_0.gguf`) and rename them to `3b-q4.gguf` and `3b-q8.gguf`, in the same folder as the scripts. The model files are not in this repository.

Before running: restart the Mac, close all other apps, plug in the charger, fix screen brightness, and turn off auto-brightness and Low Power Mode. Keep these conditions identical for all runs.

Single run:

```
sudo python3 energy_log.py --name test --tokens 512 -- \
  llama-batched-bench -m 3b-q4.gguf -c 4096 -ngl 99 -npp 128 -ntg 512 -npl 1
```

`--tokens` is the total number of generated tokens (512 times the number of parallel requests).

Full matrix (about 45 to 60 minutes):

```
sudo python3 run_matrix.py
```

## Method

- Workload: `llama-batched-bench` from llama.cpp, 128-token prompt, 512 generated tokens per request, all layers on the GPU.
- Power sources: `powermetrics` (CPU, GPU and Neural Engine power) and `macmon` (package, DRAM and whole-system power, plus memory use).
- A 30 s idle baseline is measured before each run and subtracted. Energy is integrated over the window in which the GPU is active, which excludes model loading.

## Limitations

- One 3B model on one 8 GB device.
- `llama-batched-bench` runs requests in lockstep, which is the best case for batching. It is not a realistic multi-user server workload, and time to first token and tail latency were not measured.
- `powermetrics` and `macmon` read the same Apple sensors, so agreement between them (about 2%) is not independent validation. No external power meter was used.
- The DRAM figure is `macmon`'s estimate and has not been validated externally.
- The active window is found from one-second samples, so each J/token figure has up to about 4% timing uncertainty.
- Memory use peaked at about 7.4 GB of 8 GB and the system swapped up to about 390 MB over the session. Repeats were consistent, but a small effect cannot be ruled out.

## License

MIT

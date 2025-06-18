import subprocess
import random
import time
import itertools
import os
import json
from threading import Thread, Lock
from queue import Queue
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--language', type=str, choices=['english', 'hebrew'], required=True, help='Language to run experiments on')
parser.add_argument('--gpus', type=int, default=3, help='Number of parallel GPUs to use')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=None, help='List of GPU IDs to use (e.g. --gpu_ids 1 2 3)')
args = parser.parse_args()

# Define the search space
LANGUAGES = [args.language]
DROPOUTS = [0.0, 0.1]
RESIDUALS = [False, True]
N_LAYERS = [4, 6, 8]
N_HEADS = [4, 6, 8]
EMBED_SIZES = [128, 192, 256]
MLP_SIZES = [512, 768, 1024]
INITS = ["xavier", "kaiming", "normal"]
OPTIMIZERS = ["adamw", "sgd"]
LEARNING_RATES = [5e-4, 3e-4, 1e-4]
WEIGHT_DECAYS = [0.0, 0.01, 0.05]

# Generate all possible configs, then sample 100
all_configs = [cfg for cfg in itertools.product(
    DROPOUTS, RESIDUALS, N_LAYERS, N_HEADS, EMBED_SIZES, MLP_SIZES, INITS, OPTIMIZERS, LEARNING_RATES, WEIGHT_DECAYS
) if cfg[4] % cfg[3] == 0]  # embed_size % n_heads == 0
random.shuffle(all_configs)
configs = all_configs[:100]

# Adaptive sampling: boost best values from previous run if available
import collections
prev_results_path = f"search_results/final_results_{args.language}.log"
best_config = None
if os.path.exists(prev_results_path):
    # Parse previous best config from summary
    with open(prev_results_path) as f:
        for line in f:
            if line.startswith('{'):
                try:
                    config = json.loads(line.split("|")[0])
                    best_config = config
                    break
                except Exception:
                    continue

# Helper for weighted random sampling
def weighted_sample(options, best_value):
    if best_value is not None and best_value in options:
        weights = [0.7 if o == best_value else 0.3/(len(options)-1) for o in options]
    else:
        weights = None
    return random.choices(options, weights=weights if weights else None, k=1)[0]

# Prepare output dir
os.makedirs("search_results", exist_ok=True)

# Thread-safe log
log_lock = Lock()

# Track results
results = []

# Helper to build command
def build_cmd(cfg, gpu):
    dropout, residuals, n_layers, n_heads, embed, mlp, init, opt, lr, wd = cfg
    args_ = [
        "python", "-u", "main.py",
        "--gpu", str(gpu),
        "--dropout", str(dropout),
        "--n_layers", str(n_layers),
        "--n_heads", str(n_heads),
        "--embed_size", str(embed),
        "--mlp_hidden_size", str(mlp),
        "--init", init,
        "--optimizer", opt,
        "--learning_rate", str(lr),
        "--weight_decay", str(wd)
    ]
    if args.language == "hebrew":
        args_.append("--hebrew")
    if residuals:
        args_.append("--with_residuals")
    args_.append("--search_run")
    args_.extend(["--num_batches_to_train", "10000"])
    return args_

# Worker function
class RunWorker(Thread):
    def __init__(self, queue, gpu_ids, worker_idx):
        super().__init__()
        self.queue = queue
        self.gpu_ids = gpu_ids
        self.worker_idx = worker_idx
    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            idx, cfg = item
            gpu = self.gpu_ids[self.worker_idx % len(self.gpu_ids)]
            run_id = f"run_{idx}_{args.language}"
            out_file = f"search_results/{run_id}.json"
            start = time.time()
            cmd = build_cmd(cfg, gpu)
            print(f"[GPU {gpu}] Starting {run_id}: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Only use proc.communicate() to capture all output
            out, _ = proc.communicate()
            # Write raw output to a debug log file for inspection
            # with open(f"search_results/{run_id}_raw_output.log", "w") as rawlog:
            #     rawlog.write(out)
            end = time.time()
            # Parse output for loss, time, and samples
            loss_log = []
            time_log = []
            samples = []
            last_loss = None
            for line in out.splitlines():
                if "last loss is:" in line:
                    try:
                        loss = float(line.split("last loss is:")[-1].strip())
                        loss_log.append(loss)
                        time_log.append(end - start)
                        last_loss = loss
                    except Exception:
                        pass
            # Extract samples: split by 'Model sample' and take the text after
            sample_splits = out.split('Model sample:')
            for chunk in sample_splits[1:]:  # skip the first split (before first sample)
                sample = chunk.strip()
                if sample:
                    samples.append(sample)
            if not samples:
                samples = ["No sample generated"]
            meta = {
                "config": {
                    "dropout": cfg[0],
                    "residuals": cfg[1],
                    "n_layers": cfg[2],
                    "n_heads": cfg[3],
                    "embed_size": cfg[4],
                    "mlp_hidden_size": cfg[5],
                    "init": cfg[6],
                    "optimizer": cfg[7],
                    "learning_rate": cfg[8],
                    "weight_decay": cfg[9],
                    "language": args.language
                },
                "loss_log": loss_log,
                "time_log": time_log,
                "final_loss": last_loss,
                "total_time": end - start,
                "samples": samples
            }
            with log_lock:
                results.append(meta)
                with open(out_file, "w") as f:
                    json.dump(meta, f, indent=2)
            # Print summary to screen after every process
            print(f"[GPU {gpu}] Finished {run_id} in {end-start:.1f}s, final loss: {last_loss}")
            self.queue.task_done()

# Main launcher
if __name__ == "__main__":
    queue = Queue()
    # Dynamic best config (by loss)
    best_config = None
    best_loss = float('inf')
    job_index = 0  # running index for unique run ids
    # Pre-fill the queue with 3 jobs to start
    for _ in range(args.gpus):
        while True:
            dropout = random.choice(DROPOUTS)
            residuals = random.choice(RESIDUALS)
            n_layers = random.choice(N_LAYERS)
            n_heads = random.choice(N_HEADS)
            embed = random.choice(EMBED_SIZES)
            mlp = random.choice(MLP_SIZES)
            init = random.choice(INITS)
            opt = random.choice(OPTIMIZERS)
            lr = random.choice(LEARNING_RATES)
            wd = random.choice(WEIGHT_DECAYS)
            if embed % n_heads == 0:
                cfg = (dropout, residuals, n_layers, n_heads, embed, mlp, init, opt, lr, wd)
                break
        queue.put((job_index, cfg))
        job_index += 1
    gpu_ids = args.gpu_ids if args.gpu_ids is not None else [1, 2, 3] if args.language == 'english' else [4, 5, 6]
    workers = [RunWorker(queue, gpu_ids, gpu_id) for gpu_id in range(args.gpus)]
    for w in workers:
        w.start()
    # Dynamically add new configs as jobs finish
    total_runs = 100
    while job_index < total_runs:
        # Wait for a result
        while len(results) <= queue.qsize():
            time.sleep(0.1)
        # Update best config
        for r in results:
            if r["final_loss"] is not None and r["final_loss"] < best_loss:
                best_loss = r["final_loss"]
                best_config = r["config"]
        # Sample next config, weighted by best so far
        tries = 0
        while True:
            tries += 1
            dropout = weighted_sample(DROPOUTS, best_config['dropout'] if best_config else None)
            residuals = weighted_sample(RESIDUALS, best_config['residuals'] if best_config else None)
            n_layers = weighted_sample(N_LAYERS, best_config['n_layers'] if best_config else None)
            n_heads = weighted_sample(N_HEADS, best_config['n_heads'] if best_config else None)
            embed = weighted_sample(EMBED_SIZES, best_config['embed_size'] if best_config else None)
            mlp = weighted_sample(MLP_SIZES, best_config['mlp_hidden_size'] if best_config else None)
            init = weighted_sample(INITS, best_config['init'] if best_config else None)
            opt = weighted_sample(OPTIMIZERS, best_config['optimizer'] if best_config else None)
            lr = weighted_sample(LEARNING_RATES, best_config['learning_rate'] if best_config else None)
            wd = weighted_sample(WEIGHT_DECAYS, best_config['weight_decay'] if best_config else None)
            if embed % n_heads == 0:
                cfg = (dropout, residuals, n_layers, n_heads, embed, mlp, init, opt, lr, wd)
                break
            if tries > 1000:
                break
        queue.put((job_index, cfg))
        job_index += 1
    # Now add sentinels for workers to stop
    for _ in range(args.gpus):
        queue.put(None)
    for w in workers:
        w.join()

    # After all runs, aggregate results
    # Normalization for loss and time
    valid_results = [r for r in results if r["final_loss"] is not None]
    if valid_results:
        min_loss = min(r["final_loss"] for r in valid_results)
        max_loss = max(r["final_loss"] for r in valid_results)
        min_time = min(r["total_time"] for r in valid_results)
        max_time = max(r["total_time"] for r in valid_results)
        def norm(val, minv, maxv):
            return 0.0 if maxv == minv else (val - minv) / (maxv - minv)
        alpha = 0.5
        for r in valid_results:
            r["norm_loss"] = norm(r["final_loss"], min_loss, max_loss)
            r["norm_time"] = norm(r["total_time"], min_time, max_time)
            r["combined_score"] = alpha * r["norm_loss"] + (1 - alpha) * r["norm_time"]
    else:
        valid_results = []
    results_path = f"search_results/final_results_{args.language}.log"
    with open(results_path, "w") as f:
        f.write("=== Sorted by total time ===\n")
        sorted_by_time = sorted(results, key=lambda x: x["total_time"])
        for i, meta in enumerate(sorted_by_time):
            f.write(json.dumps(meta["config"]) + f" | time: {meta['total_time']:.1f}s | final_loss: {meta['final_loss']}\n")
            if i == 0 or i == len(sorted_by_time) - 1:
                for s in meta["samples"]:
                    f.write(f"Sample: {s}\n")
            f.write("\n")
        f.write("\n=== Sorted by final loss ===\n")
        sorted_by_loss = sorted(results, key=lambda x: (x["final_loss"] if x["final_loss"] is not None else float('inf')))
        for i, meta in enumerate(sorted_by_loss):
            f.write(json.dumps(meta["config"]) + f" | time: {meta['total_time']:.1f}s | final_loss: {meta['final_loss']}\n")
            if i == 0 or i == len(sorted_by_loss) - 1:
                for s in meta["samples"]:
                    f.write(f"Sample: {s}\n")
            f.write("\n")
        f.write("\n=== Best by normalized combined score (alpha=0.5) ===\n")
        sorted_by_score = sorted(valid_results, key=lambda x: x["combined_score"])
        for i, meta in enumerate(sorted_by_score):
            f.write(json.dumps(meta["config"]) +
                    f" | time: {meta['total_time']:.1f}s | final_loss: {meta['final_loss']} | norm_loss: {meta.get('norm_loss', 'NA'):.3f} | norm_time: {meta.get('norm_time', 'NA'):.3f} | score: {meta.get('combined_score', 'NA'):.3f}\n")
            if i == 0 or i == len(sorted_by_score) - 1:
                for s in meta["samples"]:
                    f.write(f"Sample: {s}\n")
            f.write("\n")
    print(f"Final summary written to {results_path}")

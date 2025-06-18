import sys
import os
import json
import re
from collections import defaultdict

if len(sys.argv) != 2 or sys.argv[1] not in ("english", "hebrew"):
    print("Usage: python analyze_search_results.py <english|hebrew>")
    sys.exit(1)

language = sys.argv[1]
report_path = f"search_results/final_results_{language}.log"

if not os.path.exists(report_path):
    print(f"Report file {report_path} not found.")
    sys.exit(1)

# Read all configs and results from the log file
results = []
seen_configs = set()
with open(report_path) as f:
    for line in f:
        if line.strip().startswith("{") and "score:" in line:
            try:
                config = json.loads(line.split("|")[0])
                config_key = tuple(sorted(config.items()))
                if config_key in seen_configs:
                    continue
                seen_configs.add(config_key)
                m_loss = re.search(r"final_loss: ([\d\.eE+-]+)", line)
                m_time = re.search(r"time: ([\d\.eE+-]+)s", line)
                m_score = re.search(r"score: ([\d\.eE+-]+)", line)
                loss = float(m_loss.group(1)) if m_loss else None
                time = float(m_time.group(1)) if m_time else None
                score = float(m_score.group(1)) if m_score else None
                results.append({"config": config, "loss": loss, "time": time, "score": score, "config_key": config_key})
            except Exception:
                continue

if not results:
    print("No results found in the report file.")
    sys.exit(1)

# Compute rank in the last weighted table (by score, lower is better)
scored = [r for r in results if r["score"] is not None]
scored_sorted = sorted(scored, key=lambda x: x["score"])
score_ranks = {id(r): i+1 for i, r in enumerate(scored_sorted)}

# List of hyperparameters to analyze
param_keys = [
    "dropout", "residuals", "n_layers", "n_heads", "embed_size", "mlp_hidden_size",
    "init", "optimizer", "learning_rate", "weight_decay"
]

print(f"Analysis for language: {language}\n")
for param in param_keys:
    value_stats = defaultdict(lambda: {"count": 0, "loss_sum": 0.0, "time_sum": 0.0, "rank_sum": 0.0})
    for r in results:
        val = r["config"].get(param)
        if val is not None and r["loss"] is not None and r["time"] is not None:
            value_stats[val]["count"] += 1
            value_stats[val]["loss_sum"] += r["loss"]
            value_stats[val]["time_sum"] += r["time"]
            if r["score"] is not None:
                value_stats[val]["rank_sum"] += score_ranks.get(id(r), 0)
    print(f"Parameter: {param}")
    for val, stats in value_stats.items():
        avg_loss = stats["loss_sum"] / stats["count"] if stats["count"] else float('nan')
        avg_time = stats["time_sum"] / stats["count"] if stats["count"] else float('nan')
        avg_rank = stats["rank_sum"] / stats["count"] if stats["count"] else float('nan')
        print(f"  Value: {val:>10} | Tried: {stats['count']:3d} | Avg loss: {avg_loss:.4f} | Avg time: {avg_time:.1f}s | Avg rank: {avg_rank:.2f}")
    print()

# After all parameter stats, print global averages
losses = [r["loss"] for r in results if r["loss"] is not None]
times = [r["time"] for r in results if r["time"] is not None]
ranks = [score_ranks.get(id(r), 0) for r in results if r["score"] is not None]
print("Global averages:")
print(f"  Avg loss: {sum(losses)/len(losses) if losses else float('nan'):.4f}")
print(f"  Avg time: {sum(times)/len(times) if times else float('nan'):.1f}s")
print(f"  Avg rank: {sum(ranks)/len(ranks) if ranks else float('nan'):.2f}")

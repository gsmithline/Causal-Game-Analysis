"""Combine SLURM array job chunks into a single results file."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.iterative_game_analysis.causal_analysis import _aggregate_results, save_results

n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 10
base_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/analysis/causal_bargaining_chunk")

all_raw = []
for i in range(n_chunks):
    pkl_path = Path(f"{base_path}_{i}.pkl")
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        chunk = pickle.load(f)
    all_raw.extend(chunk["raw"])
    config = chunk["config"]

config["n_bootstrap"] = len(all_raw)
print(f"Total bootstrap samples: {len(all_raw)}")

aggregated = _aggregate_results(all_raw, config["strategy_names"], config["solvers"], config["metrics"])
results = {"raw": all_raw, "config": config, "aggregated": aggregated}

output_path = base_path.parent / "causal_bargaining_1000"
save_results(results, output_path)

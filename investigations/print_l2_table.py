"""Print L2 conditional threshold table at 1% entry-mass threshold."""

import pickle
import numpy as np
from pathlib import Path

PKL_PATH = Path("data/analysis/iterative_analysis_results.pkl")

AGENTS = [
    "ppo", "psro", "mappo", "openai_5.2_none", "nfsp",
    "soft", "tough", "walk", "ef1_bargainer", "openai_5.2_low",
]
LABELS = {
    "ppo": "ppo", "psro": "psro", "mappo": "mappo",
    "openai_5.2_none": "5.2:None", "nfsp": "nfsp",
    "soft": "soft", "tough": "tough", "walk": "walk",
    "ef1_bargainer": "EF1 Bargainer", "openai_5.2_low": "5.2:Low",
}

METRICS = [
    ("uw",       "delta_eco",      "delta UW"),
    ("nw",       "delta_eco",      "delta NW"),
    ("nw_plus",  "delta_eco",      "delta NW+"),
    ("ef1",      "ef1_lift",       "delta EF1 (pp)"),
    ("ef1_plus", "ef1_plus_lift",  "delta EF1+ (pp)"),
]

THRESHOLD = 0.01  # 1%


def compute_conditional(raw, agent, metric_key, metric_source):
    entry_masses = [r["l2"][agent]["entry_mass"] for r in raw]
    mask = [em > THRESHOLD for em in entry_masses]
    n = sum(mask)
    if n < 1:
        return None

    if metric_source == "delta_eco":
        vals = [r["l2"][agent]["delta_eco"][metric_key]
                for r, m in zip(raw, mask) if m]
    else:
        vals = [r["l2"][agent][metric_source] * 100
                for r, m in zip(raw, mask) if m]

    a = np.array(vals)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(np.mean(a)), float(np.std(a)), float(lo), float(hi), n


def fmt(result):
    if result is None:
        return "N/A"
    mean, std, lo, hi = result[0], result[1], result[2], result[3]
    return f"{mean:+.4f} +/- {std:.4f}  [{lo:.4f}, {hi:.4f}]"


def main():
    with open(PKL_PATH, "rb") as f:
        results = pickle.load(f)
    raw = results["raw"]
    print(f"Loaded {len(raw)} bootstrap samples, threshold = {THRESHOLD*100:.0f}%\n")

    # Header
    header = "| **Strategy** | " + " | ".join(f"**{m[2]}**" for m in METRICS) + " |"
    sep = "|" + "|".join(["---"] * (len(METRICS) + 1)) + "|"
    print(header)
    print(sep)

    for agent in AGENTS:
        row = [f"| {LABELS[agent]} "]
        for metric_key, metric_source, _ in METRICS:
            result = compute_conditional(raw, agent, metric_key, metric_source)
            row.append(f" {fmt(result)} ")
        print("|".join(row) + "|")


if __name__ == "__main__":
    main()

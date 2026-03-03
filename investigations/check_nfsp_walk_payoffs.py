"""Check the actual payoff matrix entries for {nfsp, walk} restricted game."""

import pickle
import numpy as np
from pathlib import Path

results_path = Path("data/analysis/iterative_analysis_results.pkl")
with open(results_path, "rb") as f:
    results = pickle.load(f)

raw_samples = results["raw"]
policies = ["walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro", "openai_5.2_none"]
idx = {p: i for i, p in enumerate(policies)}

i_nfsp = idx["nfsp"]
i_walk = idx["walk"]

print("=== Payoff matrix entries for {nfsp, walk} across bootstrap samples ===\n")
print(f"{'Sample':>7s}  {'M[nfsp,nfsp]':>13s} {'M[nfsp,walk]':>13s} {'M[walk,nfsp]':>13s} {'M[walk,walk]':>13s}")
print("-" * 70)

pay_nn = []
pay_nw = []
pay_wn = []
pay_ww = []

for b_idx, sample in enumerate(raw_samples[:10]):  # first 10 samples
    matrices = sample.get("matrices", None)
    if matrices is None:
        # Try to find the payoff matrix from l2 or elsewhere
        print(f"  Sample {b_idx}: no 'matrices' key. Keys: {list(sample.keys())}")
        continue

    M = matrices["payoff"]
    nn = M[i_nfsp, i_nfsp]
    nw = M[i_nfsp, i_walk]
    wn = M[i_walk, i_nfsp]
    ww = M[i_walk, i_walk]

    pay_nn.append(nn)
    pay_nw.append(nw)
    pay_wn.append(wn)
    pay_ww.append(ww)

    print(f"  {b_idx:>5d}  {nn:>13.4f} {nw:>13.4f} {wn:>13.4f} {ww:>13.4f}")

# Check all samples
print("\n\n=== Summary across ALL bootstrap samples ===\n")
pay_nn_all = []
pay_nw_all = []
pay_wn_all = []
pay_ww_all = []

for sample in raw_samples:
    matrices = sample.get("matrices", None)
    if matrices is None:
        continue
    M = matrices["payoff"]
    pay_nn_all.append(M[i_nfsp, i_nfsp])
    pay_nw_all.append(M[i_nfsp, i_walk])
    pay_wn_all.append(M[i_walk, i_nfsp])
    pay_ww_all.append(M[i_walk, i_walk])

if pay_nn_all:
    print(f"M[nfsp,nfsp]: mean={np.mean(pay_nn_all):.4f}, min={np.min(pay_nn_all):.4f}, max={np.max(pay_nn_all):.4f}")
    print(f"M[nfsp,walk]: mean={np.mean(pay_nw_all):.4f}, min={np.min(pay_nw_all):.4f}, max={np.max(pay_nw_all):.4f}")
    print(f"M[walk,nfsp]: mean={np.mean(pay_wn_all):.4f}, min={np.min(pay_wn_all):.4f}, max={np.max(pay_wn_all):.4f}")
    print(f"M[walk,walk]: mean={np.mean(pay_ww_all):.4f}, min={np.min(pay_ww_all):.4f}, max={np.max(pay_ww_all):.4f}")
    print(f"\nnfsp weakly dominates walk: M[nfsp,nfsp] > M[walk,nfsp]?  {np.mean(pay_nn_all):.4f} > {np.mean(pay_wn_all):.4f}")
    print(f"M[nfsp,walk] == M[walk,walk]?  {np.mean(pay_nw_all):.4f} ~= {np.mean(pay_ww_all):.4f}")

    # How many samples does nfsp dominate walk?
    nn = np.array(pay_nn_all)
    wn = np.array(pay_wn_all)
    nw_arr = np.array(pay_nw_all)
    ww = np.array(pay_ww_all)

    dominates = (nn > wn) & (nw_arr >= ww)
    print(f"\nnfsp strictly dominates walk in {np.sum(nn > wn)}/100 samples (M[nfsp,nfsp] > M[walk,nfsp])")
    print(f"nfsp weakly dominates walk in {np.sum(dominates)}/100 samples")

    # But does WALK dominate NFSP??
    walk_dom = (wn > nn) & (ww >= nw_arr)
    print(f"\nwalk strictly dominates nfsp?  M[walk,nfsp] > M[nfsp,nfsp] in {np.sum(wn > nn)}/100 samples")
    print(f"walk weakly dominates nfsp in {np.sum(walk_dom)}/100 samples")

    # Check: is M[nfsp,nfsp] < M[walk,nfsp] (i.e., walk BEATS nfsp head to head)?
    print(f"\n=== The real question: who gets higher payoff when facing nfsp? ===")
    print(f"M[nfsp,nfsp] (nfsp's payoff in self-play): {np.mean(pay_nn_all):.4f}")
    print(f"M[walk,nfsp] (walk's payoff vs nfsp):      {np.mean(pay_wn_all):.4f}")
    print(f"Difference (nfsp - walk when facing nfsp):  {np.mean(nn - wn):.4f}")

    print(f"\n=== Who gets higher payoff when facing walk? ===")
    print(f"M[nfsp,walk] (nfsp's payoff vs walk):       {np.mean(pay_nw_all):.4f}")
    print(f"M[walk,walk] (walk's payoff vs walk):        {np.mean(pay_ww_all):.4f}")
    print(f"Difference (nfsp - walk when facing walk):  {np.mean(nw_arr - ww):.4f}")
else:
    print("No matrices found in raw samples. Let me check the data structure.")
    print(f"Keys in first sample: {list(raw_samples[0].keys())}")
    for k, v in raw_samples[0].items():
        if isinstance(v, dict):
            print(f"  {k}: {list(v.keys())[:10]}")

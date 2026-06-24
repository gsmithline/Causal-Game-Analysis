# Plan: CURB Set Contribution Figures

## What we're building
Two figures showing welfare deltas (CURB set restricted-game equilibrium minus full-game equilibrium):

### Figure 1: CURB Set Contribution (rows = CURB sets)
- Horizontal bar chart, one panel per metric (UW, NW, NW+, EF1, EF1+)
- Each row = a CURB set (top ~15-20 by size/interest, sorted by delta)
- Bar length = welfare delta (restricted MENE − full game MENE)
- Color: red for positive delta (restricted game is better), blue for negative
- Error bars from bootstrap CIs if available
- Re-evaluating paper style: clean horizontal bars with labels

### Figure 2: Strategy Contribution (rows = strategies)
- Horizontal stacked bar chart, one panel per metric
- Each row = a strategy
- Stacked segments = sum of deltas across CURB sets containing that strategy (weighted by frequency or count)
- Shows which strategies are "responsible" for welfare gains/losses when games restrict
- Colors: one color per CURB set, or positive/negative coloring

## Data source
- `data/analysis/curb_results.pkl` — already has:
  - Per-CURB-set MENE welfare metrics (point estimate)
  - Full game included as a CURB set (frozenset of all 10 indices)
  - 77 total CURB sets with pre-computed equilibria
  - Bootstrap per-sample results for CIs

## Implementation steps

1. **Create `visuals/curb_contribution.py`** (~150 lines)
   - Load `curb_results.pkl`
   - Extract full-game metrics as baseline
   - Compute delta per CURB set per metric
   - Filter to interesting CURB sets (exclude full game, singletons optional)
   - Figure 1: horizontal bar chart (one panel per metric, rows = CURB sets)
   - Figure 2: horizontal stacked bar chart (rows = strategies, segments = CURB sets)
   - Save to `data/analysis/figures/`

## Conventions (matching existing codebase)
- SHORT_MAP for strategy names
- Red (#d62728) positive, blue (#1f77b4) negative
- `_restricted_game_label()` for CURB set names
- axvline at 0, light grid, capsize=3 error bars
- figsize ~(24, 5.5) for 5-panel layout

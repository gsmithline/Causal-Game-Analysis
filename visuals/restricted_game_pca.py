"""PCA analysis of restricted games (CURB sets) and bootstrap samples.

Two panels:
  A) Bootstrap regime PCA — each of 1000 bootstraps is a point in
     477-dim existence space.  Colored by PPO/PSRO attractor regime.
  B) Welfare profile PCA — each of 477 restricted games is a point in
     5-dim mean-welfare space.  Colored by minimality/size category.

Reads ecology_matrix.pkl produced by evaluation/curb_ecology_matrix.py.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

SHORT_MAP = {
    'openai_5.2_low': '5.2:L',
    'openai_5.2_none': '5.2:N',
    'openai_5.2_medium': '5.2:M',
    'ef1_bargainer': 'EF1',
    'psro': 'PSRO',
    'ppo': 'PPO',
    'mappo': 'MAPPO',
    'nfsp': 'NFSP',
    'walk': 'Walk',
    'soft': 'Soft',
    'tough': 'Tough',
}


def _short_label(eco, strategy_names):
    """Short label for a restricted game."""
    names = sorted(
        [SHORT_MAP.get(strategy_names[i], strategy_names[i]) for i in eco]
    )
    if len(names) <= 4:
        return '{' + ', '.join(names) + '}'
    return '{' + ', '.join(names[:3]) + ', +' + str(len(names) - 3) + '}'


def _categorize(eco, minimal_flags, strategy_names):
    """Assign each restricted game to a display category.

    Categories:
      - 'minimal_singleton': minimal CURB set with 1 strategy (attractor)
      - 'minimal': minimal CURB set with >1 strategy
      - 'full_game': the set of all strategies
      - 'small' : |set| <= 3, non-minimal
      - 'medium': 4 <= |set| <= 6, non-minimal
      - 'large' : |set| >= 7, non-minimal
    """
    is_min = minimal_flags.get(eco, 0) >= 0.5
    n_strats = len(strategy_names)

    if len(eco) == n_strats:
        return 'full_game'
    if is_min and len(eco) == 1:
        return 'minimal_singleton'
    if is_min:
        return 'minimal'
    if len(eco) <= 3:
        return 'small'
    if len(eco) <= 6:
        return 'medium'
    return 'large'


# Visual style per category
CATEGORY_STYLE = {
    'minimal_singleton': dict(color='#d62728', marker='D', size=60, zorder=5, alpha=0.95),
    'minimal':           dict(color='#e6550d', marker='s', size=45, zorder=4, alpha=0.90),
    'full_game':         dict(color='#2ca02c', marker='*', size=120, zorder=5, alpha=0.95),
    'small':             dict(color='#aec7e8', marker='o', size=20, zorder=2, alpha=0.50),
    'medium':            dict(color='#6baed6', marker='o', size=25, zorder=2, alpha=0.50),
    'large':             dict(color='#2171b5', marker='o', size=30, zorder=2, alpha=0.50),
}

CATEGORY_LABELS = {
    'minimal_singleton': 'Minimal singleton (attractor)',
    'minimal':           'Minimal (multi-strategy)',
    'full_game':         'Full game',
    'small':             'Non-minimal (2–3 strategies)',
    'medium':            'Non-minimal (4–6 strategies)',
    'large':             'Non-minimal (7–9 strategies)',
}


def _draw_ellipse(ax, points, color, alpha=0.15):
    """Draw a 1-std-dev ellipse around a cluster of 2D points."""
    if len(points) < 3:
        return
    mean = points.mean(axis=0)
    cov = np.cov(points.T)
    # Eigendecomposition for ellipse orientation
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(vals)  # 1-std-dev
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor=color, edgecolor=color, alpha=alpha,
                  linewidth=1.5, linestyle='--')
    ax.add_patch(ell)


def _should_label(eco, cat, freq, minimal_flags):
    """Decide which restricted games get text labels.

    Label minimals (freq >= 2%), full game, and singleton attractors.
    """
    if cat == 'full_game':
        return True
    if cat == 'minimal_singleton' and freq >= 0.02:
        return True
    if cat == 'minimal' and freq >= 0.02:
        return True
    return False


def _load_data(ecology_matrix_path):
    """Load ecology matrix pkl and return parsed data dict."""
    with open(ecology_matrix_path, 'rb') as f:
        return pickle.load(f)


# -----------------------------------------------------------------------
# Panel A: Bootstrap regime PCA
# -----------------------------------------------------------------------

REGIME_STYLE = {
    'Both PPO + PSRO':  dict(color='#7570b3', marker='o', size=12, alpha=0.45),
    'Only PSRO':        dict(color='#1b9e77', marker='o', size=12, alpha=0.45),
    'Only PPO':         dict(color='#d95f02', marker='o', size=14, alpha=0.55),
    'Neither':          dict(color='#e7298a', marker='D', size=18, alpha=0.70),
}


def _bootstrap_regime(existence, ecologies, strategy_names):
    """Classify each bootstrap into an attractor regime.

    Returns (regime_labels, regime_order) where regime_labels[j] is the
    regime string for bootstrap j.
    """
    ppo_idx = strategy_names.index('ppo')
    psro_idx = strategy_names.index('psro')
    ppo_eco = next(i for i, e in enumerate(ecologies) if e == frozenset({ppo_idx}))
    psro_eco = next(i for i, e in enumerate(ecologies) if e == frozenset({psro_idx}))

    ppo_exists = existence[ppo_eco]    # (B,)
    psro_exists = existence[psro_eco]  # (B,)

    labels = []
    for j in range(existence.shape[1]):
        has_ppo = ppo_exists[j] > 0.5
        has_psro = psro_exists[j] > 0.5
        if has_ppo and has_psro:
            labels.append('Both PPO + PSRO')
        elif has_psro:
            labels.append('Only PSRO')
        elif has_ppo:
            labels.append('Only PPO')
        else:
            labels.append('Neither')
    return labels


def _plot_bootstrap_pca(ax, data):
    """Panel A: PCA on bootstrap samples in existence space."""
    ecologies = data['ecologies']
    matrices = data['matrices']
    strategy_names = data['config']['strategy_names']
    n_bootstraps = data['config']['n_bootstraps']

    uw_mat = matrices['uw']
    existence = (~np.isnan(uw_mat)).astype(np.float64)  # (E, B)

    # Transpose: each bootstrap is a point in E-dimensional space
    X = existence.T  # (B, E) = (1000, 477)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)  # (1000, 2)

    regimes = _bootstrap_regime(existence, ecologies, strategy_names)

    # Number of CURB sets per bootstrap (for size encoding)
    n_curb = existence.sum(axis=0)  # (B,)

    # Draw by regime
    regime_order = ['Both PPO + PSRO', 'Only PSRO', 'Only PPO', 'Neither']
    for regime in regime_order:
        style = REGIME_STYLE[regime]
        mask = np.array([r == regime for r in regimes])
        if not mask.any():
            continue

        # Size proportional to number of CURB sets
        sizes = 8 + (n_curb[mask] - n_curb.min()) / max(n_curb.max() - n_curb.min(), 1) * 25
        count = mask.sum()

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=style['color'], marker=style['marker'], s=sizes,
            alpha=style['alpha'], edgecolors='none',
            label=f'{regime} (n={count})',
            zorder=3 if regime == 'Neither' else 2,
        )

    # Ellipses per regime
    for regime in regime_order:
        mask = np.array([r == regime for r in regimes])
        pts = coords[mask]
        if len(pts) >= 5:
            _draw_ellipse(ax, pts, REGIME_STYLE[regime]['color'], alpha=0.10)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=10)
    ax.set_title('A. Bootstrap Samples in CURB-Existence Space',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=7, framealpha=0.9, loc='best', markerscale=1.5)
    ax.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    return pca


# -----------------------------------------------------------------------
# Panel B: Welfare profile PCA
# -----------------------------------------------------------------------

METRIC_NAMES = ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus']
METRIC_DISPLAY = {'uw': 'UW', 'nw': 'NW', 'nw_plus': 'NW+',
                  'ef1': 'EF1', 'ef1_plus': 'EF1+'}


def _plot_welfare_pca(ax, data):
    """Panel B: PCA on restricted games in mean-welfare space."""
    ecologies = data['ecologies']
    matrices = data['matrices']
    minimal_flags = data['minimal_flags']
    strategy_names = data['config']['strategy_names']

    # Build (E, 5) mean welfare matrix
    n_eco = len(ecologies)
    welfare_means = np.zeros((n_eco, len(METRIC_NAMES)))
    for j, m in enumerate(METRIC_NAMES):
        welfare_means[:, j] = np.nanmean(matrices[m], axis=1)

    # Standardize each metric to zero-mean unit-variance before PCA
    mu = welfare_means.mean(axis=0)
    sigma = welfare_means.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    X = (welfare_means - mu) / sigma

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)  # (E, 2)

    # Categorize
    categories = [_categorize(eco, minimal_flags, strategy_names)
                  for eco in ecologies]
    labels = [_short_label(eco, strategy_names) for eco in ecologies]
    freqs = (~np.isnan(matrices['uw'])).astype(float).sum(axis=1) / 1000

    # Scatter by category
    cat_order = ['small', 'medium', 'large', 'minimal', 'minimal_singleton', 'full_game']
    for cat in cat_order:
        style = CATEGORY_STYLE[cat]
        mask = np.array([c == cat for c in categories])
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=style['color'], marker=style['marker'], s=style['size'],
            zorder=style['zorder'], alpha=style['alpha'],
            edgecolors='white', linewidths=0.3,
            label=CATEGORY_LABELS[cat],
        )

    # Label minimals and full game
    label_items = []
    for i, (eco, cat, freq) in enumerate(zip(ecologies, categories, freqs)):
        if not _should_label(eco, cat, freq, minimal_flags):
            continue
        display = f'{labels[i]}'
        label_items.append((i, display, cat, coords[i, 0], coords[i, 1]))

    label_items.sort(key=lambda t: -freqs[t[0]])
    placed = []
    for idx, text, cat, x, y in label_items:
        offset_x, offset_y = 6, 8
        for px, py in placed:
            if abs(x - px) < 0.15 * (coords[:, 0].max() - coords[:, 0].min()) and \
               abs(y - py) < 0.15 * (coords[:, 1].max() - coords[:, 1].min()):
                offset_y += 10

        ax.annotate(
            text, (x, y),
            textcoords='offset points', xytext=(offset_x, offset_y),
            fontsize=6, fontweight='bold',
            color=CATEGORY_STYLE[cat]['color'],
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.4),
            zorder=6,
        )
        placed.append((x, y))

    # Draw PCA loading arrows (biplot)
    loadings = pca.components_.T  # (5, 2)
    # Scale loadings to fit nicely on the plot
    coord_range = max(coords[:, 0].max() - coords[:, 0].min(),
                      coords[:, 1].max() - coords[:, 1].min())
    load_scale = coord_range * 0.3 / max(np.abs(loadings).max(), 1e-10)

    for j, m in enumerate(METRIC_NAMES):
        lx, ly = loadings[j] * load_scale
        ax.annotate(
            '', xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
            zorder=7,
        )
        ax.text(lx * 1.12, ly * 1.12, METRIC_DISPLAY[m],
                fontsize=8, fontweight='bold', color='#333333',
                ha='center', va='center', zorder=7)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=10)
    ax.set_title('B. Restricted Games in Welfare Space',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=6.5, framealpha=0.9, loc='best', markerscale=1.2)
    ax.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    return pca


# -----------------------------------------------------------------------
# Combined figure
# -----------------------------------------------------------------------

def create_pca_figure(
    ecology_matrix_path,
    filename='restricted_game_pca',
    save_dir=None,
    dpi=200,
):
    """Create 2-panel PCA figure: bootstrap regimes + welfare profiles."""

    data = _load_data(ecology_matrix_path)

    fig, (ax_boot, ax_welf) = plt.subplots(1, 2, figsize=(18, 7.5))

    pca_boot = _plot_bootstrap_pca(ax_boot, data)
    pca_welf = _plot_welfare_pca(ax_welf, data)

    fig.suptitle(
        'PCA of Bootstrap Regimes and Welfare Profiles',
        fontsize=14, fontweight='bold', y=1.01,
    )
    fig.tight_layout()

    out_path = Path(save_dir or '.') / (filename + '.png')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    boot_var = sum(pca_boot.explained_variance_ratio_[:2]) * 100
    welf_var = sum(pca_welf.explained_variance_ratio_[:2]) * 100
    print(f'Saved: {out_path}')
    print(f'  Panel A (bootstrap):  PC1+PC2 = {boot_var:.1f}% variance')
    print(f'  Panel B (welfare):    PC1+PC2 = {welf_var:.1f}% variance')


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent / 'data' / 'analysis'
    create_pca_figure(
        str(data_dir / 'ecology_matrix.pkl'),
        save_dir=str(data_dir),
    )

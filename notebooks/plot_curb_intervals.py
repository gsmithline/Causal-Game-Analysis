import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Raw data from the table
metrics = ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus']
metric_labels = {'uw': 'UW', 'nw': 'NW', 'nw_plus': 'NW+', 'ef1': 'EF1', 'ef1_plus': 'EF1+'}

agents = ['mappo', 'ppo', 'psro', 'openai_5.2_low', 'openai_5.2_medium',
          'openai_5.4_low', 'openai_5.2_none', 'openai_5.4_medium']

display = {
    'mappo': 'MAPPO', 'ppo': 'PPO', 'psro': 'PSRO',
    'openai_5.2_low': '5.2 Low', 'openai_5.2_medium': '5.2 Medium',
    'openai_5.4_low': '5.4 Low', 'openai_5.2_none': '5.2 None',
    'openai_5.4_medium': '5.4 Medium',
}

# Data: {metric: {agent: (min_mean, min_lo, min_hi, max_mean, max_lo, max_hi)}}
data = {
    'uw': {
        'mappo':             (-6.2557, -41.9325, +0.2727,  +0.7254, -30.2641, +24.9167),
        'ppo':               (-4.4151, -40.2942, +14.2578, +2.2931, -26.6938, +25.4064),
        'psro':              (-15.2433, -47.7871, +15.1644, +1.9499, -39.1909, +51.5933),
        'openai_5.2_low':    (-4.0155, -37.6804, +24.2357, +7.3636, -25.5817, +40.8062),
        'openai_5.2_medium': (-0.7537, -34.2296, +25.1833, +2.6083, -13.0731, +36.8075),
        'openai_5.4_low':    (-0.2795, -11.8373, +8.4496,  +0.7398, -0.2509, +19.8676),
        'openai_5.2_none':   (+4.7747, -0.9598, +40.3591,  +6.6845, -0.0138, +43.3009),
        'openai_5.4_medium': (+9.8204, -17.2911, +45.1872, +18.2397, -5.3787, +52.8432),
    },
    'nw': {
        'mappo':             (-3.6960, -26.2508, +0.4045,  +0.3877, -17.7593, +17.1059),
        'ppo':               (-2.5117, -24.2863, +9.7670,  +1.4991, -15.8088, +16.3199),
        'psro':              (-10.0452, -30.2457, +9.0107,  +0.2943, -25.2166, +31.5646),
        'openai_5.2_low':    (-2.5846, -24.1508, +16.2044, +4.3999, -16.4451, +25.5476),
        'openai_5.2_medium': (-0.4654, -21.8102, +15.2308, +1.5513, -8.0208, +23.1981),
        'openai_5.4_low':    (-0.1331, -7.1061, +6.0725,   +0.4615, -0.0010, +10.8123),
        'openai_5.2_none':   (+2.8423, -1.8448, +24.5330,  +4.0029, -0.2109, +26.6965),
        'openai_5.4_medium': (+6.0643, -10.4588, +27.9933, +10.9704, -3.6737, +32.5360),
    },
    'nw_plus': {
        'mappo':             (-2.5267, -18.8207, +0.5761,  +0.4766, -9.2616, +9.9444),
        'ppo':               (-1.8336, -15.0124, +4.4771,  +0.7084, -9.2926, +7.4426),
        'psro':              (-6.3683, -20.8052, +4.8478,   +0.2018, -12.9945, +16.1270),
        'openai_5.2_low':    (-1.6948, -13.3310, +8.0771,  +2.1815, -8.7839, +14.2337),
        'openai_5.2_medium': (-0.4952, -12.3618, +8.1508,  +0.6895, -5.5554, +12.0788),
        'openai_5.4_low':    (-0.0902, -4.2410, +2.5840,   +0.2587, -0.0051, +6.1796),
        'openai_5.2_none':   (+1.3134, -3.4825, +12.7978,  +2.0738, -1.0745, +14.2172),
        'openai_5.4_medium': (+3.1333, -5.5298, +14.5378,  +6.2016, -0.8539, +19.7293),
    },
    'ef1': {
        'mappo':             (-0.0380, -0.2315, +0.0000,  -0.0014, -0.1148, +0.1036),
        'ppo':               (-0.0216, -0.1677, +0.0634,  +0.0084, -0.1067, +0.0993),
        'psro':              (-0.0675, -0.2038, +0.0685,  +0.0106, -0.1536, +0.2232),
        'openai_5.2_low':    (-0.0250, -0.1699, +0.0823,  +0.0254, -0.1017, +0.1663),
        'openai_5.2_medium': (-0.0030, -0.1318, +0.1053,  +0.0111, -0.0495, +0.1475),
        'openai_5.4_low':    (-0.0009, -0.0474, +0.0339,  +0.0034, -0.0000, +0.0835),
        'openai_5.2_none':   (+0.0171, -0.0317, +0.1534,  +0.0263, -0.0060, +0.1760),
        'openai_5.4_medium': (+0.0334, -0.0838, +0.1760,  +0.0714, -0.0181, +0.2331),
    },
    'ef1_plus': {
        'mappo':             (-0.0410, -0.3055, +0.0070,  +0.0024, -0.1722, +0.1655),
        'ppo':               (-0.0353, -0.2675, +0.0691,  +0.0038, -0.1736, +0.1317),
        'psro':              (-0.1068, -0.2931, +0.0799,  -0.0034, -0.2489, +0.3022),
        'openai_5.2_low':    (-0.0314, -0.2482, +0.1396,  +0.0376, -0.1588, +0.2544),
        'openai_5.2_medium': (-0.0028, -0.1953, +0.1684,  +0.0171, -0.0783, +0.2340),
        'openai_5.4_low':    (-0.0011, -0.0720, +0.0524,  +0.0045, -0.0000, +0.1304),
        'openai_5.2_none':   (+0.0250, -0.0686, +0.2344,  +0.0375, -0.0163, +0.2505),
        'openai_5.4_medium': (+0.0563, -0.1056, +0.2585,  +0.1013, -0.0155, +0.3085),
    },
}

# Sort agents by max delta of UW
agents_sorted = sorted(agents, key=lambda s: data['uw'][s][3])
y = np.arange(len(agents_sorted))

fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 7), sharey=True)

for ax, m in zip(axes, metrics):
    for i, s in enumerate(agents_sorted):
        min_mean, min_lo, min_hi, max_mean, max_lo, max_hi = data[m][s]

        # Min delta: red square + error bar
        ax.errorbar(min_mean, y[i] - 0.15,
                     xerr=[[min_mean - min_lo], [min_hi - min_mean]],
                     fmt='s', color='firebrick', markersize=5, capsize=3,
                     elinewidth=1.5, zorder=3)

        # Max delta: green diamond + error bar
        ax.errorbar(max_mean, y[i] + 0.15,
                     xerr=[[max_mean - max_lo], [max_hi - max_mean]],
                     fmt='D', color='forestgreen', markersize=5, capsize=3,
                     elinewidth=1.5, zorder=3)

        # Shaded region between means
        ax.fill_betweenx([y[i] - 0.08, y[i] + 0.08], min_mean, max_mean,
                          color='steelblue', alpha=0.15, zorder=1)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(metric_labels[m], fontweight='bold')
    ax.grid(axis='x', alpha=0.2)
    ax.set_xlabel('Δ Metric')

axes[0].set_yticks(y)
axes[0].set_yticklabels([display[s] for s in agents_sorted], fontsize=9)


legend_elements = [
    Line2D([0], [0], marker='s', color='firebrick', label='Min Δ mean ± 95% CI',
           markersize=6, linestyle='None'),
    Line2D([0], [0], marker='D', color='forestgreen', label='Max Δ mean ± 95% CI',
           markersize=6, linestyle='None'),
]
axes[-1].legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.suptitle('CURB-Conditional LOO Intervals', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('curb_loo_intervals_correct.png', dpi=150, bbox_inches='tight')
plt.show()

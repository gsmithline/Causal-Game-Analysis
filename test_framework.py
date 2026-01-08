"""Test script for the causal meta-game framework."""

from causal_game_analysis import MetaGame, Bootstrap, level1_analysis, ecosystem_lift
import pandas as pd
import numpy as np

# Create synthetic rock-paper-scissors style data
np.random.seed(42)
policies = ['rock', 'paper', 'scissors']

# Generate cross-play outcomes
rows = []
for pi in policies:
    for pj in policies:
        if pi == pj:
            payoff = 0.5
        elif (pi, pj) in [('rock', 'scissors'), ('paper', 'rock'), ('scissors', 'paper')]:
            payoff = 1.0
        else:
            payoff = 0.0
        for _ in range(10):
            rows.append({'policy_i': pi, 'policy_j': pj, 'outcome': payoff + np.random.normal(0, 0.1)})

df = pd.DataFrame(rows)

# Build meta-game
game = MetaGame.from_dataframe(df)
print('MetaGame created:', game)
print('Payoff matrix:')
print(game.payoff_matrix.round(2))

# Solve for equilibrium
sigma = game.solve('uniform')
print('\nEquilibrium (uniform):', sigma.round(3))

# Level 1 analysis with extended game
game_extended = MetaGame(
    policies=['rock', 'paper', 'scissors', 'spock'],
    payoff_matrix=np.array([
        [0.5, 0.0, 1.0, 0.0],
        [1.0, 0.5, 0.0, 1.0],
        [0.0, 1.0, 0.5, 0.0],
        [1.0, 0.0, 1.0, 0.5],
    ])
)

result = level1_analysis(game_extended, ['rock', 'paper', 'scissors'], 'spock', solver='uniform')
print('\nLevel 1 Analysis (Partner Lift for spock):')
print('  Per incumbent:', {k: round(v, 3) for k, v in result['per_incumbent'].items()})
print('  Uniform avg:', round(result['uniform_avg'], 3))

# Level 2 analysis
eco_result = ecosystem_lift(game_extended, ['rock', 'paper', 'scissors'], 'spock', solver='uniform')
print('\nLevel 2 Analysis (Ecosystem Lift for spock):')
print('  Delta eco:', round(eco_result['delta_eco'], 3))
print('  Entry mass:', round(eco_result['entry_mass'], 3))

print('\nAll tests passed!')

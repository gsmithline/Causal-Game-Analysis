
import numpy as np
from scipy.optimize import linprog, minimize
import warnings
import scipy.sparse as sp
from scipy.special import entr
import cvxpy as cp
import nashpy as nash
import multiprocessing

EPSILON = 1e-6
def _simplex_projection(x):
    """
    Project onto probability simplex.
    """
    x = np.asarray(x)
    x = x.reshape(-1)
    if (x >= 0).all() and abs(np.sum(x) - 1) < 1e-10:
        return x
    
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u * np.arange(1, n+1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(x - theta, 0)


def compute_regret(strategy, payoff_matrix):
    
    expected_utils = payoff_matrix @ strategy
    
    nash_value = strategy.T @ payoff_matrix @ strategy
    
    regret = expected_utils - nash_value
    
    return regret, nash_value, expected_utils

def milp_max_sym_ent_2p(game_matrix, discrete_factors=100):
    """
    Compute maximum entropy Nash equilibrium for 2-player games following the Zun's implementation.
    Uses CVXPY with ECOS_BB solver to handle boolean variables and mixed integer constraints.
    
    Args:
        game_matrix: numpy array of payoffs
        discrete_factors: number of discrete factors for entropy approximation
        
    Returns:
        nash_strategy: equilibrium strategy (probability distribution over actions)
        
    Raises:
        RuntimeError: If both ECOS_BB and GLPK_MI solvers fail to find an optimal solution
    """
    shape = game_matrix.shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    
    game_matrix_np = np.array(game_matrix, dtype=np.float64)
    
    if np.isnan(game_matrix_np).any():
        for j in range(game_matrix_np.shape[1]):
            col = game_matrix_np[:, j]
            if np.isnan(col).any():
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0
                game_matrix_np[np.isnan(col), j] = col_mean
    
    M = shape[0]
    U = np.max(game_matrix_np) - np.min(game_matrix_np)
    
    
    x = cp.Variable(M)
    u = cp.Variable(1)
    z = cp.Variable(M)
    b = cp.Variable(M, boolean=True)
    #r = cp.Variable(1, nonneg=True)
    
    #objective is minimize sum of z (which approximates-entropy)
    obj = cp.Minimize(cp.sum(z))
    
    #nash equilibrium constraints
    a_mat = np.ones(M).reshape((1, M))
    u_m = game_matrix_np @ x
    
    constraints = [
        u_m <= u + EPSILON, 
        a_mat @ x == 1, 
        x >= 0,
        u - u_m <= U * b,
        x <= 1 - b,
        #r <= 1,
        #r >= u - u_m
    ]
    
    for k in range(discrete_factors):
        if k == 0:
            constraints.append(np.log(1/discrete_factors) * x <= z)
        else:
            #linear approximation of entropy at k/discrete_factors
            slope = ((k+1)*np.log((k+1)/discrete_factors) - k*np.log(k/discrete_factors))
            intercept = k/discrete_factors * np.log(k/discrete_factors)
            constraints.append(intercept + slope * (x - k/discrete_factors) <= z)
    
    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver='ECOS_BB')
        if prob.status != 'optimal':
            raise ValueError(f"ECOS_BB solver failed with status: {prob.status}")
    except Exception as e:
        warnings.warn(f"Failed to solve with ECOS_BB: {e}")
        try:
            prob.solve(solver='GLPK_MI')
            if prob.status != 'optimal':
                raise ValueError(f"GLPK_MI solver failed with status: {prob.status}")
        except Exception as e:
            raise RuntimeError(f"Both ECOS_BB and GLPK_MI solvers failed. ECOS_BB error: {e}")
    
    ne_strategy = _simplex_projection(x.value.reshape(-1))

    regret, nash_value, expected_utils = compute_regret(ne_strategy, game_matrix_np)
    max_regret = np.max(regret)
    if max_regret <= EPSILON:
        return ne_strategy
    else:
        raise RuntimeError(f"Failed to find Nash equilibrium within {EPSILON} regret")
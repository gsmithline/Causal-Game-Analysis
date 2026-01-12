from dataclasses import dataclass
from typing import Optional
import torch as th

@dataclass
class VTraceReturns:
    """Container for V-Trace returns.
    Attributes:
        vs (th.Tensor): V-Trace value function estimates.
        pg_advantages (th.Tensor): Policy gradient advantages.
        log_rhos (Optional[th.Tensor]): Log importance sampling weights.
    """
    vs: th.Tensor
    pg_advantages: th.Tensor
    log_rhos: Optional[th.Tensor] = None
def compute_vtrace(
        behavior_log_probs: th.Tensor,
        target_log_probs: th.Tensor,
        rewards: th.Tensor,
        values: th.Tensor,
        bootstrap_value: th.Tensor,
        discounts: th.Tensor,
        clip_rho_threshold: Optional[float] = 1.0,
        clip_c_threshold: Optional[float] = 1.0
) -> VTraceReturns:
    """
    Compute V-Trace returns (IMPALA) from precomputed log-probabilities.
    Inputs
    Args:
        behavior_log_probs (Tensor):
            Log-prob of the taken action under the behavior
            (actor) policy used to generate data. Shape [T, B].
            If the action is multi-dimensional/factorized, sum per-dimension
            log-probs to one scalar per step.
        target_log_probs (Tensor):
            Log-prob of the same taken action under the
            current learner (target) policy. Shape [T, B].
        rewards (Tensor):
            r_t. Per-step rewards aligned with transitions. Shape [T, B], float.
        values (Tensor):
            V(s_t). Critic predictions for times t = 0..T-1 (no bootstrap here).
            Shape [T, B], float.
        bootstrap_value (Tensor):
            V(s_T). Critic prediction for the state following the last step in
            the unroll (used to bootstrap). Shape [B].
        discounts (Tensor):
            γ_t * (1 - done_t). Discount factor per step with terminals masked
            to zero. Shape [T, B]. If you use a fixed γ, set
            discounts[t, b] = γ * (1 - done_t,b).
        clip_rho_threshold (float | None, default=1.0):
            Clip cap for importance ratios used in the policy and value updates.
            Conventionally  ≥ 1. Set None to disable clipping.
        clip_c_threshold (float | None, default=1.0):
            c̄. Clip cap for trace coefficients controlling backward credit flow.
            Usually its ≤ 1. Set None to disable clipping.
    Shape Notes:
    - All tensors are time-major [T, B]. If you store [B, T], transpose first.
    - Terminations are handled by setting discounts[t] = 0 at terminal steps.
    - Importance weights are computed from (target_log_probs - behavior_log_probs)
      and gradients are typically stopped through those weights.
    Returns
    VTraceReturns:
        vs (Tensor): V-trace value targets, shape [T, B].
        pg_advantages (Tensor): Policy-gradient advantages, shape [T, B].
        log_rhos (Tensor): log(π/μ) per step (diagnostics), shape [T, B].
    From
    Espeholt et al., “IMPALA: Scalable Distributed Deep-RL with Importance
    Weighted Actor-Learner Architectures”
    """
    #actor policy
    behavior_log_probs = behavior_log_probs.detach()
    #gradient of target and actor
    log_rhos = target_log_probs - behavior_log_probs
    rhos = th.exp(log_rhos)
    #clamp gradients through importance sampling weights
    clipped_rhos = th.clamp(rhos, max=clip_rho_threshold) if clip_rho_threshold is not None else rhos
    cs = th.clamp(rhos, max=clip_c_threshold) if clip_c_threshold is not None else rhos
    clipped_rhos = clipped_rhos.detach()
    cs = cs.detach()
    #broadcast bootstraps
    bootstrap_value = bootstrap_value.unsqueeze(0)
    while bootstrap_value.ndim < values.ndim:
        bootstrap_value = bootstrap_value.unsqueeze(-1)
    values_tp1 = th.cat([values[1:], bootstrap_value], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_tp1 - values)
    acc = th.zeros_like(bootstrap_value.squeeze(0))
    vs = th.zeros_like(values)
    for t in range(values.shape[0] - 1, -1, -1):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs[t] = values[t] + acc
    vs_tp1 = th.cat([vs[1:], bootstrap_value], dim=0)
    #policy gradient advantage
    adv = (rewards + discounts * vs_tp1 - values).detach()
    pg_advantages = clipped_rhos * adv
    return VTraceReturns(vs=vs, pg_advantages=pg_advantages, log_rhos=log_rhos.detach())
"""
Aspiration-based concession schedule agent for the 7-4-1 bargaining game.

Heuristic negotiator that follows a predefined concession schedule.
The schedule is defined by an aspiration function A(t_k) combined with BATNA
protection, discounted utilities, and a mirroring offer construction rule.

Reference: LLM Meta-Game paper (aspiration agent).
Aspiration schedule based on NegMas.
"""

import random

import numpy as np
import torch
import torch.nn as nn

QUANTITIES = np.array([7, 4, 1], dtype=int)
NUM_ITEM_TYPES = 3
NUM_ACTIONS = 82
ACTION_ACCEPT = 80
ACTION_WALK = 81
# Action encoding: n0*10 + n1*2 + n2 (items given to opponent for P1, items kept for P2)
ENCODING_BASES = np.array([10, 2, 1], dtype=int)


class AspirationAgent:
    """
    Concession schedule negotiator for the 7-4-1 bargaining game.

    Uses an aspiration curve (polynomial, exponential, or geometric) to
    determine target utility each round, then constructs offers via DP
    with optional mirroring of the opponent's last offer.
    """

    def __init__(
        self,
        max_rounds=3,
        aspiration_family="poly",
        aspiration_profile="linear",
        max_aspiration=0.85,
        aspiration_margin=0.02,
        use_discount_warp=False,
        discount_factor=1.0,
        acceptance_slack=0.0,
        initial_aspiration_scale=None,
    ):
        self.max_rounds = max_rounds

        self.aspiration_family = aspiration_family.lower()
        self.aspiration_profile = aspiration_profile.lower()
        self.max_aspiration = max_aspiration
        self.aspiration_margin = max(0.0, aspiration_margin)
        self.use_discount_warp = use_discount_warp
        self.default_discount = discount_factor
        self.acceptance_slack = max(0.0, acceptance_slack)
        self.initial_aspiration_scale = (
            None if initial_aspiration_scale is None
            else float(np.clip(initial_aspiration_scale, 0.0, 1.0))
        )

        self.poly_presets = {"boulware": 4.0, "linear": 1.0, "conceder": 0.25}
        self.exp_presets = {"boulware": 0.125, "linear": 0.725, "conceder": 4.0}

    # ------------------------------------------------------------------ #
    #  Observation decoding (92-dim obs from CUDA bargain env)
    # ------------------------------------------------------------------ #

    def _decode_obs(self, obs):
        """Parse the 92-dim observation into game state."""
        values = np.array([obs[i] * 100.0 for i in range(3)])
        max_possible = float(np.dot(values, QUANTITIES))
        outside_offer = obs[3] * max_possible if max_possible > 0 else 0.0

        offer_valid = obs[7] > 0.5
        my_items = None
        opp_offer_to_me = None
        if offer_valid:
            my_items = np.array(
                [int(round(obs[4 + i] * QUANTITIES[i])) for i in range(NUM_ITEM_TYPES)],
                dtype=int,
            )
            opp_offer_to_me = my_items  # items I would receive if I accept

        round_num = int(round(obs[8] * 2))  # 0-indexed round mapped from normalized
        current_player = int(round(obs[9]))
        action_mask = np.array([obs[10 + i] > 0.5 for i in range(NUM_ACTIONS)])

        return {
            "values": values,
            "outside_offer": outside_offer,
            "offer_valid": offer_valid,
            "opp_offer_to_me": opp_offer_to_me,
            "round": round_num,
            "current_player": current_player,
            "action_mask": action_mask,
        }

    # ------------------------------------------------------------------ #
    #  Action encoding / decoding
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_action(give_vector, current_player):
        """Encode a give-to-opponent vector (P1) or keep vector (P2) to action id.

        P1 actions encode what P1 gives to P2: action = g0*10 + g1*2 + g2
        P2 actions encode what P2 keeps: action = k0*10 + k1*2 + k2
        """
        if current_player == 0:
            vec = give_vector
        else:
            vec = QUANTITIES - give_vector  # P2 encodes items kept
        action = int(vec[0]) * 10 + int(vec[1]) * 2 + int(vec[2])
        if action < 0 or action >= 80:
            return None
        return action

    @staticmethod
    def _decode_action(action_id):
        """Decode action id to (n0, n1, n2) encoding."""
        if action_id < 0 or action_id >= 80:
            return None
        n0 = action_id // 10
        n1 = (action_id % 10) // 2
        n2 = action_id % 2
        return np.array([n0, n1, n2], dtype=int)

    # ------------------------------------------------------------------ #
    #  Utility helpers
    # ------------------------------------------------------------------ #

    def _total_value(self, values):
        return float(np.dot(QUANTITIES, values))

    def _keep_utility(self, keep_vector, values):
        if keep_vector is None:
            return -np.inf
        return float(np.dot(keep_vector, values))

    def _receive_utility(self, receive_vector, values):
        """Utility of receiving receive_vector items."""
        if receive_vector is None:
            return -np.inf
        return float(np.dot(receive_vector, values))

    # ------------------------------------------------------------------ #
    #  Discount / time helpers
    # ------------------------------------------------------------------ #

    def _discount_power(self, round_index, discount_factor):
        return discount_factor ** round_index

    def _time_fraction(self, round_index, discount_factor):
        if self.max_rounds <= 1:
            return 0.0
        if self.use_discount_warp and discount_factor not in (None, 0.0, 1.0):
            denom = 1.0 - (discount_factor ** self.max_rounds)
            if abs(denom) < 1e-9:
                return round_index / self.max_rounds
            return (1.0 - (discount_factor ** round_index)) / denom
        return round_index / self.max_rounds

    # ------------------------------------------------------------------ #
    #  Aspiration schedule
    # ------------------------------------------------------------------ #

    def _schedule_exponent(self, discount_factor):
        presets = self.exp_presets if self.aspiration_family == "exp" else self.poly_presets
        # Honor explicit profile when it matches a known preset; otherwise
        # fall back to the discount-tiered heuristic.
        if self.aspiration_profile in presets:
            return presets[self.aspiration_profile]
        gamma = discount_factor
        if gamma <= 0.92:
            tier = "conceder"
        elif gamma <= 0.96:
            tier = "linear"
        else:
            tier = "boulware"
        return presets.get(tier, 1.0)

    def _initial_scale(self, discount_factor):
        if self.initial_aspiration_scale is not None:
            return self.initial_aspiration_scale
        gamma = discount_factor
        if gamma <= 0.92:
            return 0.15
        if gamma <= 0.96:
            return 0.25
        return 0.4

    def _base_curve_value(self, t_k, exponent, discount_factor):
        t_k = float(np.clip(t_k, 0.0, 1.0))
        if self.aspiration_family == "exp":
            numerator = np.exp((1.0 - t_k) ** exponent) - 1.0
            denom = np.e - 1.0
            base = numerator / denom if denom > 0 else 0.0
        elif self.aspiration_family == "geometric":
            base = discount_factor ** t_k
        else:  # poly
            base = 1.0 - (t_k ** exponent)
        return float(np.clip(base, 0.0, 1.0))

    def _threshold_schedule(self, values, outside_offer, discount_factor):
        total_value = self._total_value(values)
        if total_value <= 0:
            total_value = 1.0

        batna = max(0.0, outside_offer)
        normalized_batna = np.clip(batna / total_value, 0.0, 1.0)
        cap_fraction = np.clip(self.max_aspiration, 0.0, 1.0)
        headroom = max(0.0, total_value - batna)
        raw_cap = batna + cap_fraction * headroom
        effective_cap = np.clip(raw_cap / total_value, normalized_batna, 1.0) if total_value > 0 else normalized_batna
        margin_cap = np.clip(normalized_batna + self.aspiration_margin, normalized_batna, 1.0)
        effective_cap = max(effective_cap, margin_cap)
        effective_cap = min(effective_cap, 1.0 - 1e-6)

        scale = np.clip(self._initial_scale(discount_factor), 0.0, 1.0)
        starting_fraction = normalized_batna + scale * (effective_cap - normalized_batna)
        exponent = self._schedule_exponent(discount_factor)

        thresholds = []
        for k in range(1, self.max_rounds + 1):
            t_k = self._time_fraction(k - 1, discount_factor)
            base_value = self._base_curve_value(t_k, exponent, discount_factor)
            normalized_target = normalized_batna + (starting_fraction - normalized_batna) * base_value
            thresholds.append(normalized_target * total_value)

        return thresholds

    # ------------------------------------------------------------------ #
    #  Offer construction (DP with mirroring)
    # ------------------------------------------------------------------ #

    def _dp_best_keep(self, target_utility, values, desired_give=None):
        """Find keep vector achieving >= target_utility, closest to desired_give via DP/DFS."""
        best_keep = None
        best_distance = np.inf
        best_value = None
        keep = np.zeros(NUM_ITEM_TYPES, dtype=int)

        def dfs(idx, current_value):
            nonlocal best_keep, best_distance, best_value
            if idx == NUM_ITEM_TYPES:
                if current_value + 1e-9 < target_utility:
                    return
                if desired_give is not None:
                    give_vec = QUANTITIES - keep
                    distance = float(np.sum(np.abs(give_vec - desired_give)))
                    is_better = (
                        distance + 1e-9 < best_distance
                        or (abs(distance - best_distance) < 1e-9
                            and (best_value is None or current_value > best_value + 1e-9))
                    )
                else:
                    distance = float(current_value - target_utility)
                    is_better = (
                        distance + 1e-9 < best_distance
                        or (abs(distance - best_distance) < 1e-9
                            and (best_value is None or current_value + 1e-9 < best_value))
                    )
                if is_better:
                    best_keep = keep.copy()
                    best_distance = distance
                    best_value = current_value
                return

            quantity = int(QUANTITIES[idx])
            if desired_give is not None:
                desired_keep = int(np.clip(quantity - desired_give[idx], 0, quantity))
                options = sorted(range(quantity + 1), key=lambda cnt: abs(cnt - desired_keep))
            else:
                options = range(quantity, -1, -1)

            for count in options:
                keep[idx] = count
                dfs(idx + 1, current_value + count * values[idx])
                if best_distance == 0.0:
                    break
            keep[idx] = 0

        dfs(0, 0.0)
        return best_keep

    def _select_counter_action(self, target_utility, values, current_player, action_mask,
                               desired_give=None, desired_keep=None):
        """Find the best legal counteroffer action meeting the target utility."""
        candidates = []

        for action_id in range(80):
            if not action_mask[action_id]:
                continue

            encoded = self._decode_action(action_id)
            if encoded is None:
                continue

            if current_player == 0:
                give_vector = encoded
                keep_vector = QUANTITIES - give_vector
            else:
                keep_vector = encoded
                give_vector = QUANTITIES - keep_vector

            if np.any(keep_vector < 0) or np.any(give_vector < 0):
                continue

            utility = self._keep_utility(keep_vector, values)
            if utility + 1e-9 < target_utility:
                continue

            give_sum = int(np.sum(give_vector))
            if give_sum == 0:
                continue  # drop trivial "keep everything" offers

            if desired_give is not None:
                distance = float(np.sum(np.abs(give_vector - desired_give)))
            elif desired_keep is not None:
                distance = float(np.sum(np.abs(keep_vector - desired_keep)))
            else:
                distance = utility - target_utility

            utility_gap = max(0.0, utility - target_utility)
            candidates.append((distance, utility_gap, -utility, action_id, keep_vector))

        if not candidates:
            return None

        candidates.sort()
        best_dist = candidates[0][0]
        best_gap = candidates[0][1]
        best_subset = [
            c for c in candidates
            if abs(c[0] - best_dist) < 1e-9 and abs(c[1] - best_gap) < 1e-9
        ]
        chosen = random.choice(best_subset)
        return chosen[3]  # action_id

    # ------------------------------------------------------------------ #
    #  Main act method
    # ------------------------------------------------------------------ #

    def act(self, obs):
        """Select an action given the 92-dim observation vector."""
        p = self._decode_obs(obs)
        values = p["values"]
        outside_offer = p["outside_offer"]
        offer_valid = p["offer_valid"]
        opp_offer_to_me = p["opp_offer_to_me"]
        round_num = p["round"]
        current_player = p["current_player"]
        action_mask = p["action_mask"]
        discount_factor = self.default_discount

        thresholds = self._threshold_schedule(values, outside_offer, discount_factor)
        round_index = max(0, min(round_num, self.max_rounds - 1))

        target_index = round_index
        if current_player == 0:
            target_index = min(round_index + 1, len(thresholds) - 1)

        discount_power = self._discount_power(round_index, discount_factor)
        round_target = thresholds[target_index]
        slack_raw = self.acceptance_slack * round_target
        target_raw = max(0.0, round_target - slack_raw)
        batna_raw = max(0.0, outside_offer)
        target_raw = max(target_raw, batna_raw)

        discounted_target = discount_power * target_raw
        discounted_batna = discount_power * batna_raw

        # --- Accept logic ---
        if offer_valid and opp_offer_to_me is not None:
            receive_utility = self._receive_utility(opp_offer_to_me, values)
            discounted_offer = discount_power * receive_utility
            min_accept = max(discounted_target, discounted_batna)
            if discounted_offer + 1e-9 >= min_accept and action_mask[ACTION_ACCEPT]:
                return ACTION_ACCEPT

        # --- Counter-offer logic ---
        required_utility = max(0.0, target_raw)
        required_utility = min(required_utility, self._total_value(values))

        desired_give = None
        if opp_offer_to_me is not None:
            # Mirror: give opponent what they wanted to keep
            opponent_keep = QUANTITIES - opp_offer_to_me
            desired_give = np.clip(opponent_keep, 0, None).astype(int)

        desired_keep = self._dp_best_keep(required_utility, values, desired_give)
        if (
            desired_give is None
            and desired_keep is not None
            and np.all(QUANTITIES - desired_keep == 0)
        ):
            # Avoid proposing "keep everything" when first-moving: shave the
            # lowest-value item by one unit if still feasible.
            adjusted = desired_keep.copy()
            order = sorted(range(NUM_ITEM_TYPES), key=lambda i: (values[i], i))
            for idx in order:
                if adjusted[idx] <= 0:
                    continue
                adjusted[idx] -= 1
                if self._keep_utility(adjusted, values) + 1e-9 >= required_utility:
                    desired_keep = adjusted.astype(int)
                    break
                adjusted[idx] += 1

        counter = self._select_counter_action(
            required_utility, values, current_player, action_mask,
            desired_give=desired_give, desired_keep=desired_keep,
        )
        if counter is not None:
            return counter

        # --- Walk if BATNA dominates ---
        if offer_valid and opp_offer_to_me is not None:
            receive_utility = self._receive_utility(opp_offer_to_me, values)
            discounted_offer = discount_power * receive_utility
            if discounted_batna + 1e-9 >= discounted_target and discounted_batna + 1e-9 >= discounted_offer:
                if action_mask[ACTION_WALK]:
                    return ACTION_WALK

        # --- Fallbacks ---
        if action_mask[ACTION_WALK]:
            return ACTION_WALK
        if action_mask[ACTION_ACCEPT]:
            return ACTION_ACCEPT

        # Last resort: first legal action
        for i in range(NUM_ACTIONS):
            if action_mask[i]:
                return i
        return 0

    def act_batched(self, obs_batch):
        return [self.act(obs_batch[i]) for i in range(len(obs_batch))]


class AspirationPolicy(nn.Module):
    """PyTorch wrapper for AspirationAgent, compatible with get_policy_action."""

    def __init__(self, **agent_kwargs):
        super().__init__()
        self.agent = AspirationAgent(**agent_kwargs)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, obs, action_mask):
        batch_size = obs.shape[0]
        logits = torch.full((batch_size, NUM_ACTIONS), -1e9)
        value = torch.zeros(batch_size)

        for i in range(batch_size):
            obs_np = obs[i].cpu().numpy()
            action = self.agent.act(obs_np)
            if action_mask[i, action] > 0.5:
                logits[i, action] = 0.0
            else:
                # Fallback: uniform over legal actions
                logits[i] = action_mask[i].float() * 0.0 + (1 - action_mask[i].float()) * -1e9
        return logits, value

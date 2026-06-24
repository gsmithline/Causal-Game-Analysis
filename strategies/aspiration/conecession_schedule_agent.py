import numpy as np
import random
from utils.offer import Offer
from prompts.make_prompt import make_prompt


class ConcessionScheduleNegotiator:
    """
    Heuristic negotiator that follows a predefined concession schedule.
    The schedule is defined by an aspiration function A(t_k) combined with BATNA
    protection, discounted utilities, and a mirroring offer construction rule. 
    """

    def __init__(
        self,
        game,
        valuation_vector,
        num_items,
        player_id,
        max_rounds=3,
        aspiration_family: str = "poly",
        aspiration_profile: str = "linear",
        max_aspiration: float = .85,
        aspiration_margin: float = 0.02,
        use_discount_warp: bool = False,
        discount_factor: float = .98,
        acceptance_slack: float = 0.0,
        initial_aspiration_scale: float | None = None,
    ):
        self.game = game
        self.valuation_vector = np.array(valuation_vector, dtype=float)
        if self.valuation_vector.shape != (num_items,):
            raise ValueError(
                f"Valuation vector shape mismatch. Expected ({num_items},), got {self.valuation_vector.shape}"
            )

        self.num_items = num_items
        self.player_id = player_id
        self.player_num = player_id
        self.num_actions = self.game.num_distinct_actions()
        self.max_rounds = max_rounds

        self.action = "CONCESSION_SCHEDULE"
        self.max_quantity_for_encoding = 10
        self.walk_away_value = None
        self.total_quantities = None
        self.current_counter_offer_give_format = None
        self.prompt = None

        self.history_cache = {0: [], 1: []}
        self.current_opponent_offer_for_us = None
        self.last_offer_keep_format = None
        self.last_offer = None
        self.current_round = 1
        self.discount_factor = discount_factor
        self.opponent_last_offer = None   

        
         
        self.aspiration_family = aspiration_family.lower()
        self.aspiration_profile = aspiration_profile.lower()
        self.max_aspiration = max_aspiration
        self.aspiration_margin = max(0.0, aspiration_margin)
        self.use_discount_warp = use_discount_warp
        self.acceptance_slack = max(0.0, acceptance_slack)
        self.initial_aspiration_scale = (
            None if initial_aspiration_scale is None else float(np.clip(initial_aspiration_scale, 0.0, 1.0))
        )

        self.poly_presets = {"boulware": 4.0, "linear": 1.0, "conceder": 0.25}
        self.exp_presets = {"boulware": 0.125, "linear": 0.725, "conceder": 4.0}

    def process_observation(self, observation, num_items=5):
        """
        Translate the OpenSpiel observation tensor into the structures used by our agents.
        This mirrors the helper used by other heuristic negotiators.
        """
        current_player_idx = self.player_id
        is_response_state = observation[3] == 0.0 if len(observation) > 3 else False

        round_number = int(observation[6]) + 1 if len(observation) > 6 else 1
        discount_factor = observation[7] if len(observation) > 7 else 0.9

        item_quantities = (
            observation[9 : 9 + num_items] if len(observation) > 9 + num_items else [0] * num_items
        )
        self.items = item_quantities

        player_values_start = 9 + num_items
        player_values_end = player_values_start + num_items
        player_values = (
            observation[player_values_start:player_values_end]
            if len(observation) > player_values_end
            else [0] * num_items
        )

        walk_away_value_idx = 9 + 2 * num_items
        walk_away_value = observation[walk_away_value_idx] if len(observation) > walk_away_value_idx else 0

        history_start_idx = 9 + 2 * num_items + 1

        total_utility = np.dot(player_values, item_quantities)

        self.valuation_vector = np.array(player_values, dtype=float)
        self.walk_away_value = walk_away_value
        self.total_quantities = np.array(item_quantities, dtype=int)

        max_history_entries = (len(observation) - history_start_idx) // num_items
        history = {0: [], 1: []}
        current_offer = None

        for i in range(max_history_entries):
            start_idx = history_start_idx + i * num_items
            if start_idx + num_items > len(observation):
                continue
            raw_entry = observation[start_idx : start_idx + num_items]
            if all(x == -1 for x in raw_entry):
                continue

            player = i % 2
            keep_for_self = np.array(raw_entry, dtype=int)
            give_to_other = np.maximum(0, self.total_quantities - keep_for_self)
            offer_obj = Offer(player=player, offer=give_to_other.tolist())
            history[player].append(offer_obj)

            if player != current_player_idx and is_response_state:
                current_offer = offer_obj

        if not history[0] and not history[1] and round_number > 1:
            if self.last_offer:
                history[self.player_num].append(self.last_offer)

        self.prompt = make_prompt(
            T=num_items,
            quantities=list(map(int, item_quantities)),
            V=101,
            values=list(map(int, player_values)),
            W1=int(total_utility),
            W2=1,
            w=int(walk_away_value),
            R=max_history_entries // 2 if max_history_entries > 1 else 0,
            g=discount_factor,
            r=round_number,
            history=history,
            current_offer=current_offer,
            player_num=current_player_idx,
            p1_outside_offer=[1, int(total_utility) if current_player_idx == 0 else 0],
            p2_outside_offer=[1, int(total_utility) if current_player_idx == 1 else 0],
            circle=0,
        )

        self.history_cache = history
        self.current_opponent_offer_for_us = (
            current_offer.offer if current_offer and current_offer.player != self.player_id else None
        )
        if current_offer and current_offer.player != self.player_id:
            self.opponent_last_offer = current_offer.offer
        self.current_round = round_number
        self.discount_factor = discount_factor

    def _decode_action_to_proposal(self, action_id):
        encoding_base = self.max_quantity_for_encoding + 1
        max_possible_encoded_offer = (encoding_base ** self.num_items) - 1
        if action_id < 0 or action_id > max_possible_encoded_offer:
            return None

        proposal = np.zeros(self.num_items, dtype=int)
        temp_value = action_id
        for i in range(self.num_items - 1, -1, -1):
            power = encoding_base ** i
            digit = temp_value // power
            if digit > self.max_quantity_for_encoding:
                return None
            proposal[self.num_items - 1 - i] = int(digit)
            temp_value %= power
        return proposal

    def _encode_proposal_to_action(self, proposal_keep_format):
        if proposal_keep_format is None or len(proposal_keep_format) != self.num_items:
            return None
        if np.any(proposal_keep_format < 0) or np.any(proposal_keep_format > self.max_quantity_for_encoding):
            return None

        encoding_base = self.max_quantity_for_encoding + 1
        action_id = 0
        for i in range(self.num_items):
            action_id += proposal_keep_format[i] * (encoding_base ** (self.num_items - 1 - i))

        max_possible_encoded_offer = (encoding_base ** self.num_items) - 1
        if action_id < 0 or action_id > max_possible_encoded_offer:
            return None
        return int(action_id)

    def _total_value(self):
        if self.total_quantities is None:
            return 0.0
        return float(np.dot(self.total_quantities, self.valuation_vector))

    def _calculate_keep_utility(self, keep_vector):
        if keep_vector is None or len(keep_vector) != self.num_items:
            return -np.inf
        return float(np.dot(keep_vector, self.valuation_vector))

    def _calculate_offer_value(self, offer_vector):
        if offer_vector is None or self.total_quantities is None:
            return -np.inf
        keep_vector = self.total_quantities - np.array(offer_vector, dtype=float)
        if np.any(keep_vector < 0):
            return -np.inf
        return self._calculate_keep_utility(keep_vector)

    def _calculate_receive_utility(self, receive_vector):
        if receive_vector is None:
            return -np.inf
        if len(receive_vector) != self.num_items:
            return -np.inf
        if self.total_quantities is not None and np.any(np.array(receive_vector, dtype=float) > self.total_quantities + 1e-9):
            return -np.inf
        return float(np.dot(receive_vector, self.valuation_vector))

    def _discount_factor_power(self, round_index):
        gamma = float(self.discount_factor if self.discount_factor is not None else 0.9)
        return gamma ** round_index

    def _time_fraction(self, round_index):
        '''
        aspiration schedule taken from NegMas 
        '''
        if self.max_rounds <= 1:
            return 0.0
        if self.use_discount_warp and self.discount_factor not in (None, 0.0, 1.0):
            gamma = self.discount_factor
            denom = 1.0 - (gamma ** (self.max_rounds - 1))
            if abs(denom) < 1e-9:
                return round_index / (self.max_rounds - 1)
            return (1.0 - (gamma ** round_index)) / denom
        return round_index / (self.max_rounds - 1)

    def _schedule_exponent(self):
        gamma = float(self.discount_factor if self.discount_factor is not None else 0.9)
        if gamma <= 0.92:
            tier = "conceder"
        elif gamma <= 0.96:
            tier = "linear"
        else:
            tier = "boulware" 
        presets = self.exp_presets if self.aspiration_family == "exp" else self.poly_presets
        return presets.get(tier, 1.0) 

    def _initial_scale(self):
        if self.initial_aspiration_scale is not None:
            return self.initial_aspiration_scale
        gamma = float(self.discount_factor if self.discount_factor is not None else 0.9)
        if gamma <= 0.92:
            return 0.15
        if gamma <= 0.96:
            return 0.25
        return 0.4

    def _base_curve_value(self, t_k, exponent):
        t_k = float(np.clip(t_k, 0.0, 1.0))
        if self.aspiration_family == "exp":
            numerator = np.exp((1.0 - t_k) ** exponent) - 1.0
            denom = np.e - 1.0
            base = numerator / denom if denom > 0 else 0.0
        elif self.aspiration_family == "geometric":
            gamma = float(self.discount_factor if self.discount_factor is not None else 0.9)
            # 
            # gamma = np.clip(gamma, 1e-6, 0.999999 if gamma < 1 else gamma)
            base = gamma ** t_k
        else:
            base = 1.0 - (t_k**exponent)
        return float(np.clip(base, 0.0, 1.0))

    def _threshold_schedule(self):
        total_value = self._total_value()
        if total_value <= 0:
            total_value = 1.0

        batna = max(0.0, float(self.walk_away_value or 0.0))
        normalized_batna = np.clip(batna / total_value, 0.0, 1.0)
        cap_fraction = np.clip(self.max_aspiration, 0.0, 1.0)
        headroom = max(0.0, total_value - batna)
        raw_cap = batna + cap_fraction * headroom
        if total_value > 0.0:
            effective_cap = np.clip(raw_cap / total_value, normalized_batna, 1.0)
        else:
            effective_cap = normalized_batna
        margin_cap = np.clip(normalized_batna + self.aspiration_margin, normalized_batna, 1.0)
        effective_cap = max(effective_cap, margin_cap)
        effective_cap = min(effective_cap, 1.0 - 1e-6)
        scale = np.clip(self._initial_scale(), 0.0, 1.0)
        starting_fraction = normalized_batna + scale * (effective_cap - normalized_batna)
        exponent = self._schedule_exponent()

        thresholds = []
        for k in range(1, self.max_rounds + 1):
            t_k = self._time_fraction(k - 1)
            base_value = self._base_curve_value(t_k, exponent)
            normalized_target = normalized_batna + (starting_fraction - normalized_batna) * base_value
            thresholds.append(normalized_target * total_value)
        print(f"Schedule for concession: {thresholds}")
        print(f"BATNA: {self.walk_away_value}")
        print(f"Max value {np.dot([7, 4, 1], self.valuation_vector)}")
        
        
        return thresholds

    def _greedy_keep_with_mirroring(self, target_utility, desired_keep):
        if self.total_quantities is None:
            return None

        target = float(max(target_utility, 0.0))
        desired_keep_vec = (
            np.clip(np.array(desired_keep, dtype=int), 0, None) if desired_keep is not None else None
        )
        desired_keep_vec = (
            (self.total_quantities - desired_keep).astype(int)
            if desired_keep is not None and self.total_quantities is not None
            else None
        )

        keep = np.zeros(self.num_items, dtype=int)
        remaining = target

        sorted_indices = sorted(
            range(self.num_items),
            key=lambda idx: (-self.valuation_vector[idx], idx),
        )

        for idx in sorted_indices:
            value = self.valuation_vector[idx]
            available = int(self.total_quantities[idx])
            if available <= 0 or value <= 0:
                continue

            preferred = desired_keep_vec[idx] if desired_keep_vec is not None else available
            preferred = int(np.clip(preferred, 0, available))

            options = list(range(preferred, -1, -1)) + list(range(preferred + 1, available + 1))

            best_local = None
            best_distance = np.inf
            best_value = -np.inf

            for count in options:
                contribution = count * value
                if remaining - contribution > (self._total_value() - contribution):
                    continue

                new_remaining = max(0.0, remaining - contribution)
                distance = abs(count - preferred)
                if distance + 1e-9 < best_distance or (
                    abs(distance - best_distance) < 1e-9 and contribution > best_value + 1e-9
                ):
                    best_distance = distance
                    best_value = contribution
                    best_local = count

                if best_distance == 0:
                    break

            if best_local is None:
                best_local = preferred

            keep[idx] = best_local
            remaining -= best_local * value
            if remaining <= 0:
                break

        if remaining > 0:
            keep = self.total_quantities.copy()

        return keep.astype(int)

    def _dp_keep_with_mirroring(self, target_utility, desired_give):
        if self.total_quantities is None or self.valuation_vector is None:
            return None

        target = float(max(target_utility, 0.0))
        desired_give_vec = (
            np.clip(np.array(desired_give, dtype=int), 0, None) if desired_give is not None else None
        )
        num_items = self.num_items
        keep = np.zeros(num_items, dtype=int)
        best_keep = None
        best_distance = np.inf
        best_value = None
        total_quantities = self.total_quantities.astype(int)
        values = self.valuation_vector.astype(float)

        def dfs(idx, current_value):
            nonlocal best_keep, best_distance, best_value
            if idx == num_items:
                if current_value + 1e-9 < target:
                    return
                if desired_give_vec is not None:
                    give_vec = total_quantities - keep
                    distance = float(np.sum(np.abs(give_vec - desired_give_vec)))
                    is_better = (
                        distance + 1e-9 < best_distance
                        or (
                            abs(distance - best_distance) < 1e-9
                            and (best_value is None or current_value > best_value + 1e-9)
                        )
                    )
                else:
                    distance = float(current_value - target)
                    is_better = (
                        distance + 1e-9 < best_distance
                        or (
                            abs(distance - best_distance) < 1e-9
                            and (best_value is None or current_value + 1e-9 < best_value)
                        )
                    )
                if is_better:
                    best_keep = keep.copy()
                    best_distance = distance
                    best_value = current_value
                return

            quantity = int(total_quantities[idx])
            if desired_give_vec is not None:
                
                desired_keep = np.clip(quantity - desired_give_vec[idx], 0, quantity)
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

    def _build_target_keep(self, target_utility, desired_give):
        keep = self._dp_keep_with_mirroring(target_utility, desired_give)
        if keep is not None:
            return keep.astype(int)
        desired_keep = self.total_quantities - desired_give
        keep = self._greedy_keep_with_mirroring(target_utility, desired_keep)
        if keep is not None:
            return keep.astype(int)
        return np.zeros(self.num_items, dtype=int)

    def _select_counter_action(self, state, required_utility, desired_keep, desired_give):
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None

        candidates = []
        desired_keep = desired_keep.astype(int) if desired_keep is not None else None
        desired_give = np.array(desired_give, dtype=int) if desired_give is not None else None

        for action in legal_actions:
            action_str = state.action_to_string(self.player_id, action)
            if "Agreement" in action_str or "walk away" in action_str.lower() or (
                "get" in action_str and "points" in action_str
            ):
                continue

            keep_vector = self._decode_action_to_proposal(action)
            if keep_vector is None:
                continue
            if self.total_quantities is not None and np.any(keep_vector > self.total_quantities):
                continue

            utility = self._calculate_keep_utility(keep_vector)
            if utility + 1e-9 >= required_utility:
                give_vector = None
                give_sum = 0
                if self.total_quantities is not None:
                    give_vector = (self.total_quantities - keep_vector).astype(int)
                    give_sum = int(np.sum(give_vector))
                if desired_give is not None and give_vector is not None:
                    distance = np.sum(np.abs(give_vector - desired_give))
                elif desired_keep is not None:
                    distance = np.sum(np.abs(keep_vector - desired_keep))
                else:
                    distance = utility - required_utility
                utility_gap = max(0.0, utility - required_utility)
                candidates.append((utility, distance, give_sum, utility_gap, action, keep_vector))

        if not candidates:
            return None

        non_trivial_candidates = [entry for entry in candidates if entry[2] > 0]
        effective_candidates = non_trivial_candidates if non_trivial_candidates else []
        if not effective_candidates:
            return None

        effective_candidates.sort(key=lambda entry: (entry[1], entry[3], -entry[0]))
        best_distance = effective_candidates[0][1]
        best_gap = effective_candidates[0][3]
        best_subset = [
            entry
            for entry in effective_candidates
            if abs(entry[1] - best_distance) < 1e-9 and abs(entry[3] - best_gap) < 1e-9
        ]
        _, _, _, _, chosen_action, chosen_keep = random.choice(best_subset)
        print(f"Best L1 Distance is {best_distance}, utility gap {best_gap}")


        self.last_offer_keep_format = chosen_keep
        self.current_counter_offer_give_format = (self.total_quantities - chosen_keep).astype(int).tolist()
        self.last_offer = Offer(player=self.player_id, offer=self.current_counter_offer_give_format)

        return chosen_action

    def step(self, state):
        self.process_observation(state.observation_tensor(self.player_id), self.num_items)

        if state.current_player() != self.player_id:
            legal_actions = state.legal_actions()
            self.action = "ERROR_WRONG_PLAYER"
            return legal_actions[0] if legal_actions else 0

        thresholds = self._threshold_schedule()
        round_index = max(0, min(self.current_round, self.max_rounds) - 1)
        gamma = float(self.discount_factor if self.discount_factor is not None else 0.9)
        discount_power = self._discount_factor_power(round_index)
        
        target_index = round_index
        if self.player_id == 0:
            target_index = min(round_index + 1, len(thresholds) - 1)
        total_value = self._total_value()
        round_target = thresholds[target_index]
        slack_raw = self.acceptance_slack * round_target
        target_raw = max(0.0, round_target - slack_raw)

        batna_raw = float(self.walk_away_value or 0.0)
        target_raw = max(target_raw, batna_raw)
        discounted_target = discount_power * target_raw
        discounted_batna = discount_power * batna_raw

        self.current_counter_offer_give_format = None

        if self.current_opponent_offer_for_us is not None:
            discounted_offer_value = discount_power * self._calculate_receive_utility(
                self.current_opponent_offer_for_us
            )
            min_accept_threshold = max(discounted_target, discounted_batna)
            if discounted_offer_value + 1e-9 >= min_accept_threshold:
                accept_action = self._create_accept_action(state)
                if accept_action is not None:
                    self.action = "ACCEPT"
                    return accept_action

        offer_value_discounted = None

        # Counter-offer rule
        required_utility = max(0.0, target_raw)
        required_utility = min(required_utility, self._total_value())

        desired_give = None
        print(f"Target {target_raw}")
        if self.current_opponent_offer_for_us is not None and self.total_quantities is not None:
            opponent_keep = self.total_quantities - np.array(self.current_opponent_offer_for_us, dtype=int)
            desired_give = np.clip(opponent_keep, 0, None).astype(int)
        #desired_keep = self.total_quantities - desired_give
        desired_keep = self._build_target_keep(required_utility, desired_give)
        if (
            desired_give is None
            and self.total_quantities is not None
            and desired_keep is not None
            and np.all(self.total_quantities - desired_keep == 0)
        ):
            adjusted_keep = desired_keep.copy()
            value_order = sorted(
                range(self.num_items),
                key=lambda idx: (self.valuation_vector[idx], idx),
            )
            for idx in value_order:
                if adjusted_keep[idx] <= 0:
                    continue
                adjusted_keep[idx] -= 1
                if self._calculate_keep_utility(adjusted_keep) + 1e-9 >= required_utility:
                    desired_keep = adjusted_keep.astype(int)
                    break
                adjusted_keep[idx] += 1
        counter_action = self._select_counter_action(state, required_utility, desired_keep, desired_give)
        if counter_action is not None:
            self.action = "COUNTEROFFER"
            return counter_action

        if self.current_opponent_offer_for_us is not None:
            if offer_value_discounted is None:
                offer_value_discounted = discount_power * self._calculate_receive_utility(
                    self.current_opponent_offer_for_us
                )
            if discounted_batna + 1e-9 >= discounted_target and discounted_batna + 1e-9 >= offer_value_discounted:
                walk_action = self._create_walk_action(state)
                if walk_action is not None:
                    self.action = "WALK"
                    return walk_action

        # Fallbacks
        walk_action = self._create_walk_action(state)
        if walk_action is not None:
            self.action = "WALK"
            return walk_action

        accept_action = self._create_accept_action(state)
        if accept_action is not None:
            self.action = "ACCEPT"
            return accept_action

        legal_actions = state.legal_actions()
        fallback = legal_actions[0] if legal_actions else 0
        self.action = "FALLBACK"
        return fallback

    def _create_accept_action(self, state):
        for action in state.legal_actions():
            action_str = state.action_to_string(state.current_player(), action)
            if "Agreement" in action_str:
                return action
        return None
 
    def _create_walk_action(self, state):
        try:
            legal_actions = state.legal_actions()
            for action in legal_actions:
                action_str = state.action_to_string(state.current_player(), action)
                if "walk away" in action_str.lower():
                    return action
            if legal_actions:
                return max(legal_actions)
            return None
        except Exception:
            return None

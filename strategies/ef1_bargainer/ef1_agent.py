"""
EF1 (Envy-Free up to 1 item) deterministic agent for the 7-4-1 bargaining game.

Strategy:
1. Computes all allocations satisfying EF1 fairness for the agent
2. Finds the Pareto frontier of most generous (to opponent) EF1 allocations
3. Opening: offers from frontier with most total items to opponent
4. Accepting: accepts any EF1 offer that also beats outside option
5. Countering: cycles through frontier offers emphasizing different item types
6. Walking: walks if no EF1 offer beats outside option, or opponent's
   offer is not EF1 and no counteroffer is possible
"""

import torch
import torch.nn as nn

QUANTITIES = (7, 4, 1)
NUM_ITEM_TYPES = 3
ACTION_ACCEPT = 80
ACTION_WALK = 81



class EF1Agent:

    def __init__(self):
        self._all_allocations = []
        for n0 in range(QUANTITIES[0] + 1):
            for n1 in range(QUANTITIES[1] + 1):
                for n2 in range(QUANTITIES[2] + 1):
                    self._all_allocations.append((n0, n1, n2))

    def _decode_observation(self, obs):
        values = [obs[i] * 100.0 for i in range(3)]
        current_player = int(round(obs[9]))
        offer_valid = obs[7] > 0.5
        round_num = int(round(obs[8] * 2))
        action_mask = [obs[10 + i] > 0.5 for i in range(82)]

        max_possible = sum(values[i] * QUANTITIES[i] for i in range(NUM_ITEM_TYPES))
        outside_raw = obs[3] * max_possible if max_possible > 0 else 0.0

        my_items = None
        opp_items = None
        if offer_valid:
            my_items = tuple(int(round(obs[4 + i] * QUANTITIES[i])) for i in range(NUM_ITEM_TYPES))
            opp_items = tuple(QUANTITIES[i] - my_items[i] for i in range(NUM_ITEM_TYPES))

        return {
            'values': values,
            'current_player': current_player,
            'offer_valid': offer_valid,
            'round': round_num,
            'action_mask': action_mask,
            'outside_raw': outside_raw,
            'my_items': my_items,
            'opp_items': opp_items,
        }

    @staticmethod
    def _value_of(values, items):
        return sum(v * n for v, n in zip(values, items))

    @staticmethod
    def _is_ef1(values, my_items, opp_items):
        my_val = EF1Agent._value_of(values, my_items)
        opp_val = EF1Agent._value_of(values, opp_items)

        if my_val >= opp_val:
            return True

        max_removal = 0.0
        for j in range(NUM_ITEM_TYPES):
            if opp_items[j] > 0:
                max_removal = max(max_removal, values[j])

        return my_val >= opp_val - max_removal

    def _find_ef1_allocations(self, values):
        ef1 = []
        for my_items in self._all_allocations:
            opp_items = tuple(QUANTITIES[i] - my_items[i] for i in range(NUM_ITEM_TYPES))
            if self._is_ef1(values, my_items, opp_items):
                ef1.append((my_items, opp_items))
        return ef1

    @staticmethod
    def _pareto_frontier(ef1_allocations):
        frontier = []
        for alloc in ef1_allocations:
            _, opp = alloc
            dominated = False
            for other in ef1_allocations:
                _, other_opp = other
                if other_opp == opp:
                    continue
                if (all(other_opp[i] >= opp[i] for i in range(NUM_ITEM_TYPES))
                        and any(other_opp[i] > opp[i] for i in range(NUM_ITEM_TYPES))):
                    dominated = True
                    break
            if not dominated:
                frontier.append(alloc)
        return frontier

    @staticmethod
    def _sort_frontier(frontier):
        def sort_key(alloc):
            _, opp_items = alloc
            total_given = sum(opp_items)
            fractions = tuple(opp_items[i] / QUANTITIES[i] for i in range(NUM_ITEM_TYPES))
            dominant_idx = max(range(NUM_ITEM_TYPES), key=lambda i: fractions[i])
            return (-total_given, dominant_idx, tuple(-f for f in fractions))

        return sorted(frontier, key=sort_key)

    @staticmethod
    def _allocation_to_action(my_items, opp_items, current_player):
        if current_player == 0:
            n0, n1, n2 = opp_items
        else:
            n0, n1, n2 = my_items
        return n0 * 10 + n1 * 2 + n2

    def act(self, obs):
        p = self._decode_observation(obs)
        values = p['values']
        current_player = p['current_player']
        round_num = p['round']
        offer_valid = p['offer_valid']
        action_mask = p['action_mask']
        outside_raw = p['outside_raw']

        ef1_allocs = self._find_ef1_allocations(values)

        if not ef1_allocs:
            return ACTION_WALK

        best_ef1_value = max(self._value_of(values, my) for my, _ in ef1_allocs)
        if outside_raw >= best_ef1_value:
            return ACTION_WALK

        frontier = self._pareto_frontier(ef1_allocs)
        sorted_frontier = self._sort_frontier(frontier)
        sorted_frontier = [
            (my, opp) for my, opp in sorted_frontier
            if self._value_of(values, my) > outside_raw + 1e-6
        ]



        if offer_valid:
            my_items = p['my_items']
            opp_items = p['opp_items']
            if self._is_ef1(values, my_items, opp_items):
                my_val = self._value_of(values, my_items)
                if my_val > outside_raw + 1e-6:
                    return ACTION_ACCEPT
                else:
                    return ACTION_WALK
            else:
                can_counteroffer = any(action_mask[i] for i in range(80))
                if not can_counteroffer:
                    return ACTION_WALK
                return self._pick_counteroffer(sorted_frontier, current_player, round_num)
        else:
            return self._pick_counteroffer(sorted_frontier, current_player, round_num)

    def _pick_counteroffer(self, sorted_frontier, current_player, round_num):
        if not sorted_frontier:
            return ACTION_WALK

        idx = round_num % len(sorted_frontier)
        my_items, opp_items = sorted_frontier[idx]
        return self._allocation_to_action(my_items, opp_items, current_player)

    def act_batched(self, obs_batch):
        return [self.act(obs_batch[i]) for i in range(len(obs_batch))]


if __name__ == '__main__':
    agent = EF1Agent()

    # Test 1: EF1 check with known values
    print("=== Test 1: EF1 Check ===")
    values = [50.0, 50.0, 50.0]
    # Agent keeps (4, 2, 1) = 350, opponent gets (3, 2, 0) = 250. 350 >= 250 -> EF (trivially EF1)
    assert EF1Agent._is_ef1(values, (4, 2, 1), (3, 2, 0))
    # Agent keeps (1, 0, 0) = 50, opponent gets (6, 4, 1) = 550. 50 >= 550 - 50 = 500? No.
    assert not EF1Agent._is_ef1(values, (1, 0, 0), (6, 4, 1))
    # Agent keeps (3, 2, 0) = 250, opponent gets (4, 2, 1) = 350. 250 >= 350 - 50 = 300? No.
    assert not EF1Agent._is_ef1(values, (3, 2, 0), (4, 2, 1))
    # Agent keeps (3, 2, 1) = 300, opponent gets (4, 2, 0) = 300. 300 >= 300 -> EF.
    assert EF1Agent._is_ef1(values, (3, 2, 1), (4, 2, 0))
    print("  All EF1 checks passed")

    # Test 2: EF1 with asymmetric values
    print("=== Test 2: Asymmetric Values ===")
    values = [100.0, 1.0, 1.0]
    # Agent keeps (4, 0, 0) = 400, opponent gets (3, 4, 1) = 305. EF.
    assert EF1Agent._is_ef1(values, (4, 0, 0), (3, 4, 1))
    # Agent keeps (3, 4, 1) = 305, opponent gets (4, 0, 0) = 400. 305 >= 400 - 100 = 300? Yes -> EF1.
    assert EF1Agent._is_ef1(values, (3, 4, 1), (4, 0, 0))
    # Agent keeps (2, 4, 1) = 205, opponent gets (5, 0, 0) = 500. 205 >= 500 - 100 = 400? No.
    assert not EF1Agent._is_ef1(values, (2, 4, 1), (5, 0, 0))
    print("  All asymmetric checks passed")

    # Test 3: Pareto frontier
    print("=== Test 3: Pareto Frontier ===")
    values = [50.0, 50.0, 50.0]
    ef1_allocs = agent._find_ef1_allocations(values)
    frontier = EF1Agent._pareto_frontier(ef1_allocs)
    sorted_f = EF1Agent._sort_frontier(frontier)
    print(f"  EF1 allocations: {len(ef1_allocs)}")
    print(f"  Pareto frontier size: {len(frontier)}")
    print(f"  Frontier offers (my, opp):")
    for my, opp in sorted_f:
        total = sum(opp)
        print(f"    keep={my} give={opp} total_to_opp={total}")

    # Test 4: Action encoding P1 vs P2
    print("=== Test 4: Action Encoding ===")
    my_items = (4, 2, 0)
    opp_items = (3, 2, 1)
    # P1: encode opp_items -> 3*10 + 2*2 + 1 = 35
    assert EF1Agent._allocation_to_action(my_items, opp_items, 0) == 35
    # P2: encode my_items -> 4*10 + 2*2 + 0 = 44
    assert EF1Agent._allocation_to_action(my_items, opp_items, 1) == 44
    print("  Action encoding passed")

    # Test 5: Full strategy with sample values
    print("=== Test 5: Full Strategy ===")
    values = [80.0, 20.0, 50.0]
    ef1_allocs = agent._find_ef1_allocations(values)
    frontier = EF1Agent._pareto_frontier(ef1_allocs)
    sorted_f = EF1Agent._sort_frontier(frontier)
    print(f"  Values: {values}")
    print(f"  EF1 allocations: {len(ef1_allocs)}")
    print(f"  Pareto frontier: {len(sorted_f)}")
    print(f"  Opening offer (most generous):")
    if sorted_f:
        my, opp = sorted_f[0]
        print(f"    keep={my} give={opp} total_to_opp={sum(opp)}")
        print(f"    My value: {EF1Agent._value_of(values, my)}")
        print(f"    Opp value (from my perspective): {EF1Agent._value_of(values, opp)}")

    print("\nAll tests passed!")


class EF1Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = EF1Agent()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, obs, action_mask):
        batch_size = obs.shape[0]
        logits = torch.full((batch_size, 82), -1e9)
        value = torch.zeros(batch_size)

        for i in range(batch_size):
            obs_np = obs[i].cpu().numpy()
            action = self.agent.act(obs_np)
              # Only set logit if action is valid per mask
            if action_mask[i, action] > 0.5:
                logits[i, action] = 0.0
            else:
                  # Fallback: pick best valid masked action
                logits[i] = action_mask[i].float() * 0.0 + (1 - action_mask[i].float()) * -1e9
        return logits, value
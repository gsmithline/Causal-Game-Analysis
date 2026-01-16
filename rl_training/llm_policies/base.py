"""
Base LLM Policy interface for bargaining game.

Provides abstract interface and common utilities for LLM-based negotiators.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import re
import json


@dataclass
class BargainGameState:
    """Parsed bargaining game state from observation."""

    # Player's private info
    item_values: Tuple[float, float, float]  # Normalized values for 3 item types
    outside_offer: float  # Normalized outside option value

    # Current game state
    current_offer: Optional[Tuple[int, int, int]]  # Items offered to player (None if no offer)
    offer_valid: bool
    round_num: int  # 1, 2, or 3
    is_my_turn: bool

    # Valid actions
    valid_actions: List[int]
    can_accept: bool
    can_walk: bool

    # Item quantities for reference
    item_quantities: Tuple[int, int, int] = (7, 4, 1)

    def compute_offer_value(self, offer: Tuple[int, int, int]) -> float:
        """Compute value of an offer given player's values."""
        return sum(o * v for o, v in zip(offer, self.item_values))

    def compute_remaining_value(self, offer: Tuple[int, int, int]) -> float:
        """Compute value of items NOT in the offer (what opponent keeps)."""
        remaining = tuple(q - o for q, o in zip(self.item_quantities, offer))
        return sum(r * v for r, v in zip(remaining, self.item_values))

    def to_prompt_context(self) -> str:
        """Convert game state to natural language for LLM."""
        lines = []

        # Describe the game setup
        lines.append("=== BARGAINING GAME STATE ===")
        lines.append(f"Round: {self.round_num} of 3")
        lines.append("")

        # Item values (denormalized approximately for readability)
        lines.append("Your private item values (per unit):")
        lines.append(f"  - Item A (7 available): {self.item_values[0] * 100:.0f} points")
        lines.append(f"  - Item B (4 available): {self.item_values[1] * 100:.0f} points")
        lines.append(f"  - Item C (1 available): {self.item_values[2] * 100:.0f} points")
        lines.append("")

        # Outside option
        lines.append(f"Your outside option (if you WALK): {self.outside_offer * 100:.0f} points")
        lines.append("")

        # Current offer
        if self.current_offer is not None and self.offer_valid:
            offer_value = self.compute_offer_value(self.current_offer)
            lines.append(f"Current offer on the table (items YOU would receive):")
            lines.append(f"  - {self.current_offer[0]} of Item A")
            lines.append(f"  - {self.current_offer[1]} of Item B")
            lines.append(f"  - {self.current_offer[2]} of Item C")
            lines.append(f"  Value to you: {offer_value * 100:.0f} points")
            lines.append("")

            if self.can_accept:
                lines.append("You can ACCEPT this offer.")
        else:
            lines.append("No offer currently on the table.")
            lines.append("")

        # Available actions
        lines.append("Your options:")
        if self.can_accept:
            lines.append("  1. ACCEPT - Accept the current offer")
        lines.append("  2. WALK - Take your outside option and end negotiation")
        lines.append("  3. COUNTEROFFER - Propose a new division of items")

        return "\n".join(lines)


def parse_observation(obs: torch.Tensor) -> BargainGameState:
    """
    Parse a single observation tensor into BargainGameState.

    Args:
        obs: [92] float tensor

    Returns:
        BargainGameState with parsed information
    """
    obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs

    # Extract fields from observation
    # [0-2]: Player's normalized item values
    item_values = tuple(obs_np[0:3])

    # [3]: Normalized outside offer
    outside_offer = obs_np[3]

    # [4-6]: Current offer items (-1 if none)
    offer_items = obs_np[4:7]
    if offer_items[0] >= 0:  # Valid offer exists
        current_offer = tuple(int(x) for x in offer_items)
    else:
        current_offer = None

    # [7]: Offer valid flag
    offer_valid = obs_np[7] > 0.5

    # [8]: Normalized round (0, 0.33, 0.67 for rounds 1, 2, 3)
    round_num = int(round(obs_np[8] * 3)) + 1
    round_num = max(1, min(3, round_num))

    # [9]: Current player (0 or 1)
    is_my_turn = True  # Observation is from current player's perspective

    # [10-91]: Action mask (82 booleans)
    action_mask = obs_np[10:92]
    valid_actions = [i for i, m in enumerate(action_mask) if m > 0.5]

    can_accept = 80 in valid_actions
    can_walk = 81 in valid_actions

    return BargainGameState(
        item_values=item_values,
        outside_offer=outside_offer,
        current_offer=current_offer,
        offer_valid=offer_valid,
        round_num=round_num,
        is_my_turn=is_my_turn,
        valid_actions=valid_actions,
        can_accept=can_accept,
        can_walk=can_walk,
    )


def parse_llm_action(
    response: str,
    valid_actions: List[int],
    item_quantities: Tuple[int, int, int] = (7, 4, 1),
) -> int:
    """
    Parse LLM response to extract action.

    Looks for:
    - "ACCEPT" -> action 80
    - "WALK" -> action 81
    - Offer pattern like "(3, 2, 1)" or "3 of A, 2 of B, 1 of C" -> encoded action

    Args:
        response: LLM response text
        valid_actions: List of valid action indices
        item_quantities: Max quantities for each item type

    Returns:
        Action index
    """
    response_upper = response.upper()

    # Check for ACCEPT
    if "ACCEPT" in response_upper and 80 in valid_actions:
        return 80

    # Check for WALK
    if "WALK" in response_upper and 81 in valid_actions:
        return 81

    # Try to parse counteroffer
    # Pattern 1: (n0, n1, n2) or [n0, n1, n2]
    tuple_pattern = r'[\(\[]?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]?'
    match = re.search(tuple_pattern, response)

    if match:
        n0, n1, n2 = int(match.group(1)), int(match.group(2)), int(match.group(3))
        # Validate
        if 0 <= n0 <= item_quantities[0] and 0 <= n1 <= item_quantities[1] and 0 <= n2 <= item_quantities[2]:
            action_idx = n0 * 10 + n1 * 2 + n2
            if action_idx in valid_actions:
                return action_idx

    # Pattern 2: "n of Item A, m of Item B, k of Item C"
    item_pattern = r'(\d+)\s*(?:of\s*)?(?:item\s*)?[aA].*?(\d+)\s*(?:of\s*)?(?:item\s*)?[bB].*?(\d+)\s*(?:of\s*)?(?:item\s*)?[cC]'
    match = re.search(item_pattern, response, re.IGNORECASE)

    if match:
        n0, n1, n2 = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 0 <= n0 <= item_quantities[0] and 0 <= n1 <= item_quantities[1] and 0 <= n2 <= item_quantities[2]:
            action_idx = n0 * 10 + n1 * 2 + n2
            if action_idx in valid_actions:
                return action_idx

    # Pattern 3: JSON format {"action": "COUNTEROFFER", "offer": [n0, n1, n2]}
    try:
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            data = json.loads(json_match.group())
            if "offer" in data:
                offer = data["offer"]
                if len(offer) == 3:
                    n0, n1, n2 = int(offer[0]), int(offer[1]), int(offer[2])
                    action_idx = n0 * 10 + n1 * 2 + n2
                    if action_idx in valid_actions:
                        return action_idx
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Fallback: return first valid counteroffer action
    counteroffer_actions = [a for a in valid_actions if a < 80]
    if counteroffer_actions:
        return counteroffer_actions[len(counteroffer_actions) // 2]  # Middle action

    # Last resort: WALK if available
    if 81 in valid_actions:
        return 81

    return valid_actions[0] if valid_actions else 0


class BaseLLMPolicy(ABC):
    """
    Abstract base class for LLM-based negotiation policies.

    Subclasses implement the actual LLM API calls.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        self.system_prompt = system_prompt or self._default_system_prompt()

        # Track conversation history for multi-turn games
        self.conversation_history: List[Dict[str, str]] = []

        # Stats
        self.total_api_calls = 0
        self.total_tokens_used = 0

    def _default_system_prompt(self) -> str:
        return """You are an expert negotiator in a bargaining game. You must divide items between yourself and another player.

GAME RULES:
- There are 3 item types with quantities: 7 of Item A, 4 of Item B, 1 of Item C
- Each player has private values for each item type (you don't know opponent's values)
- Each player has a private outside option (fallback if negotiation fails)
- You have up to 3 rounds to reach an agreement
- If no agreement after round 3, both players get their outside options

YOUR GOAL: Maximize your total points while reaching an agreement.

ACTIONS:
- ACCEPT: Accept the current offer on the table
- WALK: Take your outside option and end negotiation
- COUNTEROFFER: Propose how to divide items (specify how many of each item YOU receive)

RESPONSE FORMAT:
Respond with your action. For counteroffers, specify the items YOU want to receive.
Example: "COUNTEROFFER: (4, 2, 1)" means you want 4 of Item A, 2 of Item B, 1 of Item C.
The opponent receives the remaining items: (3, 2, 0).

Think strategically about what division would be acceptable to both parties."""

    @abstractmethod
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Make API call to LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            LLM response text
        """
        pass

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get actions for a batch of observations.

        Note: LLM policies process one observation at a time due to API constraints.
        For batched envs, this loops through each observation.

        Args:
            obs: [batch, obs_dim] observations
            action_mask: [batch, num_actions] valid action masks
            deterministic: If True, use temperature=0 (ignored for now)

        Returns:
            actions: [batch] selected actions
        """
        batch_size = obs.shape[0]
        device = obs.device
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            action = self._get_single_action(obs[i], action_mask[i])
            actions[i] = action

        return actions

    def _get_single_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> int:
        """Get action for a single observation."""
        # Parse observation
        state = parse_observation(obs)

        # Build prompt
        user_message = state.to_prompt_context()
        user_message += "\n\nWhat is your action? (ACCEPT, WALK, or COUNTEROFFER with your proposed division)"

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Add conversation history for context
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        # Call LLM
        response = self._call_llm(messages)

        if self.verbose:
            print(f"\n=== LLM ({self.model_name}) ===")
            print(f"State: Round {state.round_num}, Offer: {state.current_offer}")
            print(f"Response: {response[:200]}...")

        # Parse response to action
        valid_actions = [i for i, m in enumerate(action_mask.cpu().numpy()) if m > 0.5]
        action = parse_llm_action(response, valid_actions)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})

        self.total_api_calls += 1

        return action

    def reset_conversation(self):
        """Reset conversation history for a new game."""
        self.conversation_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model": self.model_name,
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
        }

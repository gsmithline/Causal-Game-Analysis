import os
import re
import json
import torch
import torch.nn as nn
from openai import OpenAI
from typing import Tuple, Optional
import numpy as np

_KEY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "OPENAI_API_KEY.txt")

NUM_OFFERS = 80
ACTION_ACCEPT = 80
ACTION_WALK = 81
NUM_ACTIONS = 82
ITEM_QUANTITIES = [7, 4, 1]
MAX_VALUE = 100
TOTAL_ROUNDS = 3

SYSTEM_PROMPT = (
    "You are an AI negotiator participating in a negotiation game.\n\n"
    "First, explain your reasoning about the current game state and your strategy.\n"
    "Then, provide your action as a JSON object on its own line in one of these formats:\n"
    '1. {"action": "ACCEPT"}\n'
    '2. {"action": "WALK"}\n'
    '3. {"action": "COUNTEROFFER", "offer": [n1, n2, n3]}\n\n'
    "You MUST include both your reasoning AND the JSON action in your response."
)

def _parse_obs(obs_row: torch.Tensor) -> dict:
    """
    parse single observation for prompt
    
    :param obs_row: Description
    :type obs_row: torch.Tensor
    :return: dict
    :rtype: dict[Any, Any]
    """

    item_values_norm = obs_row[0:3].tolist()
    outside_option_norm = obs_row[3].item()
    offer_items_norm = obs_row[4:7].tolist()
    offer_valid = obs_row[7].item() > 0.5
    round_norm = obs_row[8].item()
    current_player = int(obs_row[9].item() > 0.5)

    item_values = [v * MAX_VALUE for v in item_values_norm]

    #denormalizr round 
    round_num = int(round(round_norm * 2))
    # Denormalize offer (was divided by ITEM_QUANTITIES)
    current_offer = None
    if offer_valid:
        current_offer = [
            int(round(offer_items_norm[i] * ITEM_QUANTITIES[i]))
            for i in range(3)
        ]

    return {
        "item_values": item_values,
        "outside_option_norm": outside_option_norm,
        "offer_valid": offer_valid,
        "current_offer": current_offer,
        "round": round_num,
        "player": current_player,
    }

def _build_user_prompt(state: dict) -> str:
    """
    takes in an observation and builds prompt for current game state
    """
    current_player_num = state["player"] + 1
    other_player_num = 2 if current_player_num == 1 else 1
    current_round = state["round"] + 1

    current_offer_str = ""
    if state["offer_valid"]:
        offer = state["current_offer"]
        current_offer_str = f"\nCurrent offer on the table (the amount of each item being offered to you): [{offer[0]}, {offer[1]}, {offer[2]}]"
    else:
        current_offer_str = "\nCurrent offer on the table (the amount of each item being offered to you): None"

    if current_round == 1 and current_player_num == 1:
        action_prompt = f"""
    What is your action? As the first player, your available actions are:
    - WALK to walk away
    - A list of numbers [n1, n2, n3] representing your initial offer (what you give to Player {other_player_num})"""
    elif not state["offer_valid"]:
        action_prompt = f"""
    What is your action? You can:
    - WALK to walk away
    - A list of numbers [n1, n2, n3] representing your offer (what you give to Player {other_player_num})"""
    else:
        action_prompt = f"""
    What is your action? You can:
    - ACCEPT to accept the current offer
    - WALK to walk away
    - A list of numbers [n1, n2, n3] representing your counteroffer (what you give to Player {other_player_num})"""

    values_str = ", ".join(
        [f"{v:.0f} for item {i+1}" for i, v in enumerate(state["item_values"])]
    )
    quantities_str = ", ".join(
        [f"{q} unit{'s' if q != 1 else ''} of item {i+1}"
         for i, q in enumerate(ITEM_QUANTITIES)]
    )

    core_prompt = f"""
    You and another agent have to negotiate a division of items between the two of you.
    You are Player {current_player_num} and the other agent is Player {other_player_num}.
    There are three types of items, called item 1 through item 3.
    There are {quantities_str} to divide.
    Both you and Player {other_player_num} have a private value per unit of each item type.
    These values are drawn from a uniform random distribution of integers, ranging from 1 to {MAX_VALUE}.
    Your private values are {values_str}.
    You have a private outside offer drawn from a uniform random distribution ranging from 1 to your total value of all items. Player {other_player_num} has a private outside offer drawn from a uniform random distribution ranging from 1 to their total value of all items.
    Your outside offer value (normalized) is {state["outside_option_norm"]:.2f}.
    Your outside offer value in utility is {int(state["outside_option_norm"] * sum(v * q for v, q in zip(state["item_values"], ITEM_QUANTITIES)))}. 
    Your goal is to maximize your utility.

    The negotiation proceeds in {TOTAL_ROUNDS} rounds.
    At each round, Player 1 takes an action, followed by Player 2.
    The possible actions are to ACCEPT the other player's current offer (if any), make a COUNTEROFFER, or WALK away. If the game gets to the last round, and Player 2 chooses to make a counteroffer, this is treated as a WALK.
    If a player chooses ACCEPT, the negotiation ends in a deal to divide the items according to the accepted offer.
    The value of an outcome is determined by each player's private values per unit of each item and the quantities they receive in the deal.
    If a player chooses WALK, the negotiation ends without a deal, and each player receives the value of their private outside offer.

    Please provide your action in one of these formats in your response (if you do not do this your response will be invalid):
    {{"action": "ACCEPT"}} - to accept the current offer
    {{"action": "WALK"}} - to walk away from negotiations
    {{"action": "COUNTEROFFER", "offer": [n1, n2, n3]}} - where n1, n2, n3 are numbers representing the number of units of each item being offered to the other player as part of the counteroffer.

    Any response not in these exact formats will be invalid and treated as a WALK. If you provide a counteroffer, it must be a valid offer, otherwise it will be treated as a WALK.

    It is now round {current_round}.
    """

    return f"{core_prompt}\n{current_offer_str}\n{action_prompt}"

def _parse_llm_response(response_text: str, action_mask: torch.Tensor, player: int) -> int:
    """
    Parse llm json response into an action index.

    The LLM always thinks in terms of "what I give to the other player."
    For P1 (player=0) this matches the env encoding directly (offer = items to P2).
    For P2 (player=1) we invert: the LLM says "I give X to P1" but the env
    encodes offers as items P2 keeps, so we convert: env_offer = ITEM_QUANTITIES - llm_offer.
    """

    try:
        # Extract JSON from response (may contain reasoning text before/after)
        text = response_text.strip()

        # Try 1: handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Try 2: find a JSON object with "action" key in the text
        try:
            decision = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
            if match:
                text = match.group(0)
            else:
                return ACTION_WALK

        decision = json.loads(text)
        action_type = decision["action"].upper()

        if action_type == "ACCEPT":
            if action_mask[ACTION_ACCEPT] > 0:
                return ACTION_ACCEPT
            return ACTION_WALK

        if action_type == "WALK":
            return ACTION_WALK

        if action_type == "COUNTEROFFER":
            offer = decision["offer"]
            n1, n2, n3 = int(offer[0]), int(offer[1]), int(offer[2])

            # validate ranges (LLM gives what it sends to the other player)
            if not (0 <= n1 <= 7 and 0 <= n2 <= 4 and 0 <= n3 <= 1):
                return ACTION_WALK

            if player == 1:
                # P2: LLM says "I give [n1,n2,n3] to P1"
                # Env encodes offer as items going to P2 (what P2 keeps)
                # So env_offer = ITEM_QUANTITIES - llm_offer
                n1 = ITEM_QUANTITIES[0] - n1
                n2 = ITEM_QUANTITIES[1] - n2
                n3 = ITEM_QUANTITIES[2] - n3

            action_idx = n1 * 10 + n2 * 2 + n3
            if action_mask[action_idx] > 0:
                return action_idx
            return ACTION_WALK

    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
          return ACTION_WALK

    return ACTION_WALK

class OpenAIPolicy(nn.Module):
    """
      LLM-based bargaining policy using the OpenAI API.

      Translates observations into natural language, queries the LLM,
      and converts the response back into action logits.

      Note: This policy makes API calls and is NOT differentiable.
      Use with batch_size=1 or small batches for cost/latency reasons.
    """

    def __init__(
          self,
          model: str = "gpt-4o",
          temperature: float = 0.7,
          reasoning_effort: Optional[str] = None,
          api_key: Optional[str] = None,
        ):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None and os.path.exists(_KEY_FILE):
            with open(_KEY_FILE) as f:
                api_key = f.read().strip()
        self._api_key = api_key
        self.client = OpenAI(api_key=api_key)  
        self._last_trace = None  

    def __deepcopy__(self, memo):
        """Create a fresh instance to avoid pickling the httpx client."""
        return OpenAIPolicy(
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            api_key=self._api_key,
        )

    def _query_llm(self, state: dict) -> Tuple[str, Optional[list]]:
        """Send game state to the LLM and get a response.

        Returns:
            (output_text, reasoning_summary) where reasoning_summary is a list
            of summary strings from the model's reasoning, or None.
        """
        prompt = _build_user_prompt(state)
        kwargs = dict(
            model=self.model,
            #temperature=self.temperature,
            instructions=SYSTEM_PROMPT,
            input=prompt,
        )
        if self.reasoning_effort is not None:
            reasoning_config = {"summary": "detailed"}
            if self.reasoning_effort != "none":
                reasoning_config["effort"] = self.reasoning_effort
            kwargs["reasoning"] = reasoning_config
        response = self.client.responses.create(**kwargs)

        # Extract reasoning summary from response output items
        reasoning_summary = None
        for item in response.output:
            if getattr(item, 'type', None) == 'reasoning':
                summaries = getattr(item, 'summary', None)
                if summaries:
                    reasoning_summary = [
                        s.text for s in summaries if hasattr(s, 'text')
                    ]

        return response.output_text, reasoning_summary

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
          Args:
              obs: [batch, 92] observation tensor
              action_mask: [batch, 82] action mask tensor

          Returns:
              logits: [batch, 82] action logits (deterministic — one-hot)
              value: [batch] dummy value (zeros)
          """
        batch_size = obs.shape[0]
        device = obs.device
        logits = torch.full((batch_size, NUM_ACTIONS), -1e9, device=device)

        for i in range(batch_size):
            state = _parse_obs(obs[i])
            prompt = _build_user_prompt(state)
            response_text, reasoning_summary = self._query_llm(state)
            action_idx = _parse_llm_response(response_text, action_mask[i], state["player"])
            logits[i, action_idx] = 0.0

            self._last_trace = {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "player": state["player"],
                "round": state["round"],
                "prompt": prompt,
                "response": response_text,
                "reasoning_summary": reasoning_summary,
                "parsed_action": action_idx,
            }

        value = torch.zeros(batch_size, device=device)
        return logits, value
"""
OpenAI LLM Policy for bargaining game.

Supports GPT-5.2, o3, GPT-4o, and other OpenAI models.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import os
import time

from rl_training.llm_policies.base import BaseLLMPolicy


# Model configurations
OPENAI_MODELS = {
    # GPT-5.2 series (latest)
    "gpt-5.2-pro": {
        "api_name": "gpt-5.2-pro",
        "supports_system": True,
        "is_reasoning": False,
        "max_context": 128000,
    },
    "gpt-5.2-thinking": {
        "api_name": "gpt-5.2-thinking",
        "supports_system": True,
        "is_reasoning": True,
        "max_context": 128000,
    },
    "gpt-5.2-instant": {
        "api_name": "gpt-5.2-instant",
        "supports_system": True,
        "is_reasoning": False,
        "max_context": 128000,
    },
    "gpt-5.2": {
        "api_name": "gpt-5.2",
        "supports_system": True,
        "is_reasoning": False,
        "max_context": 128000,
    },
    # GPT-5.1 series
    "gpt-5.1": {
        "api_name": "gpt-5.1",
        "supports_system": True,
        "is_reasoning": False,
        "max_context": 128000,
    },
    "gpt-5.1-thinking": {
        "api_name": "gpt-5.1-thinking",
        "supports_system": True,
        "is_reasoning": True,
        "max_context": 128000,
    },
    # GPT-5 original
    "gpt-5": {
        "api_name": "gpt-5",
        "supports_system": True,
        "is_reasoning": False,
        "max_context": 128000,
    },
    # o3 reasoning models
    "o3": {
        "api_name": "o3",
        "supports_system": False,  # Reasoning models handle system differently
        "is_reasoning": True,
        "max_context": 128000,
    },
    "o3-mini": {
        "api_name": "o3-mini",
        "supports_system": False,
        "is_reasoning": True,
        "max_context": 128000,
    },
    # o1 reasoning models
    "o1": {
        "api_name": "o1",
        "supports_system": False,
        "is_reasoning": True,
        "max_context": 128000,
    },
    "o1-mini": {
        "api_name": "o1-mini",
        "supports_system": False,
        "is_reasoning": True,
        "max_context": 128000,
    },
    # GPT-4o series (previous gen)
    "gpt-4o": {
        "api_name": "gpt-4o",
        "supports_system": True,
        "is_reasoning": False,
        "max_context": 128000,
    },
    "gpt-4o-mini": {
        "api_name": "gpt-4o-mini",
        "supports_system": True,
        "is_reasoning": False,
        "max_context": 128000,
    },
}


class OpenAIPolicy(BaseLLMPolicy):
    """
    OpenAI-based negotiation policy.

    Supports GPT-5.2, o3, o1, GPT-4o, and other OpenAI models.

    Usage:
        policy = OpenAIPolicy("gpt-5.2-pro")
        action = policy.get_action(obs, action_mask)
    """

    def __init__(
        self,
        model_name: str = "gpt-5.2-pro",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        reasoning_effort: Optional[Literal["low", "medium", "high", "xhigh"]] = None,
        verbose: bool = False,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize OpenAI policy.

        Args:
            model_name: Model to use (e.g., "gpt-5.2-pro", "o3", "gpt-4o")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            system_prompt: Custom system prompt (optional)
            temperature: Sampling temperature (0-2)
            max_tokens: Max tokens in response
            reasoning_effort: For reasoning models (o1, o3, gpt-5.2-thinking)
            verbose: Print debug info
            retry_attempts: Number of retry attempts on API failure
            retry_delay: Delay between retries in seconds
        """
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        )

        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key param."
            )

        # Get model config
        if model_name not in OPENAI_MODELS:
            # Allow custom model names
            self.model_config = {
                "api_name": model_name,
                "supports_system": True,
                "is_reasoning": False,
                "max_context": 128000,
            }
        else:
            self.model_config = OPENAI_MODELS[model_name]

        self.reasoning_effort = reasoning_effort
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Initialize OpenAI client
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Make API call to OpenAI."""
        client = self._get_client()

        # Handle reasoning models that don't support system messages
        if not self.model_config["supports_system"]:
            # Prepend system message to first user message
            system_content = None
            filtered_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    filtered_messages.append(msg)

            if system_content and filtered_messages:
                # Prepend to first user message
                filtered_messages[0] = {
                    "role": filtered_messages[0]["role"],
                    "content": f"{system_content}\n\n---\n\n{filtered_messages[0]['content']}"
                }
            messages = filtered_messages

        # Build API call parameters
        params = {
            "model": self.model_config["api_name"],
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        # Add temperature for non-reasoning models
        if not self.model_config["is_reasoning"]:
            params["temperature"] = self.temperature

        # Add reasoning effort for reasoning models
        if self.model_config["is_reasoning"] and self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        # Retry logic
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                response = client.chat.completions.create(**params)

                # Track token usage
                if hasattr(response, "usage") and response.usage:
                    self.total_tokens_used += response.usage.total_tokens

                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"OpenAI API call failed after {self.retry_attempts} attempts: {last_error}")

    @property
    def name(self) -> str:
        """Policy name for identification."""
        return f"openai_{self.model_name}"


class OpenAIBatchPolicy(OpenAIPolicy):
    """
    OpenAI policy optimized for batch evaluation.

    Uses async API calls for better throughput when evaluating many games.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.2-pro",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        **kwargs,
    ):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self.max_concurrent = max_concurrent
        self._async_client = None

    def _get_async_client(self):
        """Lazy initialization of async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._async_client

    async def get_actions_async(
        self,
        observations: List,
        action_masks: List,
    ) -> List[int]:
        """
        Get actions for multiple observations concurrently.

        Args:
            observations: List of observation tensors
            action_masks: List of action mask tensors

        Returns:
            List of action indices
        """
        import asyncio

        async def get_single_action_async(obs, mask):
            # This would use async API - simplified here
            return self._get_single_action(obs, mask)

        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_call(obs, mask):
            async with semaphore:
                return await asyncio.to_thread(
                    self._get_single_action, obs, mask
                )

        tasks = [
            bounded_call(obs, mask)
            for obs, mask in zip(observations, action_masks)
        ]

        return await asyncio.gather(*tasks)


# Convenience factory functions
def create_gpt52_pro(**kwargs) -> OpenAIPolicy:
    """Create GPT-5.2 Pro policy."""
    return OpenAIPolicy("gpt-5.2-pro", **kwargs)


def create_gpt52_thinking(**kwargs) -> OpenAIPolicy:
    """Create GPT-5.2 Thinking policy."""
    return OpenAIPolicy("gpt-5.2-thinking", reasoning_effort="high", **kwargs)


def create_o3(**kwargs) -> OpenAIPolicy:
    """Create o3 reasoning policy."""
    return OpenAIPolicy("o3", reasoning_effort="high", **kwargs)


def create_gpt4o(**kwargs) -> OpenAIPolicy:
    """Create GPT-4o policy."""
    return OpenAIPolicy("gpt-4o", **kwargs)

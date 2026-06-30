"""
Minimal model price table (USD per 1M tokens).

Used to estimate cost from token counts. Prices are approximate and easy to
override — call `set_price()` or pass your own at span time. Keeping this local
avoids any network dependency.
"""

from typing import Dict, Tuple

# (prompt_per_1m, completion_per_1m)
_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-opus-4": (15.00, 75.00),
    "anthropic.claude-3-5-sonnet": (3.00, 15.00),  # bedrock id style
    "demo-model": (0.50, 1.50),
    # Groq (LPU) — very low cost, OpenAI-compatible
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "moonshotai/kimi-k2-instruct": (1.00, 3.00),
}


def set_price(model: str, prompt_per_1m: float, completion_per_1m: float) -> None:
    _PRICES[model] = (prompt_per_1m, completion_per_1m)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in _PRICES:
        return 0.0
    p_rate, c_rate = _PRICES[model]
    return (prompt_tokens / 1_000_000) * p_rate + (completion_tokens / 1_000_000) * c_rate

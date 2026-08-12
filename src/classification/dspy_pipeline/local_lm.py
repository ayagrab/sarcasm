"""dspy.BaseLM adapter that runs DSPy programs against the local Qwen
client (`LocalHFClient`) instead of an OpenAI-compatible API/litellm
provider -- so the DSPy experiments (M5) use the exact same underlying
model as the manual-prompt LLM experiments (M2-M4), with no external
server, no vLLM (unsupported on Tesla M60), and no API key.

Implements DSPy's documented "legacy custom LM for an OpenAI-like
provider" pattern (`dspy.BaseLM`, `forward_contract = "legacy"`):
`forward(prompt=None, messages=None, **kwargs)` returns an object shaped
like an OpenAI chat-completion response (`.choices[i].message.content`,
`.model`, optional `.usage` dict). Determined by inspecting
`dspy.BaseLM`'s source directly on the Stage B VM (dspy 3.3.0) -- see
EXPERIMENT_LOG.md, Stage B, M5 section.

Note: this response shape is intentionally NOT the same `_Response`
dataclass used by `local_client.py` for the OpenRouter-interface-shaped
client -- DSPy reads `.usage` with `dict(response.usage or {})`, which
requires a real dict (or falsy), not an arbitrary object; keeping the two
adapters separate avoids coupling two different consumer contracts to one
class.
"""
from __future__ import annotations

import dspy

from src.classification.llm.local_client import DEFAULT_CHECKPOINT, get_local_hf_client


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _DSPyChatResponse:
    def __init__(self, text: str, model: str, prompt_tokens: int, completion_tokens: int):
        self.choices = [_Choice(text)]
        self.model = model
        self.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


class LocalQwenLM(dspy.BaseLM):
    """DSPy LM backed by the local Qwen client. Loads the model once
    (via `get_local_hf_client`'s process-wide singleton) and reuses it for
    every DSPy call, including repeated calls during optimizer bootstrapping."""

    forward_contract = "legacy"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        temperature: float = 0.0,
        max_new_tokens: int = 200,
        **kwargs,
    ):
        super().__init__(model=checkpoint, temperature=temperature, **kwargs)
        # DSPy's default ChatAdapter output format (field markers, e.g.
        # "[[ ## label ## ]]\nsarcastic") is more verbose than the compact
        # JSON object the manual LLM pipeline uses, hence a higher default
        # max_new_tokens than local_client.py's (64).
        self._client = get_local_hf_client(checkpoint=checkpoint, max_new_tokens=max_new_tokens)

    def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        temperature = kwargs.get("temperature", self.kwargs.get("temperature"))
        temperature = 0.0 if temperature is None else temperature

        response = self._client.chat.completions.create(
            model=self.model, temperature=temperature, messages=messages
        )
        usage = response.usage
        return _DSPyChatResponse(
            text=response.choices[0].message.content,
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

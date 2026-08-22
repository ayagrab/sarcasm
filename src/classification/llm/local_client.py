"""Local Hugging Face Transformers inference client -- an alternative to
the OpenRouter API client, for running a small LLM directly on GPU.
Default checkpoint: `Qwen/Qwen3-4B-Instruct-2507`.

Built for the Azure Standard_NV24s_v3 VM: **Tesla M60 GPUs (Maxwell
architecture, CUDA compute capability 5.2)**. Maxwell does NOT support:
- bfloat16 (needs Ampere+, compute capability >= 8.0) -- use float16 or float32
- FlashAttention 2 (needs Ampere+) -- use `attn_implementation="eager"`
- modern vLLM (needs compute capability >= 7.0) -- use plain `transformers`
  generation instead (this module), not vLLM

**Do not import-and-instantiate `LocalHFClient` (i.e. do not load a model)
before confirming a CUDA GPU is actually visible on the target machine**
(`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`).
This module raises immediately if no CUDA GPU is visible, but that alone
doesn't confirm compute-capability-dependent assumptions (BF16/
FlashAttention2/vLLM support) -- check those explicitly first if targeting
different hardware than the Tesla M60s this was built for.

Mimics the small subset of the OpenAI/OpenRouter client interface that
`src.classification.llm.run_llm_classification` uses
(`client.chat.completions.create(model=..., temperature=..., messages=...)`
-> `.choices[0].message.content`), so the exact same zero-shot/few-shot/
reasoning pipeline runs unchanged against a local model -- only
`get_llm_client(provider="local_hf")` needs to be selected instead of
`provider="openrouter"`.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHECKPOINT = "Qwen/Qwen3-4B-Instruct-2507"

# Feature -> minimum CUDA compute capability required (Tesla M60 = 5.2,
# below both thresholds below).
UNSUPPORTED_ON_M60 = {
    "bfloat16": "Maxwell (compute capability 5.2) has no native BF16 support -- use float16 or float32.",
    "flash_attention_2": "FlashAttention2 requires compute capability >= 8.0 (Ampere+) -- use attn_implementation='eager'.",
}


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list
    usage: _Usage | None = None


class LocalHFClient:
    """Loads a causal LM once and exposes an OpenAI-client-shaped
    `.chat.completions.create(...)` for reuse by the existing LLM
    classification pipeline."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        dtype: str = "float16",
        attn_implementation: str = "eager",
        device_map: str = "auto",
        max_memory: dict | None = None,
        max_new_tokens: int = 64,
    ):
        import torch

        if dtype == "bfloat16":
            raise ValueError(UNSUPPORTED_ON_M60["bfloat16"])
        if attn_implementation == "flash_attention_2":
            raise ValueError(UNSUPPORTED_ON_M60["flash_attention_2"])
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU visible to torch -- do not load a model on "
                "unverified GPU hardware."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]

        # `max_memory` defaults to the verified-working per-GPU cap for a
        # 2x Tesla M60 (~7.93 GiB VRAM each) -- leaves headroom for
        # activations during generation. Override for other hardware.
        if max_memory is None and device_map == "auto":
            n_gpus = torch.cuda.device_count()
            max_memory = {i: "7GiB" for i in range(n_gpus)}
            max_memory["cpu"] = "180GiB"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch_dtype,  # `dtype=`, not the deprecated `torch_dtype=` (see PROJECT_SUMMARY.md, Stage B)
            attn_implementation=attn_implementation,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.chat = _ChatNamespace(self)


class _ChatNamespace:
    def __init__(self, client: "LocalHFClient"):
        self.completions = _CompletionsNamespace(client)


class _CompletionsNamespace:
    def __init__(self, client: "LocalHFClient"):
        self._client = client

    def create(self, model: str, temperature: float, messages: list[dict]) -> _Response:
        """`model` is accepted for interface compatibility but ignored --
        this client always uses the checkpoint it was loaded with."""
        import torch

        client = self._client
        prompt = client.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = client.tokenizer(prompt, return_tensors="pt").to(client.model.device)

        with torch.no_grad():
            output_ids = client.model.generate(
                **inputs,
                max_new_tokens=client.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=client.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = client.tokenizer.decode(generated, skip_special_tokens=True)

        usage = _Usage(
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(generated.shape[0]),
        )
        return _Response(choices=[_Choice(message=_Message(content=text))], usage=usage)


_client_singleton: LocalHFClient | None = None


def get_local_hf_client(checkpoint: str = DEFAULT_CHECKPOINT, **kwargs) -> LocalHFClient:
    """Loads (once per process) and returns a shared client -- loading a
    multi-GB model per call would be far too slow for a per-example loop."""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LocalHFClient(checkpoint=checkpoint, **kwargs)
    return _client_singleton

"""Model identifiers used in the project."""

GENERATION_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "nvidia": "nvidia/nemotron-nano-9b-v2:free",
    "liquid": "liquid/lfm-2.5-1.2b-thinking:free",
}

JUDGE_MODELS = {
    "openrouter_llm_judge": "openai/gpt-oss-20b:free",
    "nli": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
}

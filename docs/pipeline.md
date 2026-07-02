# Pipeline

The project pipeline combines the codebase and the methodology documented in the meeting presentations.

```text
Original SIGN paper/test data
        ↓
Remove reference/result column
        ↓
Clean sarcastic test tweets
        ↓
Run multiple LLMs with the same sarcasm-interpretation prompt
        ↓
Generate non-sarcastic interpretations
        ↓
Evaluate with automatic metrics: BLEU, ROUGE, PINC, combined score
        ↓
Evaluate semantic correctness with an independent LLM judge
        ↓
Run prompt-sensitivity experiments
        ↓
Create manual-scoring sample
        ↓
Summarize model performance
```

## Code mapping

| Pipeline step | Code |
|---|---|
| Clean dataset | `src.preprocessing.clean_dataset` |
| Gemini generation | `src.generation.generate_with_gemini` |
| OpenRouter generation | `src.generation.generate_with_openrouter` |
| LLM judge evaluation | `src.evaluation.evaluate_with_llm` |
| NLI evaluation alternative | `src.evaluation.evaluate_with_nli` |
| Automatic text metrics | `src.analysis.calculate_text_metrics` |
| Classification summary | `src.analysis.summarize_classifications` |
| Manual sample | `src.analysis.create_manual_sample` |

## Experiment folders

The refactored project stores repeated prompt/model runs in:

```text
data/model_outputs/experiment_01/
data/model_outputs/experiment_02/
data/model_outputs/experiment_03/
data/model_outputs/experiment_04/
```

These folders correspond to repeated experimental runs or prompt/model-output batches from the original project. The meeting slides specifically describe prompt-sensitivity testing across three prompt versions.


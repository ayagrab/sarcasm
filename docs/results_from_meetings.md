# Results Documented in Meeting Slides

This file records the main results described in the uploaded meeting presentations.

## Automatic metric results from Meeting 2

| Model | BLEU | ROUGE-1 | ROUGE-2 | PINC | Combined PINC × sigmoid(BLEU) |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash Lite | 4.25 | 29.20 | 11.17 | 74.62 | 38.10 |
| Baidu Qianfan OCR FastFree | 6.14 | 37.65 | 17.14 | 73.76 | 38.01 |
| Liquid LFM 2.5-1.2B Thinking | 1.14 | 10.57 | 2.40 | 89.08 | 44.79 |
| Nvidia Nemotron Nano 9B v2 | 5.47 | 34.52 | 14.11 | 72.94 | 37.47 |
| OpenAI GPT-OSS 20B | 6.64 | 38.88 | 17.20 | 67.02 | 34.62 |

Meeting interpretation:

- Liquid showed high novelty/PINC but weak meaning preservation.
- GPT-OSS showed stronger BLEU/ROUGE, suggesting better preservation of source structure and meaning.
- Automatic metrics alone were not considered sufficient, so the project continued toward LLM-judge evaluation.

## Prompt-sensitivity results from Meeting 3

| Prompt | Model | Average | Median | Count 1 | Count 2 | Count 3 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | gemini-2.5-flash-lite | 1.672 | 1 | 164 | 24 | 77 |
| 1 | liquid-lfm-2.5-1.2b-thinkingfree | 1.570 | 1 | 164 | 51 | 50 |
| 1 | nvidia-nemotron-nano-9b-v2free | 2.079 | 3 | 117 | 10 | 138 |
| 2 | gemini-2.5-flash-lite | 2.230 | 3 | 95 | 14 | 156 |
| 2 | liquid-lfm-2.5-1.2b-thinkingfree | 1.351 | 1 | 202 | 33 | 30 |
| 2 | nvidia-nemotron-nano-9b-v2free | 2.185 | 3 | 98 | 20 | 147 |
| 3 | gemini-2.5-flash-lite | 1.475 | 1 | 196 | 12 | 57 |
| 3 | liquid-lfm-2.5-1.2b-thinkingfree | 1.106 | 1 | 245 | 12 | 8 |
| 3 | nvidia-nemotron-nano-9b-v2free | 1.660 | 1 | 173 | 9 | 83 |

Key observation:

- Prompt 2 improved Gemini substantially.
- Nvidia performed strongly in prompts 1 and 2.
- Liquid performed weakly under all three prompt versions in the LLM-judge setup.


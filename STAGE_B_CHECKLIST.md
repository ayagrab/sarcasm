# Stage B Checklist — Resume Point

Purpose: if this session crashes/restarts, read this file first to see
exactly where things stood. Kept in sync after every milestone — a
checkbox is only ticked once genuinely done (not "started"). For full
detail/evidence behind any line here, see `EXPERIMENT_LOG.md` (the
authoritative, detailed audit trail — this file is just the map of it).
For final results, see `PROJECT_SUMMARY.md`.

**Last updated:** 2026-08-11, ~18:50 UTC. **PAUSED at user's request (VM being
shut down).** M2 DONE (Macro F1 0.6008). M3-random DONE (Macro F1 0.5880,
underperforms M2 -- see EXPERIMENT_LOG.md). M3-curated (EXP-004) was
**killed intentionally at 447/1340 examples (33%)** -- not a crash, a
deliberate stop. The chain script (`run_m3_m4_chain.sh`, PID 24865) was
also killed first, so it will NOT auto-launch M4 on its own anymore --
that script is gone and must be restarted manually (or just rerun the
EXP-004 command directly, see below). GPU confirmed idle (0 MiB used on
both GPUs) after stopping.

### How to resume (read this first when reconnecting)

1. `ssh -i ~/.ssh/azure_vm_key vmadmin@20.245.56.28`, confirm hostname is
   `dpmlgpuNC6sv32025s-0003`.
2. `source /mnt/vmadmin/sarcasm-env/bin/activate && export HF_HOME=/mnt/vmadmin/huggingface`
3. `cd /mnt/vmadmin/projects/sarcasm`
4. Re-run the **exact same EXP-004 command** -- do NOT change the config:
   `python -m src.classification.run_experiment --config configs/llm_few_shot_curated_8_qwen_local.json`
   Thanks to the per-example disk cache (`data/llm_cache/`, 3,088 cached
   responses as of the stop, including the 447 already computed for
   EXP-004), the already-computed examples return near-instantly; only the
   remaining ~893 will actually call the model. Expect roughly 45-55 min
   for the rest, not the full 1.5h.
5. Once EXP-004 finishes, M4 (`configs/llm_reasoning_qwen_local.json`,
   EXP-005) needs to be launched manually (the old chain script is dead) --
   either run it directly or write a fresh small chain script following the
   same pattern as `run_m3_m4_chain.sh` (that file still exists in the repo
   for reference even though its running instance was killed).
6. Tell me you're back and I'll pick up from here -- validate EXP-004,
   record it, and continue to M4/M5/M6 per this checklist.

**Remote machine:** Azure `Standard_NV24s_v3`, `dpmlgpuNC6sv32025s-0003`,
2x Tesla M60. Connect: `ssh -i ~/.ssh/azure_vm_key vmadmin@20.245.56.28`.
Repo on VM: `/mnt/vmadmin/projects/sarcasm`. Activate env:
`source /mnt/vmadmin/sarcasm-env/bin/activate && export HF_HOME=/mnt/vmadmin/huggingface`.

## TEST-sealing policy (corrected 2026-08-11)

**TEST is not touched at all until every method below is frozen on DEV.**
An earlier version of this checklist had each method (M2, M3, ...)
evaluate TEST right after its own DEV run -- corrected before any TEST
evaluation actually happened (EXP-002 TEST was never run), per explicit
instruction: even a config with "no real hyperparameters" should not
touch TEST until the whole development phase (M2-M6, all on TRAIN/DEV
only) is complete. See EXPERIMENT_LOG.md for the full note. Concretely:

- **Phase 1 (current):** run every method's DEV-only development below
  (M2 through M6). No TEST access of any kind -- not for scoring, not for
  inspection, not for example selection.
- **Phase 2 (only after Phase 1 fully checked off):** freeze exactly one
  configuration per method, record why, then evaluate each frozen config
  on TEST exactly once. No going back to change a method because its TEST
  result disappoints.

---

## 0. Infrastructure

- [x] SSH access working (BLOCKER-4b resolved)
- [x] Repo synced to VM (`scripts/sync_to_vm.sh`)
- [x] Remote env independently verified (torch/CUDA/transformers/dspy/accelerate/sentencepiece)
- [x] `scripts/verify_gpu.py` run for real on VM -- `fp16_transformers_pipeline_ok: True`
- [x] `environment_stage_b.txt` captured (hostname/python/nvidia-smi/pip freeze)
- [x] Repository-level Qwen smoke test (20 DEV examples, real GPU, real model) -- passed
- [x] Configs repointed: local Qwen = primary for M2-M5; OpenRouter = optional (`*-openrouter-optional` IDs)
- [x] DSPy local adapter written (`dspy_pipeline/local_lm.py`, `dspy.BaseLM` subclass) -- not yet smoke-tested (needs GPU free)
- [x] DeBERTa loading fixed (`use_fast_tokenizer=False`, `use_safetensors=True`, smoke-test config added) -- not yet run
- [x] 143/143 local tests pass, 42/42 GPU-independent tests pass on VM

## 1. M1 — TF-IDF + Logistic Regression (classical baseline)

- [x] **DONE (Stage A).** EXP-001, TEST Macro F1 = 0.740. Frozen -- not re-tuned, not touched in Stage B.

## PHASE 1 — DEV-only development (TEST sealed)

### 2. M2 — Qwen3-4B zero-shot (`configs/llm_zero_shot_qwen_local.json`, EXP-002)

- [x] Smoke test (20 DEV examples)
- [x] **Full DEV run -- DONE.** 1340/1340, ~37 min. Macro F1 0.6008, Accuracy 0.6440.
      Heavy skew toward predicting "sarcastic" (83% of predictions) -- investigated,
      confirmed genuine model behavior (uniform across categories, plausible on
      manual read of false positives), not a pipeline bug.
- [x] Inspect DEV metrics/predictions -- done, see above + EXPERIMENT_LOG.md
- [x] Record DEV result in `EXPERIMENT_LOG.md` (full metric set)
- [x] **TEST NOT touched** (per sealing policy) -- correct.

### 3. M3 — Qwen3-4B few-shot (random EXP-003, curated EXP-004)

- [x] Random variant, full DEV -- **DONE.** 1340/1340, ~1h34m. Macro F1 0.5880,
      Accuracy 0.6328 -- *underperforms* M2 zero-shot (0.6008/0.6440). Investigated
      (demo balance, category uniformity, agreement-with-M2 check) -- confirmed
      genuine result, not a bug. See EXPERIMENT_LOG.md.
- [ ] Curated variant, full DEV (`configs/llm_few_shot_curated_8_qwen_local.json`) -- **IN PROGRESS**
      (auto-started right after random finished). Log: `logs/EXP-004-curated-dev.log`.
      Check: `ssh -i ~/.ssh/azure_vm_key vmadmin@20.245.56.28 'tail -c 300 /mnt/vmadmin/projects/sarcasm/logs/EXP-004-curated-dev.log'`
- [ ] Compare the two on DEV Macro F1, pick a winning *variant* (document demo example IDs used) -- this is DEV-based selection, allowed
- [ ] Record in `EXPERIMENT_LOG.md`

### 4. M4 — Qwen3-4B structured reasoning (`configs/llm_reasoning_qwen_local.json`, EXP-005)

- [ ] Full DEV run
- [ ] Compare against EXP-002 (direct zero-shot) on DEV -- do NOT assume reasoning wins
- [ ] Record in `EXPERIMENT_LOG.md`

### 5. M5 — DSPy + local Qwen (EXP-006 Predict, EXP-007 BootstrapFewShot, EXP-008 MIPROv2)

- [ ] Smoke-test `LocalQwenLM` adapter with `dspy.Predict` on a handful of TRAIN/DEV examples (GPU must be free of the M2-M4 jobs first)
- [ ] EXP-006: `dspy.Predict`, DEV
- [ ] Before BootstrapFewShot/MIPROv2: estimate/record expected # LM calls for the planned budget
- [ ] EXP-007: `BootstrapFewShot` with a SMALL budget first (few demos) -- record wall-clock + DEV delta vs. EXP-006; expand only if it justifies the cost
- [ ] EXP-008: `MIPROv2`, `auto="light"` first -- record wall-clock + DEV delta; expand only if justified
- [ ] Compare EXP-006/007/008 vs. EXP-002 (manual zero-shot) on DEV
- [ ] Record in `EXPERIMENT_LOG.md`

### 6. M6 — Fine-tuned DeBERTa-v3-base (EXP-009), TRAIN+DEV only

- [ ] Tiny overfit/smoke test (`configs/transformer_deberta_v3_base_smoke.json`, 64 train / 32 dev) -- forward/backward pass, loss decreases, checkpoint save, DEV eval works
- [ ] Confirm `fp16=true` stability on Tesla M60 during the smoke test; fall back + document if unstable
- [ ] Full training run (`configs/transformer_deberta_v3_base.json`) -- single GPU (`CUDA_VISIBLE_DEVICES=0`), early stopping on DEV Macro F1
- [ ] If runtime permits: repeat the final chosen config across 2-3 seeds, report DEV mean/variance
- [ ] (Optional) `configs/transformer_roberta_base.json` as a second-encoder DEV comparison
- [ ] Record in `EXPERIMENT_LOG.md`

### 7. Cross-model DEV analysis

- [ ] Per-example disagreement table across M1/M2/M3/M4/M5/M6 DEV predictions (+ gold)
- [ ] Error analysis: false positives/negatives per model, patterns (rhetorical Qs, hyperbole, short/ambiguous, the 22 label-conflict rows)
- [ ] Confidence/low-confidence review where available (TF-IDF proba, DeBERTa softmax)

## PHASE 2 — Freeze + sealed TEST evaluation (only after Phase 1 fully checked off)

- [ ] Review complete DEV results for every method
- [ ] Select exactly ONE final configuration per method (M2, M3, M4, M5, M6), record why in `EXPERIMENT_LOG.md`, mark FROZEN with its config file path
- [ ] Evaluate M1 (already frozen from Stage A), M2, M3, M4, M5, M6 frozen configs on TEST -- once each
- [ ] No re-tuning after seeing TEST results
- [ ] Final cross-model TEST-based comparison table

## 8. Final writeup

- [ ] `PROJECT_SUMMARY.md` results table fully populated (no leftover `TBD` for anything actually run)
- [ ] Final recommendation: predictive performance vs. computational cost, separately
- [ ] `EXPERIMENT_LOG.md` Current Status section updated to reflect Stage B completion
- [ ] This checklist fully checked off

# Stage B Checklist — Resume Point

Purpose: if this session crashes/restarts, read this file first to see
exactly where things stood. Kept in sync after every milestone — a
checkbox is only ticked once genuinely done (not "started"). For full
detail/evidence behind any line here, see `EXPERIMENT_LOG.md` (the
authoritative, detailed audit trail — this file is just the map of it).
For final results, see `PROJECT_SUMMARY.md`.

**Last updated:** 2026-08-12, ~12:20 UTC. **RECOVERED after a VM restart
that wiped the ephemeral `/mnt` disk** (repo checkout, Python env, HF
model cache, and EXP-003's raw predictions all lost -- EXP-003's metrics
were not, already recorded in `EXPERIMENT_LOG.md`). Full incident detail:
`EXPERIMENT_LOG.md`, "VM restart -- `/mnt` ephemeral-disk data loss
incident and recovery." Environment fully recreated with the identical
pinned stack, `verify_gpu.py` and the classification test suite
(61/61) and a Qwen smoke test all re-passed. M2 remains DONE (Macro F1
0.6008). EXP-003 (M3-random) is being regenerated (deterministic rerun of
the identical frozen config, to restore its lost `predictions.csv`) as
step 1 of a fresh `scripts/run_m3_m4_chain.sh` (now git-tracked), followed
automatically by EXP-004 (M3-curated, restarted from scratch -- nothing
valid survived to resume from) and EXP-005 (M4 reasoning). All Stage A/B
work, including this file, is now committed to git so it no longer
depends on the VM's ephemeral disk or the local Mac's uncommitted working
tree.

### How to resume / check on the running chain

1. `ssh -i ~/.ssh/azure_vm_key vmadmin@20.245.56.28`, confirm hostname is
   `dpmlgpuNC6sv32025s-0003`.
2. `source /mnt/vmadmin/sarcasm-env/bin/activate && export HF_HOME=/mnt/vmadmin/huggingface`
3. `cd /mnt/vmadmin/projects/sarcasm`
4. Check chain progress: `tail -c 500 logs/m3_m4_chain.log` (or the
   per-step logs `logs/EXP-003-random-dev.log`, `logs/EXP-004-curated-dev.log`,
   `logs/EXP-005-reasoning-dev.log`). `ps -ef | grep run_experiment` to
   confirm it's still alive.
5. **After it finishes, pull results back immediately**:
   `bash scripts/sync_from_vm.sh` (run on the local Mac) -- this is now the
   standard step after every experiment, specifically to avoid repeating
   the EXP-003 loss (results previously only ever got pushed one
   direction, never pulled back until a planned shutdown).
6. If the chain died instead of finishing, `scripts/run_m3_m4_chain.sh` is
   git-tracked now (unlike its predecessor) -- just rerun it; each step
   only starts after the previous one exits 0 and the per-example disk
   cache makes any already-computed example near-instant on retry.

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

- [x] Random variant, full DEV -- **DONE (metrics).** 1340/1340, ~1h34m. Macro F1 0.5880,
      Accuracy 0.6328 -- *underperforms* M2 zero-shot (0.6008/0.6440). Investigated
      (demo balance, category uniformity, agreement-with-M2 check) -- confirmed
      genuine result, not a bug. See EXPERIMENT_LOG.md.
      Raw `predictions.csv` was lost in the 2026-08-12 `/mnt` wipe (metrics
      survived) -- **being regenerated now** via an identical deterministic
      rerun (step 1 of `scripts/run_m3_m4_chain.sh`).
- [ ] Curated variant, full DEV (`configs/llm_few_shot_curated_8_qwen_local.json`) -- **RESTARTING FROM SCRATCH**
      (previous partial run, 447/1340, and its cache did not survive the
      `/mnt` wipe -- nothing valid to resume from). Auto-runs as step 2 of
      `scripts/run_m3_m4_chain.sh`, right after EXP-003 regeneration finishes.
      Log: `logs/EXP-004-curated-dev.log`.
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

# Stage B Checklist — Resume Point

Purpose: if this session crashes/restarts, read this file first to see
exactly where things stood. Kept in sync after every milestone — a
checkbox is only ticked once genuinely done (not "started"). For full
detail/evidence behind any line here, see `EXPERIMENT_LOG.md` (the
authoritative, detailed audit trail — this file is just the map of it).
For final results, see `PROJECT_SUMMARY.md`.

## START HERE (next session) — read this section first, top to bottom

**Where things stand (2026-08-15, ~16:00 local VM time):** M1-M5 are all
DEV-evaluated and committed (M5 complete, EXP-008 Macro F1 0.6700 is the
current best of any method in Stage B). **M6 is in progress**: recovered
from a sixth `/mnt` wipe, resolved the DeBERTa-v3-base download blocker
(flagged but unresolved as of the 2026-08-14 session), fixed two more
real code bugs surfaced by the M6 smoke test (a `transformers==5.15.0`
`TrainingArguments` API change, and an fp16 NaN root-caused to the HF-hub
checkpoint's `pytorch_model.bin` itself being float16), and the smoke
test (`SMOKE-deberta`) now passes cleanly end-to-end with `fp16=true`.
Full detail: `EXPERIMENT_LOG.md`'s 2026-08-15 entries ("Sixth VM restart"
section onward). **EXP-009 (the full M6 training run) has not been
launched yet** -- that's the very next action, no further setup needed.

1. **Reconnect and check for another VM restart first** (this has now
   happened SIX times -- assume nothing): `ssh -i ~/.ssh/azure_vm_key
   vmadmin@20.245.56.28` (`ConnectTimeout=20`; if it times out completely,
   that itself is the finding -- report it, don't guess, but it's fine to
   just wait a few minutes and retry once, that resolved it before).
   Then `bash scripts/verify_kernel.sh` (fails loudly on a kernel/driver
   mismatch -- if it reports the wrong kernel despite the `grub-set-default`
   fix, check `cat /boot/grub/grubenv | grep saved_entry` to see if
   something reset it, then re-apply the fix from EXPERIMENT_LOG.md's
   2026-08-14 entry, step 7) and `ls /mnt/vmadmin` (if "No such file or
   directory", `/mnt` was wiped again -- run the full recovery procedure
   documented in `EXPERIMENT_LOG.md`'s VM-restart incident entries,
   verbatim; it's been used six times already and works every time.
   Note: `/mnt` may come back **root-owned** -- if `mkdir`/`chown` on
   `/mnt/vmadmin` gives "Permission denied", `sudo mkdir -p
   /mnt/vmadmin/{projects,sarcasm-env,huggingface} && sudo chown -R
   vmadmin:vmadmin /mnt/vmadmin` first).
2. **If the environment needed rebuilding**, after it's back: re-run
   `scripts/verify_gpu.py`, `pytest tests/test_classification_*.py`
   (expect 61/61), and the Qwen zero-shot smoke test (`--limit 20`) --
   should reproduce accuracy 0.25 / macro F1 0.20 exactly, same as every
   prior rebuild (seven confirmations now). If it doesn't match, STOP and
   report -- that would mean something about the environment actually
   changed. Remember `optuna` is now part of `environment_stage_b.txt`'s
   pinned freeze -- installing from that file covers it, no separate step
   needed. **Also redo the DeBERTa safetensors conversion** (the fix from
   the sixth-restart recovery, `/mnt`-ephemeral so it doesn't survive a
   wipe): `snapshot_download('microsoft/deberta-v3-base')`, then
   `torch.load` the cached `pytorch_model.bin` (`weights_only=True`),
   `safetensors.torch.save_file(...)` it as `model.safetensors` into the
   *same* snapshot directory -- ~5s, see EXPERIMENT_LOG.md's 2026-08-15
   entry for the exact snippet. Without this, `from_pretrained(...,
   use_safetensors=True)` fails with the `check_torch_load_is_safe`
   `ValueError` again.
3. **Re-arm background monitoring** (Monitor-tool tasks do NOT persist
   across sessions -- these need to be relaunched fresh every session,
   even if the VM itself didn't restart):
   - Cache backup, every 5 min (crash-resume safety net):
     `cd "<repo root>" && while true; do bash scripts/sync_cache_from_vm.sh; sleep 300; done`
   - 15-min progress heartbeat (only if the user still wants them --
     confirmed wanted historically; ask if unsure after a long gap):
     tail the relevant experiment's log + `ps -ef | grep run_experiment`
     over SSH, `sleep 900` loop.
   - State-change watcher on whichever log is about to run.
4. **Launch EXP-009** (`configs/transformer_deberta_v3_base.json`, full
   M6 training run) -- smoke test already passed, config already has
   `fp16=true` (validated stable) and `use_safetensors=true`; `finetune.py`
   now forces `dtype=torch.float32` on load, so no further setup is
   needed. Single GPU: `CUDA_VISIBLE_DEVICES=0`. After it finishes:
   validate + `sync_from_vm.sh` + record in `EXPERIMENT_LOG.md` + check
   off "6. M6" below, per the standing rhythm in item 7.
5. **After M6: cross-model DEV disagreement analysis** (section "7"),
   then **Phase 2** (freeze one config per method, unseal TEST, evaluate
   each once), then the **final `PROJECT_SUMMARY.md` writeup** (section
   "8"). Follow this file's numbered sections in order from here.
   **Note on Phase 2 timing for M5:** whichever DSPy variant gets frozen,
   running it on TEST means *rebuilding the DSPy program from scratch*
   (`build_program()` always recompiles -- there's no "load saved
   compiled program" path in the current code), not just loading
   `compiled_program.json` and evaluating -- so freezing EXP-007 or
   EXP-008 costs a full re-run (~2h48m / ~2h29m) on TEST, not a quick
   eval. Budget for this when picking which M5 config to freeze.
6. **After Stage B fully completes** (or if there's a natural lull with
   the GPU busy and nothing else to validate): resume the **web app**
   (`web/`, currently a fully-built and tested but *dormant* app -- see
   `web/README.md`). The one remaining connective step is writing
   `results/frozen_configs.json` once Phase 2 freezes configs -- that
   single file flips methods from "not frozen yet" to serving real
   predictions, no code changes needed. Was explicitly paused mid-Stage-B
   per user request ("don't do anything in the meantime") -- safe to
   resume once Stage B experiments are running/paused and there's nothing
   else productive to do, or once Stage B is fully done.
7. **After every experiment finishes** (M6's runs, Phase 2's TEST runs,
   anything else): validate (quality checks -- n_examples, dup/missing
   IDs, gold label distribution, category-uniformity + agreement checks
   if the result looks skewed), `bash scripts/sync_from_vm.sh` immediately
   (not at the end of the session), record it in `EXPERIMENT_LOG.md` with
   the same level of detail as every entry so far, update this checklist's
   checkboxes, commit, push. This is now a well-established rhythm --
   just keep doing it exactly the same way for every remaining experiment.

**Durability infra now in place** (all git-tracked, all already exist --
don't recreate, just use): `scripts/verify_kernel.sh` (startup guard),
`scripts/sync_to_vm.sh` (push code, excludes `node_modules`/`.next`/
`results`/`data/llm_cache`), `scripts/sync_from_vm.sh` (pull `results/`,
`logs/`, `models/`), `scripts/sync_cache_from_vm.sh` (pull `data/llm_cache/`
specifically, safe to run continuously), `scripts/run_m3_m4_chain.sh`
(done, for reference), `scripts/run_m5_chain.sh` (ready to launch).

---

**Last updated:** 2026-08-12, ~13:35 UTC. **RECOVERED after a SECOND VM
restart** (~13:22 UTC, cause unknown -- not a deliberate reboot this
time; VM went unreachable mid-EXP-003 at 56%, came back healthy per user
confirmation) **that wiped the ephemeral `/mnt` disk again.** Same
recovery procedure as the first incident, re-run verbatim and confirmed
working a second time -- see EXPERIMENT_LOG.md, "Second VM restart --
`/mnt` wiped again." Nothing new was scientifically lost (EXP-003 hadn't
reached its completed-artifact point either time). `run_m3_m4_chain.sh`
relaunched from scratch.

**Original recovery (first incident, for reference):** RECOVERED after a VM restart
that wiped the ephemeral `/mnt` disk (repo checkout, Python env, HF
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
2. **Before anything else, run the kernel/driver startup guard:**
   `cd /mnt/vmadmin/projects/sarcasm && bash scripts/verify_kernel.sh`.
   It fails loudly (non-zero exit) if the machine booted into anything
   other than the one verified-working kernel (`6.8.0-1029-azure`) or if
   `nvidia-smi` doesn't work -- do not proceed with any GPU work if it
   fails; see the script's comments and `EXPERIMENT_LOG.md` for the
   recovery procedure. `scripts/run_m3_m4_chain.sh` also runs this guard
   automatically as its first step.
3. `source /mnt/vmadmin/sarcasm-env/bin/activate && export HF_HOME=/mnt/vmadmin/huggingface`
4. `cd /mnt/vmadmin/projects/sarcasm`
5. **If recovering after a fresh `/mnt` wipe, restore the LLM disk cache
   BEFORE relaunching anything**, so already-computed examples resume
   instantly instead of being recomputed from scratch:
   `rsync -av -e "ssh -i ~/.ssh/azure_vm_key" data/llm_cache/ vmadmin@20.245.56.28:/mnt/vmadmin/projects/sarcasm/data/llm_cache/`
   (run on the local Mac). This only works if `scripts/sync_cache_from_vm.sh`
   was running periodically before the crash -- see below.
6. Check chain progress: `tail -c 500 logs/m3_m4_chain.log` (or the
   per-step logs `logs/EXP-003-random-dev.log`, `logs/EXP-004-curated-dev.log`,
   `logs/EXP-005-reasoning-dev.log`). `ps -ef | grep run_experiment` to
   confirm it's still alive.
7. **After it finishes, pull results back immediately**:
   `bash scripts/sync_from_vm.sh` (run on the local Mac) -- this is now the
   standard step after every experiment, specifically to avoid repeating
   the EXP-003 loss (results previously only ever got pushed one
   direction, never pulled back until a planned shutdown).
8. If the chain died instead of finishing, `scripts/run_m3_m4_chain.sh` is
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

## Full item-by-item audit (2026-08-12, ~12:35 UTC)

Explicit status for every checklist item below, per an instruction to
resume Stage B end-to-end rather than stopping after the currently
running chain. Legend: DONE / IN PROGRESS / BLOCKED / NOT STARTED / N/A.

| Section | Item | Status |
|---|---|---|
| 0 | SSH access | DONE |
| 0 | Repo synced | DONE (re-synced post-recovery) |
| 0 | Remote env verified | DONE (re-verified post-recovery, identical pinned stack) |
| 0 | `verify_gpu.py` | DONE (re-run post-recovery) |
| 0 | `environment_stage_b.txt` | DONE -- refreshed to the recovered environment (Python 3.10.12 vs. 3.10.11, otherwise identical) |
| 0 | Qwen smoke test | DONE (re-run post-recovery, exact metric reproduction) |
| 0 | Configs repointed to local Qwen | DONE |
| 0 | DSPy adapter written | DONE (code); adapter itself NOT STARTED (smoke test) -- blocked on GPU, occupied by the M3/M4/M5 chain |
| 0 | DeBERTa loading fixed | DONE (code); execution NOT STARTED -- blocked on GPU |
| 0 | Test suite | DONE (61/61 classification tests, both local Mac and VM, re-verified post-recovery) |
| 1 | M1 TF-IDF | DONE, FROZEN (Stage A, untouched) |
| 2 | M2 zero-shot | DONE, DEV-EVALUATED |
| 3 | M3-random (EXP-003) | IN PROGRESS -- regenerating lost `predictions.csv` via identical deterministic rerun; metrics already DEV-EVALUATED and unaffected |
| 3 | M3-curated (EXP-004) | NOT STARTED -- queued as chain step 2, restarting from scratch (nothing valid survived to resume) |
| 3 | M3 variant comparison + record | BLOCKED on EXP-004 completing |
| 4 | M4 reasoning (EXP-005) | NOT STARTED -- queued as chain step 3 |
| 5 | M5 DSPy (EXP-006/007/008) | NOT STARTED -- blocked on GPU (M3/M4/M5-chain must finish first; project's own resource-management rule is no concurrent GPU-heavy jobs) |
| 6 | M6 DeBERTa (EXP-009) | NOT STARTED -- blocked on GPU, same reason |
| 7 | Cross-model DEV analysis | NOT STARTED -- blocked on M3-M6 predictions existing. Code already implemented and ready (`src/classification/evaluation/error_analysis.py`, pairwise disagreement/agreement-rate tooling) |
| Phase 2 | Freeze + TEST eval | NOT STARTED -- correctly blocked, Phase 1 incomplete. TEST remains fully sealed |
| 8 | Final writeup | NOT STARTED -- blocked on all of the above |

**Plan for the rest of Stage B (this session, autonomous):** let the
currently running chain (EXP-003 regen -> EXP-004 -> EXP-005) finish
(each step syncs back to the local Mac + gets recorded as it completes),
then immediately continue, without waiting for confirmation, to: M5 DSPy
smoke test -> EXP-006 -> EXP-007 (small budget) -> EXP-008 (`auto="light"`)
-> M6 DeBERTa smoke test -> EXP-009 full run -> (seeds, if time permits)
-> cross-model DEV disagreement analysis -> Phase 2 freeze -> sealed TEST
evaluation (each frozen config, once) -> final `PROJECT_SUMMARY.md`
writeup. Progress updates every ~15 minutes; will stop only for a genuine
blocker (credentials, a destructive action, or a methodological question
that would change the study).

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

- [x] Random variant, full DEV -- **DONE, predictions.csv restored.** 1340/1340,
      ~1h35m (3rd attempt, survived both VM restarts this time). Macro F1
      0.5880, Accuracy 0.6328 -- *underperforms* M2 zero-shot (0.6008/0.6440).
      Investigated (demo balance, category uniformity, agreement-with-M2
      check) -- confirmed genuine result, not a bug. See EXPERIMENT_LOG.md.
      Regenerated metrics/confusion-matrix are byte-for-byte identical to the
      original pre-wipe run (deterministic reproduction re-confirmed) --
      pulled back to the local Mac and committed, no longer at risk from
      another `/mnt` wipe.
- [x] Curated variant, full DEV -- **DONE.** 1340/1340, ~1h22m. Macro F1
      0.5011, Accuracy 0.5821 -- underperforms *both* zero-shot (M2) and
      random few-shot (EXP-003); most sarcastic-skewed result yet (90.4%
      predicted sarcastic). Investigated (category uniformity, demo
      composition, agreement with EXP-002/EXP-003) -- confirmed genuine,
      not a bug. See EXPERIMENT_LOG.md.
- [x] Compare the two on DEV Macro F1, pick a winning *variant* -- **random
      (EXP-003) wins** (0.588 vs. 0.501 Macro F1). Demo example IDs for
      both variants recorded in EXPERIMENT_LOG.md. Note: zero-shot (M2)
      still beats both few-shot variants outright -- final M2-M4 candidate
      selection happens at Phase 2, after EXP-005.
- [x] Record in `EXPERIMENT_LOG.md` -- done, full detail + comparison table.
- [ ] Compare the two on DEV Macro F1, pick a winning *variant* (document demo example IDs used) -- this is DEV-based selection, allowed
- [ ] Record in `EXPERIMENT_LOG.md`

### 4. M4 — Qwen3-4B structured reasoning (`configs/llm_reasoning_qwen_local.json`, EXP-005)

- [x] Full DEV run -- **DONE.** 1340/1340, ~1h02m. Macro F1 0.5796, Accuracy 0.6276.
- [x] Compare against EXP-002 (direct zero-shot) on DEV -- **zero-shot wins**
      (0.6008 vs. 0.5796 Macro F1); reasoning barely moved predictions at all
      (94.6% agreement with EXP-002, highest of any variant) -- confirmed
      genuine, not a bug.
- [x] Record in `EXPERIMENT_LOG.md` -- done, full detail + final M2-M4 comparison table.

**M2-M4 development complete.** Zero-shot (EXP-002) is the best of all four
manual-prompt variants on DEV. **Paused here per explicit user request** --
M5/M6 not started automatically; awaiting go-ahead.

### 5. M5 — DSPy + local Qwen (EXP-006 Predict, EXP-007 BootstrapFewShot, EXP-008 MIPROv2)

**M5 is COMPLETE (2026-08-14, ~15:10 local VM time).** All three variants run, quality-checked, recorded, committed. **This is now the best result of any method in Stage B.**

- [x] Smoke-test `LocalQwenLM` adapter with `dspy.Predict` on a handful of TRAIN/DEV examples -- **PASSED**, 5/5.
- [x] EXP-006: `dspy.Predict`, DEV -- **DONE.** 1340/1340, 56m49s. Macro F1 0.6619, Accuracy 0.6799.
- [x] EXP-007: `BootstrapFewShot` (`max_bootstrapped_demos=4`, `max_labeled_demos=8`) -- **DONE.** 1340/1340, 2h48m24s. Macro F1 0.6406, Accuracy 0.6664 -- slightly underperforms EXP-006.
- [x] EXP-008: `MIPROv2`, `auto="light"` -- **DONE.** 1340/1340, ~2h29m total (optimization + final eval). **Macro F1 0.6700, Accuracy 0.6843 -- new best of any method in Stage B**, beating EXP-006 by +0.0081. Winning config: default instruction + a compact 4-demo few-shot set. Also fixed a real code bug along the way (`signatures.py`'s `from __future__ import annotations` broke `MIPROv2.with_instructions()` -- see EXPERIMENT_LOG.md, 2026-08-14).
- [x] Compare EXP-006/007/008 vs. EXP-002 (manual zero-shot) on DEV -- done, see EXPERIMENT_LOG.md's EXP-008 entry.
- [x] Record in `EXPERIMENT_LOG.md` -- done.

### 6. M6 — Fine-tuned DeBERTa-v3-base (EXP-009), TRAIN+DEV only

- [x] Tiny overfit/smoke test (`configs/transformer_deberta_v3_base_smoke.json`, 64 train / 32 dev) -- forward/backward pass, checkpoint save, DEV eval all confirmed working, no NaN. (Loss doesn't visibly decrease in 24 steps on 64 examples -- expected, not a bug; not enough signal/steps for a base encoder, see EXPERIMENT_LOG.md 2026-08-15.)
- [x] Confirm `fp16=true` stability on Tesla M60 during the smoke test -- **stable after fixing a real bug**: `finetune.py` now forces `dtype=torch.float32` on model load (the HF-hub checkpoint's `pytorch_model.bin` is itself fp16, which caused the earlier "unscale FP16 gradients" crash and NaN losses). See EXPERIMENT_LOG.md, 2026-08-15.
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

# Stage B Checklist — Resume Point

Purpose: if this session crashes/restarts, read this file first to see
exactly where things stood. Kept in sync after every milestone — a
checkbox is only ticked once genuinely done (not "started"). For full
detail/evidence behind any line here, see `EXPERIMENT_LOG.md` (the
authoritative, detailed audit trail — this file is just the map of it).
For final results, see `PROJECT_SUMMARY.md`.

## START HERE (next session) — read this section first, top to bottom

**Where things stand (2026-08-16, ~18:31 local VM time):** Phase 1 and
Phase 2 are now **both fully complete**. Every one of the 6 methods has a
sealed, one-shot, frozen-configuration TEST score:

| Method | TEST Macro F1 |
|---|---:|
| M1 (TF-IDF+LR) | 0.7403 |
| M2 (Qwen zero-shot) | 0.6005 |
| M3 (Qwen few-shot random) | 0.5947 |
| M4 (Qwen reasoning) | 0.5758 |
| M5 (DSPy MIPROv2) | 0.6681 |
| M6 (DeBERTa-v3-base, fine-tuned) | **0.8209 (best)** |

M5 finished last (2026-08-16, 16:23-18:31, after restarting from scratch
following an unplanned VM outage -- see EXPERIMENT_LOG.md's "unplanned VM
outage mid-M5-run" and "M5 (EXP-008-TEST, MIPROv2) -- DONE" entries for
full detail, including the recovered exact winning prompt in
`results/EXP-008-TEST/compiled_program.json`). `sync_from_vm.sh` already
run -- all results are on the local Mac and durable; the VM is **no
longer needed** for anything in the immediate next steps (comparison
table + `PROJECT_SUMMARY.md` are pure local analysis; the web app's
adapter-code fixes are also GPU-free, per `web/README.md`'s
mocked-model-loading test setup) -- safe to shut down now unless/until a
live Qwen/DSPy web demo is specifically wanted later (see the parked
OpenRouter decision a few paragraphs below).

**Next, in order:**
1. Final cross-model TEST comparison table (extends the existing DEV-only
   cross-model analysis in EXPERIMENT_LOG.md with the now-complete TEST
   numbers).
2. `PROJECT_SUMMARY.md` writeup (section "8") -- **must include the
   explicit content the user required this session**, detailed further
   down in this file: full train/dev/test split methodology, a
   per-method TRAIN/DEV/TEST usage table, and M5's exact optimized prompt
   quoted from `compiled_program.json`.
3. The web app (`web/`) -- two adapter fixes (`qwen_adapter.py`,
   `dspy_adapter.py`) plus the still-open, explicitly parked OpenRouter
   decision (below) about whether/how the live demo should work without
   requiring the VM.

**Reference runbook below (steps 1-8) -- NOT "next steps" anymore now
that Phase 2 is done and the VM isn't needed for the immediate work.**
Keep this for whenever the VM is needed again (e.g. if the OpenRouter
decision above lands on "keep needing the VM for live Qwen/DSPy demo," or
any future re-run/extension of Stage B). It predates Phase 2 completing
and its "next" language refers to that now-finished work, not the current
resume point -- go by the numbered list above for what to actually do
next.

1. **Reconnect and check for another VM restart first** (this has now
   happened SEVEN times, and an eighth is expected given the VM was
   deliberately shut down this time): `ssh -i ~/.ssh/azure_vm_key
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
4. **Finish Phase 2's TEST evaluation -- only M5 left:** `bash
   scripts/run_phase2_test_chain.sh` (launch detached, e.g. `nohup ...
   < /dev/null > logs/phase2-test-chain.log 2>&1 & disown` -- plain `&`
   without `< /dev/null` has left the SSH command hanging on stdin
   before). The script is resume-aware (skips any step whose
   `results/<experiment_id>/metrics.json` already exists), so it will
   skip M2/M3/M4/M6 automatically and go straight to M5. Or run M5 alone
   directly: `python -m src.classification.run_experiment --config
   configs/EXP-008-TEST.json`. Budget: **~2h29m** (MIPROv2, full
   recompile -- no "load compiled program" path exists, this is the last
   known TEST-run cost in the project). After it finishes: validate +
   `sync_from_vm.sh` + record in `EXPERIMENT_LOG.md` (same rhythm as
   every prior experiment).
5. **After all of Phase 2's TEST runs are in:** final cross-model
   TEST-based comparison table, then the **final `PROJECT_SUMMARY.md`
   writeup** (section "8"). Follow this file's numbered sections in order
   from here.

   **Explicit user requirement (2026-08-16) for what `PROJECT_SUMMARY.md`
   must cover -- do not skip when writing it:**
   - **Full train/dev/test split methodology, written out and explained**,
     not just referenced: exact sizes (TRAIN 6,706 / DEV 1,340 / TEST
     1,340 -- 71.4/14.3/14.3%, target was 70/15/15, small deviation because
     the split is grouped), `StratifiedGroupKFold` grouped by
     `dup_group_id` (so near-duplicate text can't land in both TRAIN and
     TEST -- no leakage) and stratified by label, fixed seed
     (`src/classification/data/make_splits.py`,
     `config/classification_settings.py`'s `train_frac`/`dev_frac`/`test_frac`).
   - **A per-method table of exactly how TRAIN/DEV/TEST were each used**,
     since it differs by method type and that difference should be
     explained, not left implicit: M1/M6 train on TRAIN + select via DEV;
     M3 selects its few-shot demos from TRAIN; M2/M4 have no TRAIN role at
     all (zero-shot/reasoning has nothing to fit or select -- this is by
     design, not an oversight, and the writeup should say so explicitly);
     M5 samples 150 from TRAIN for bootstrapping and 100 from DEV as
     MIPROv2's optimization valset (confirmed hardcoded to DEV even during
     the TEST run, so TEST never leaks into optimization). Every method's
     TEST number is a single, one-shot, frozen-configuration evaluation --
     state this explicitly as the sealing methodology, not just show the
     numbers.
   - **M5's actual optimized prompt, documented concretely, not just
     "MIPROv2 found something better."** **CORRECTION (2026-08-16, after
     EXP-008-TEST actually finished): the assumption below that the
     compiled program is never persisted was WRONG** -- re-reading
     `run_dspy_experiment` past line 190 (missed on first read) shows it
     already calls `program.save(str(Path(out_dir) / "compiled_program.json"))`
     whenever `optimizer != "predict"`. `results/EXP-008-TEST/compiled_program.json`
     exists and contains the full winning program: instructions text
     (matches what was independently found by reading the log, confirming
     both methods agree) **and the exact 4 few-shot demos actually used**:
     *"Dude, go jack off to your god somewhere else. We don't need to see
     it. emoticonXKill"* (sarcastic, bootstrapped), *"i'm not disputing the
     numbers... don't you find it the least bit odd that it's been 30
     years since a president received a plurality? waxy"* (not_sarcastic,
     bootstrapped), *"Nope. However, he does get to pay child support if
     he gets caught."* (sarcastic, labeled), *"you really believed me? wow!
     i never knew i had such power ;)"* (sarcastic, labeled). Winning
     instruction: *"Classify the given sentence as \"sarcastic\" or
     \"not_sarcastic\" based on linguistic cues such as irony,
     exaggeration, contradiction, or mocking tone."* **`PROJECT_SUMMARY.md`
     should quote `compiled_program.json` directly (it's the authoritative
     source) rather than the log cross-referencing method used before this
     was found** -- both happen to agree here, but the JSON file is the
     ground truth going forward. **No web-app action item needed** -- this
     already works for every DSPy run/optimizer other than `predict`, no
     code change required.
6. **Resume the web app** (`web/`, fully built and tested but still
   *dormant*). `results/frozen_configs.json` already exists (written
   2026-08-15) so all 6 methods now report FROZEN -- but **two real gaps
   need fixing first**, both already flagged in code, not silently wrong:
   `web/backend/app/adapters/qwen_adapter.py`'s `DEFAULT_CONFIG_PATHS["qwen_few_shot"]`
   is hardcoded to the curated config, not the frozen random one (`config_path`
   from the registry is currently unread); `web/backend/app/adapters/dspy_adapter.py`
   only ever builds the unoptimized `dspy.Predict` baseline, not the frozen
   MIPROv2 program (its own docstring already anticipated this exact
   situation -- needs extending to load a compiled program, matching
   whatever DSPy 3.3.0's save/load API looks like). See EXPERIMENT_LOG.md's
   "PHASE 2 -- Config freeze" entry for detail. Do this once Phase 2's TEST
   runs are done (or sooner, if there's a natural lull).

   **Open decision, parked here per explicit user instruction (2026-08-16)
   -- resolve before/while doing this step, not before:** the user asked
   whether an OpenRouter API key (free tier) could let the live web demo's
   Qwen/DSPy methods run without needing the always-on GPU VM. Checked:
   OpenRouter does **not** offer the exact frozen checkpoint
   (`Qwen/Qwen3-4B-Instruct-2507`) for free -- only `qwen/qwen3-4b:free`,
   which is a genuinely different checkpoint (the earlier dual-mode
   thinking/non-thinking Qwen3-4B, not the instruct-only 2507 release this
   project's M2-M5 results are all based on). The `-Instruct-2507` line
   does exist on OpenRouter but only at 30B/235B sizes, not 4B. So a free
   OpenRouter key would mean the live web demo runs a **different model**
   than the one the frozen TEST scores describe -- not just "the same
   model via a different API." User was informed of this and chose to
   defer the decision (three options discussed: (a) live-demo-only via the
   free substitute model, clearly labeled as not the frozen benchmark
   model; (b) treat it as a genuinely new side-by-side comparison and
   re-run M2-M5 against it; (c) skip API entirely, keep requiring the VM
   for live Qwen/DSPy demo) until Phase 2 + `PROJECT_SUMMARY.md` are done.
   **Do not implement any OpenRouter adapter path without re-confirming
   which of these three the user wants first.**
7. **After every experiment finishes** (Phase 2's TEST runs, anything
   else): validate (quality checks -- n_examples, dup/missing
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
- [x] Full training run (`configs/transformer_deberta_v3_base.json`) -- single GPU (`CUDA_VISIBLE_DEVICES=0`), early stopping on DEV Macro F1 -- **DONE (EXP-009).** ~22min. Best checkpoint epoch 2: **Macro F1 0.8254, Accuracy 0.8254 -- new best of any method in Stage B by a huge margin** (prior best EXP-008 MIPROv2: 0.6700, +0.1554 absolute). Epoch 3+ regressed (overfitting), early stopping (patience=2) restored the epoch-2 checkpoint correctly.
- [ ] If runtime permits: repeat the final chosen config across 2-3 seeds, report DEV mean/variance -- not done (margin over other methods is wide enough that this wasn't judged necessary before Phase 2; revisit if Phase 2 needs a variance estimate)
- [ ] (Optional) `configs/transformer_roberta_base.json` as a second-encoder DEV comparison
- [x] Record in `EXPERIMENT_LOG.md` -- done, see the "EXP-009" entry, 2026-08-15.

### 7. Cross-model DEV analysis

- [x] Per-example disagreement table across M1/M2/M3/M4/M5/M6 DEV predictions (+ gold) -- done, `results/cross_model_dev_analysis.csv`. Key finding: LLM-based methods (M2-M5) cluster at 86-95% mutual agreement; M1/M6 (trained-on-labels methods) agree with each other (82.3%) far more than with any LLM method (58-71%).
- [x] Error analysis: false positives/negatives per model, patterns (rhetorical Qs, hyperbole, short/ambiguous, the 22 label-conflict rows) -- done. Headline finding: every LLM method is heavily FP-skewed (over-predicts sarcastic, e.g. M5-MIPROv2 FP=353/FN=70) while M1/M6 are balanced (M6 FP=112/FN=122) -- explains most of the gap. Worst category for nearly every method is HYP (hyperbole); only 2/22 label-conflict rows fall in DEV, too few for a firm conclusion.
- [x] Confidence/low-confidence review where available (TF-IDF proba, DeBERTa softmax) -- done. Both M1 and M6 are reasonably well-calibrated (accuracy rises monotonically with confidence bucket); M6's confidence is far more concentrated near 1.0.

See `EXPERIMENT_LOG.md`'s "Cross-model DEV analysis" entry (2026-08-15) for full detail.

## PHASE 2 — Freeze + sealed TEST evaluation (only after Phase 1 fully checked off)

- [x] Review complete DEV results for every method -- done, see cross-model DEV analysis above.
- [x] Select exactly ONE final configuration per method (M2, M3, M4, M5, M6), record why in `EXPERIMENT_LOG.md`, mark FROZEN with its config file path -- done, see EXPERIMENT_LOG.md's "PHASE 2 -- Config freeze" entry and `results/frozen_configs.json`. Every method freezes its DEV-best: M2=EXP-002 (only candidate), M3=EXP-003 (random beats curated), M4=EXP-005 (only candidate), M5=EXP-008 (MIPROv2, best of 3, accepting its TEST-time recompile cost), M6=EXP-009 (only candidate, and overall best -- `production_model`).
- [x] Evaluate M1 (already frozen from Stage A), M2, M3, M4, M5, M6 frozen configs on TEST -- once each -- **DONE, ALL 6.** M1 0.7403, M2 0.6005, M3 0.5947, M4 0.5758, M5 **0.6681** (finished 2026-08-16 ~18:31, after restarting from scratch following an unplanned VM outage), M6 **0.8209 (best)**. See EXPERIMENT_LOG.md's "PHASE 2 -- Sealed TEST evaluation" and "M5 (EXP-008-TEST, MIPROv2) -- DONE" entries.
- [x] No re-tuning after seeing TEST results -- confirmed: no config was touched or re-run after any TEST score was seen.
- [x] Final cross-model TEST-based comparison table -- `results/cross_model_test_analysis.csv`, written up in `PROJECT_SUMMARY.md` Section 7.

## 8. Final writeup

- [x] `PROJECT_SUMMARY.md` results table fully populated (no leftover `TBD` for anything actually run) -- full rewrite 2026-08-16: final 6-method TEST table, split methodology (3.1), per-method TRAIN/DEV/TEST usage table (3.2), M5's exact optimized prompt quoted from `compiled_program.json` (6.1), cross-model comparison (7), error analysis (8), conclusions (9), updated limitations (10), recommended production approach (11), future work (12).
- [x] Final recommendation: predictive performance vs. computational cost, separately -- `PROJECT_SUMMARY.md` Section 11 (M6 recommended: best on both axes; M1 as the CPU-only fallback).
- [x] `EXPERIMENT_LOG.md` Current Status section updated to reflect Stage B completion -- see this file's own "START HERE" section and EXPERIMENT_LOG.md's latest entries.
- [x] This checklist fully checked off (remaining open item is the web app, tracked separately below -- not part of Stage B's core experimental work).

## 9. Web app (post-Stage-B, in progress)

- [x] `qwen_adapter.py`'s few-shot config now reads `config_path` from `results/frozen_configs.json` via `frozen_registry.frozen_config_path()` instead of a hardcoded curated-config guess -- fixed 2026-08-16, covered by `test_qwen_few_shot_uses_registry_config_not_hardcoded_curated`.
- [x] `dspy_adapter.py` now loads the actual frozen MIPROv2 compiled program (`results/<experiment_id>/compiled_program.json`, via `dspy.Predict.load()`) instead of reconstructing an unoptimized `Predict` baseline -- fixed 2026-08-16, covered by `test_dspy_adapter_loads_frozen_compiled_program` and its missing-file fallback test. Loads `EXP-008`'s program (the DEV-frozen run) by default, matching `frozen_experiment_id("dspy")`.
- [x] `requirements-classification.txt`: added missing `sentencepiece` (needed by `DebertaV2Tokenizer`, was in the VM's pinned freeze but never added here -- a real gap for any machine other than the VM, e.g. this local Mac).
- [x] Backend test suite updated to match the new reality (all 6 methods now genuinely FROZEN, not a hypothetical future state) -- `test_api.py`'s status assertions now expect UNAVAILABLE (not NOT_FROZEN_YET) for GPU-only methods on a no-GPU dev machine, `test_predict_valid_sentence` follows `production_model=deberta`. All 20 backend tests pass.
- [ ] **Open, real blocker found while fixing the above, NOT silently worked around:** the DeBERTa checkpoint (`models/EXP-009/best_checkpoint/`) was saved by `transformers==5.15.0` (the Azure VM's pinned freeze) but this repo's own `requirements.txt` caps `transformers<5.0.0` -- loading the checkpoint's tokenizer fails under the repo's pinned version, both `use_fast=True` (a version-specific `AttributeError` in `transformers==4.57.6`'s `DebertaV2TokenizerFast.__init__`) and `use_fast=False` (the checkpoint only saved `tokenizer.json`, the fast-tokenizer artifact -- no raw `.model` SentencePiece vocab file for the slow tokenizer to load). **Confirmed the fix**: installing `transformers==5.15.0` in an isolated scratch venv loads both fast and slow tokenizers cleanly, identical token IDs either way. **Not applied to `requirements.txt` yet** -- this is a shared, repo-wide pin that could also affect the pre-existing sarcasm-interpretation pipeline (untested here), so it needs the user's decision, not a unilateral bump. Until resolved, `deberta` genuinely reports `UNAVAILABLE` on this local Mac (correct, honest behavior given the real environment -- not a bug in the adapter).
- [ ] The OpenRouter live-demo decision, parked earlier in this file (see the "Open decision" note under the old numbered step 6) -- still unresolved, still requires the user's choice among the three options listed there before any adapter code changes toward it.
- [ ] Frontend: no changes made this session -- `web/frontend/` still needs to be pointed at the corrected backend behavior if any UI text assumed the old NOT_FROZEN_YET-everywhere state (not verified either way this session).

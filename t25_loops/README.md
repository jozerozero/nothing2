# T25 Safe-Tail RW25 with the complete native G5SC backbone

This v2 deployment replaces the rejected regression-backbone gated-encoder port. It copies and hashes every original G5SC model module, retaining its native column/row stack, support-conditioned shared-depth gate, attention, GELU, biased LayerNorm, QASSMax defaults, and original Muon/scheduler implementation. A thin task adapter supplies the frozen T25 generator, continuous support labels and the 999-quantile pinball regression task. Historical `gated_encoder.py` and old launchers here are not deployed or executed.

Loop3 and Loop4 each reuse the same 12 ICL blocks. Both start fresh with seed 2026080101, global batch 1024/microbatch 2, LR 6e-4, Muon momentum .95, cautious weight decay .01, cosine warmup .02 and floor 0. AMP and GradScaler are enabled; recompute is disabled (threshold 20000). All 1000 checkpoints, every 25 steps through 25000, are retained per arm.

T25 data remains graph_scm, fixed 4096 rows, RW-SAMPLE50 probability .25, group size 4, frozen train-only profile and Safe-Tail mapping. Four DataLoader workers use one internal generation job each, avoiding daemon-nested multiprocessing. E4 cross-table and TL curriculum patches are disabled. `PYTHONPATH` is exactly the new stage and its complete source tree; no shared package is installed.

Each arm requests faculty/bgqos, 8 nodes × 8 GPUs, 128 CPUs and 2 TiB per node, three days, Nice 0, no requeue and no dependencies. Historical full-G5 exclusions plus nodes 193, 195, 216, 228, 287 and 296 are retained. Default stage is `stage/t25_fullg5sc_loop34_bg64_20260912_v2`; a fresh explicitly versioned stage can be supplied after a failed preflight. Existing stages, checkpoints and other jobs are never overwritten.

Preparation requires explicit paths and an account. It does not submit:

```bash
python prepare_deployment.py --g5-source ORIGINAL_G5_SOURCE \
  --t25-source FROZEN_ADAPTED_T25_SOURCE --t25-prior-overlay FROZEN_ADAPTED_T25_SOURCE \
  --profile FROZEN_T25_PROFILE --profile-audit FROZEN_T25_PROFILE_AUDIT \
  --account USER_APPROVED_ACCOUNT
```

Run the real native identity, pinball/backward, generator mix-0/mix-1 and trainer-step smoke with the same Python environment, saving `native_identity_smoke.json` in the new stage:

```bash
python NEW_STAGE/test_identity.py --source NEW_STAGE/source \
  --regression-target-profile NEW_STAGE/artifacts/profile.json \
  --device cpu --receipt NEW_STAGE/native_identity_smoke.json
python NEW_STAGE/validate_launch.py --stage NEW_STAGE --passes 3 --account USER_APPROVED_ACCOUNT
python NEW_STAGE/validate_launch.py --stage NEW_STAGE --passes 4 --account USER_APPROVED_ACCOUNT
```

Only after the user explicitly chooses the account, run `submit_pair.py --stage NEW_STAGE --account USER_APPROVED_ACCOUNT`. There is no default account. The script checks account association, rejects MaxSubmitJobs=0, parses the actual trainer arguments, checks source/profile/smoke identity, checks duplicate jobs and fresh output roots, and runs `sbatch --test-only` for both. An exclusive durable intent prevents repeated submission. Both jobs are submitted held; resources, exclusions, paths and absence of dependencies must pass before either is released. Failures preserve exact state and leave unreleased jobs held; inspect the receipt before any manual recovery.

Every training rank performs its native-model forward/backward probe before DDP/training, with RNG, parameters and gradients preserved. End-of-run completion additionally requires all 64 rank probe receipts and exactly 1000 checkpoint files. Submission alone is not reported as successful training.

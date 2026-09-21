# Isolated classification actual32 reevaluation

This campaign preserves every historical result and uses the original frozen457
classification memberships and split caches. It writes only to
`evaluation/classification_actual32_20260921_v1` and never updates Excel or old
ranking tables. This is not a regression experiment and it does not train weights.

New targets (457 each; total4570): TabPFN2, TabPFN2.5, TabPFN3, LimiX2M,
LimiX16M, TabICLv1, TabICLv2, Taffy Loop3 step19650, Taffy Loop4 step22400,
Mitra2. E4/Loop1 and Loop2 are excluded by user choice. Original Mitra's existing
strict32 campaign is independent and is NOT resubmitted or changed.

Actual32 means32 executed member contributions to every test-row prediction,
or32 at each fitted hierarchy node where native many-class decomposition is
required. It does not mean equal FLOPs, equal model size, statistically independent
predictions, or identical preprocessing across methods.

- TabPFN: explicit native32 instead of native8/auto9; original inference factory,
  class handling and chunk-backoff retained.
- LimiX: original four pipeline types repeated eight times in the native predictor,
  with advancing native seeds and32 real pipeline executions; original MinMax,
  retrieval, class decomposition, softmax and averaging retained.
- TabICL/Taffy: original generator/aggregation untouched when it already gives32.
  If fewer, all original transforms are retained and supplemented by deterministic
  paired support-row permutations. Query order remains unchanged. These are real
  additional forwards, not padding with cached predictions. This supplemental
  augmentation protocol is recorded explicitly and may yield equal predictions.
- Mitra2: frozen official classifier2; native32 with seed0, no fine-tuning,
  original tiny support-only initialization validation and native query order.

Every method must pass three full-data smoke tasks (small binary low-dimensional,
multiclass, and >10-class examples) before formal dispatch. GPU checks use actual
ROCm visibility plus physical PCI/UUID identities, not CUDA environment strings.
Each node validates four distinct devices. ROCr's Slurm-assigned physical mask is
retained; HIP/CUDA aliases are removed to avoid double filtering.

Allocation: four independent bgqos jobs, each one node, four GPUs,16 CPUs,
256GiB,12h maximum, no requeue. A shared atomic (model,membership) queue avoids
duplicates. Each child has a2h runtime and56GiB RSS bound to contain failures.
Completed outputs are immutable; failed attempts and claims stay for audit.
No automatic recovery/deletion of claims or resubmission is performed.

Commands: `classification32_campaign.py prepare`, then
`classification32_submit.py submit`. Both are one-time, fail-closed operations.
Submission holds all jobs until resource verification and journals every returned
ID before release. A repeat invocation reports state without submitting again.

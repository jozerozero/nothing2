# Fixed-checkpoint G5SC inference-depth ablation

User-approved scope: training 174381, step19750, inference Loop=3/4, all 457 classification suite memberships, one faculty/gtqos node with eight GPUs. No full Loop=2 evaluation and no retraining.

The shared ICL stack and learned support-conditioned alpha are reused at every pass. Each mixed output enters the next pass. A runtime override changes only pass-count attributes after strict checkpoint loading; no tensor or checkpoint is edited. The override is installed in spawned evaluator processes too.

Before evaluation: eight ROCm-aware single-device GPU probes; real-checkpoint old/new Loop=2 bitwise consistency; Loop=3/4 finite output and realized block-call counts. Eight deterministic disjoint dataset shards each evaluate both loops. Output is isolated from training and other evaluators. No claims are reused and no prior results are overwritten.

Evaluation protocol follows the existing cross-suite evaluator: same fixed splits, 32 estimators, none/power normalization, batch8, four CPU threads, no KV cache, FP32/noAMP/noFA3. There are 457 memberships per loop: TALENT200, BCCO106, OpenML-CC18 62, PFN29, TabArena33, TabZilla27.

The report separately ranks each variant against eight frozen baseline methods. Displayed428 excludes PFN and reproduces the screenshot's coverage. Missing baseline metrics are explicitly reported, never imputed. Model selection on these benchmark scores is exploratory, not an unbiased held-out estimate.

# T25 with G5SC gated shared-depth Loop3 and Loop4

User authorized training two new regression arms, then explicitly selected G5SC-style gated loops on 2026-09-12. This is not inference-only and does not resume the trained T25 checkpoint.

Baseline: T25 Safe-Tail RW25, training151162. All original data generation, original train-only profile, regression head999, seed43, Muon8e-4, fixed4096, batch1024/micro2, 25k steps and every50 checkpoint remain. Both arms start from scratch. No classification model, SwiGLU, E4 transform, or extra attention gates are imported.

Each pass shares the original12 ICL blocks. For t>=2, h_t=h_(t-1)+alpha_D*(Blocks(h_(t-1))-h_(t-1)), alpha_D=tanh(a+0.1*(2*sigmoid(w@c_D)-1)). Same a and w reused at every pass. Parameters a=zeros(()), w=zeros(51) add52 scalars without consuming RNG. G5SC support-statistics code is copied verbatim. In regression its four classification-label statistic slots stay zero. Query values do not enter context, although the original statistic includes total sequence length.

Necessary implementation changes: additional passes use activation recomputation to bound memory without changing batch or numerical function; original Muon scalar reshape is made safe for the new scalar gate, with no change to existing tensor shapes or optimizer settings. Every training rank runs a small actual Loop3/4 FP32 forward/backward probe while restoring RNG and clearing gradients before training.

Resources per arm: faculty/test-acc/bgqos,8 nodes x8GPU=64GPU,128CPU/node,2T/node,3days,Nice0,no-requeue. Keep original exclusion set and add known unsafe nodes193,195,216,228,287,296. Total requested128GPU. No other jobs are modified.

Prepared source is copied from frozen adapted T25 source, with exact hash checks on edited baseline files; it never mutates baseline source. The deployment repo is only an overlay and is not installed into a shared environment. PYTHONPATH selects the isolated derived source and existing dependencies.

Only t25_loops is active in this branch. Inherited Loop4 evaluator files outside it are historical and are not executed.

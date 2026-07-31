# Chapter 07 two-rank HCCL scripts

Run these commands from 07_distributed_comms:

    torchrun --standalone --nproc_per_node=2 scripts/fsdp_collectives.py \
      --output-json results/fsdp_latest.json \
      --profile-dir results/profiles/fsdp_latest
    torchrun --standalone --nproc_per_node=2 scripts/tp_collectives.py \
      --output-json results/tp_classic_latest.json \
      --profile-dir results/profiles/tp_classic_latest
    torchrun --standalone --nproc_per_node=2 scripts/cp_collectives.py \
      --output-json results/cp_latest.json \
      --profile-dir results/profiles/cp_latest

The commands intentionally do not set `ASCEND_VISIBLE_DEVICES`. The scheduler must
expose exactly the two allocated devices; each worker selects its visible device by
`LOCAL_RANK`, without assuming the physical device IDs are 0 and 1.

All scripts default to Qwen3-1.7B shapes, bf16, 5 warmup iterations, and 20 measured
iterations. Use --help to change tensor dimensions or iteration counts. `--output-json`
retains every aligned iteration from every rank. The summary reports rank-local and
slow-rank median/P95/range, plus the spread between rank medians.

The reported send value is the ring/pairwise baseline for one rank's one-way remote
payload. Effective bandwidth divides that payload by the slow-rank median wall time. It
is not the same as aggregate logical bytes, send-plus-receive traffic, or a physical-link
bandwidth reported by HCCL Profiler.

`--profile-dir` captures one additional synchronized collective per operation and rank.
That trace is deliberately separate from the wall-time samples. The scripts do not claim
that a trace exists unless this option was used; inspect both rank directories before
drawing a communication conclusion.

The TP script is a classic all-reduce microbenchmark. It is not the communication ledger
of TorchTitan's default TP+sequence-parallel plan, which must be resolved and traced as a
full workload. The CP script times Q, K, V, and the attention output as four separate
All-to-All calls, matching the unfused Ulysses communication path used in the notebook.

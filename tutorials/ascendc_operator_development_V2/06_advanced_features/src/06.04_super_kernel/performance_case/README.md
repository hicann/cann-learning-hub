# custom_sk Performance Case

This case runs the earlier six-operator, 50-layer shared-weight network. Each layer runs
`GroupedMatmul -> DequantSwigluQuant -> QuantBatchMatmul -> DynamicQuant ->`
`GroupedMatmul -> DynamicQuant`, and then calls the ACLGraph clearops custom operator.
The active entry point is `network_fragment.py`; the current four-operator custom_sk
network is retained as `network_fragment_4ops.py` for comparison. `static` maps to
`aclgraph`; `sk` maps to `aclgraph+sk`.

`run_case.sh` builds and temporarily installs the ACLGraph clearops package, then uses
external `msprof` to capture one formal execution after warmup. It removes clearops when
the command finishes. Before each run, it clears this case's static Kernel compilation
output and cache so the selected mode is compiled again.

```bash
chmod +x run_case.sh
./run_case.sh --mode static --device-id 0
./run_case.sh --mode sk --device-id 0
```

The two captures are written to `profiling-static/` and `profiling-sk/`. The script uses
the MSTX range named `custom_sk_model_steps` to report each Device duration. After the
`sk` run, it also compares the latest `static` result and reports the SK improvement.

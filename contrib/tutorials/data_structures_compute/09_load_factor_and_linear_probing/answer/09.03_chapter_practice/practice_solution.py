def probe_sequence(key, table_size, max_probe):
    if table_size <= 0 or table_size & (table_size - 1):
        raise ValueError("table_size must be a positive power of two")
    if max_probe <= 0 or max_probe > table_size:
        raise ValueError("max_probe must be in [1, table_size]")
    home = hash32(key) & (table_size - 1)
    return [(home + step) & (table_size - 1) for step in range(max_probe)]


def lookup_once(table_keys, table_values, states, key, max_probe):
    for probes, slot in enumerate(
        probe_sequence(key, len(table_keys), max_probe), start=1
    ):
        state = int(states[slot])
        if state == EMPTY:
            return 0, 0, probes
        if state == FULL and int(table_keys[slot]) == key:
            return int(table_values[slot]), 1, probes
        if state not in (FULL, TOMBSTONE):
            raise ValueError("invalid state")
    return 0, 0, max_probe


def probe_statistics(probe_counts):
    values = np.asarray(probe_counts, dtype=np.int64)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("probe_counts must be nonempty and positive")
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def compute_tiling(query_count, available_cores, query_tile=128):
    if query_count <= 0 or available_cores <= 0 or query_tile <= 0:
        raise ValueError("all counts must be positive")
    block_num = min(query_count, available_cores)
    queries_per_core = (query_count + block_num - 1) // block_num
    tiles_per_core = (queries_per_core + query_tile - 1) // query_tile
    # ceil(Q / block_num) 可能使末尾若干启动核没有任务，因此不能用block_num - 1 定位尾块。tail_queries 专指最后一个实际活跃核的 valid_len。
    active_blocks = (query_count + queries_per_core - 1) // queries_per_core
    tail_queries = query_count - queries_per_core * (active_blocks - 1)
    return block_num, queries_per_core, tiles_per_core, tail_queries

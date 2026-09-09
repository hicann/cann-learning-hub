def linked_lookup(heads, keys, values, next_idx, bucket_id, key):
    cur = int(heads[bucket_id])
    while cur != -1:
        if int(keys[cur]) == key:
            return int(values[cur]), 1
        cur = int(next_idx[cur])
    return 0, 0


def bucket_lookup(bucket_keys, bucket_values, bucket_id, key):
    for slot in range(BUCKET_SIZE):
        if int(bucket_keys[bucket_id, slot]) == key:
            return int(bucket_values[bucket_id, slot]), 1
    return 0, 0


def compute_tiling(query_count, available_cores):
    if query_count <= 0 or available_cores <= 0:
        raise ValueError("counts must be positive")
    block_num = min(query_count, available_cores)
    queries_per_core = (query_count + block_num - 1) // block_num
    return block_num, queries_per_core


def access_budget(query_count, average_chain_length, bucket_bytes_per_query=32):
    if query_count < 0 or average_chain_length < 0:
        raise ValueError("counts must be nonnegative")
    linked_node_reads = query_count * average_chain_length
    bucket_loads = query_count
    bucket_bytes = query_count * bucket_bytes_per_query
    return linked_node_reads, bucket_loads, bucket_bytes

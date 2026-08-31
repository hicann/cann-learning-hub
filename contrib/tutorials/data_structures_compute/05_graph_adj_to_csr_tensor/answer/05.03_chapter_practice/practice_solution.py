import numpy as np


def pull_relax_once(row_ptr, src_idx, weights, dist_in, inf=1.0e30):
    vertex_count = len(row_ptr) - 1
    dist_out = np.array(dist_in, dtype=np.float32, copy=True)
    for vertex in range(vertex_count):
        best = float(dist_in[vertex])
        for edge in range(int(row_ptr[vertex]), int(row_ptr[vertex + 1])):
            src = int(src_idx[edge])
            if dist_in[src] < inf * 0.5:
                best = min(best, float(dist_in[src]) + float(weights[edge]))
        dist_out[vertex] = best
    return dist_out


def compute_tiling(vertex_count, available_cores):
    if vertex_count <= 0 or available_cores <= 0:
        raise ValueError("counts must be positive")
    block_num = min(vertex_count, available_cores)
    vertices_per_core = (vertex_count + block_num - 1) // block_num
    return block_num, vertices_per_core

def queue_payload_bytes(tile_length, buffer_num, tensor_count=3, dtype_bytes=4):
    if tile_length <= 0 or buffer_num not in (1, 2):
        raise ValueError("invalid pipeline configuration")
    return tensor_count * buffer_num * tile_length * dtype_bytes


tile_length = 28672 // 4 // 7
single = queue_payload_bytes(tile_length, 1)
double = queue_payload_bytes(tile_length, 2)
print("tile_length:", tile_length)
print("single:", single)
print("double:", double)
assert single == 12288
assert double == 24576
assert double == 2 * single

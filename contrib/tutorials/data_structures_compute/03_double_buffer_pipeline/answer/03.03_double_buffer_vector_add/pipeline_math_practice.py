def queue_payload_bytes(tile_length, buffer_num, tensor_count=3, dtype_bytes=4):
    if tile_length <= 0 or buffer_num not in (1, 2):
        raise ValueError("invalid pipeline configuration")
    return tensor_count * buffer_num * tile_length * dtype_bytes


single = queue_payload_bytes(256, 1)
double = queue_payload_bytes(256, 2)
print("single:", single)
print("double:", double)
assert single == 3072
assert double == 6144
assert double == 2 * single

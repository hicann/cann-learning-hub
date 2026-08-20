N = 28672
BLOCK_DIM = 4
TILE_COUNT = 7
BLOCK_INDEX = 3
TILE_INDEX = 6
ELEMENT_BYTES = 4

block_length = N // BLOCK_DIM
tile_length = block_length // TILE_COUNT
tile_bytes = tile_length * ELEMENT_BYTES

block_start = BLOCK_INDEX * block_length
tile_start = block_start + TILE_INDEX * tile_length
block_range = (block_start, block_start + block_length)
tile_range = (tile_start, tile_start + tile_length)

print("block_length=", block_length)
print("tile_length=", tile_length)
print("tile_bytes=", tile_bytes)
print("block_range=", block_range)
print("tile_range=", tile_range)

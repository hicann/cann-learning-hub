from gemm_kernel import check_config
for args in [(32769,512,128,128,128,1),(32768,512,128,128,128,4),(128,128,128,128,128,1)]:
    try:check_config(*args)
    except ValueError as e:print('Expected rejection:',str(e))
    else:raise AssertionError('Invalid config accepted')
print('Host checks passed')

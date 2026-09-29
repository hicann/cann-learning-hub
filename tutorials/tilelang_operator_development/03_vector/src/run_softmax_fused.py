"""Check the final output of fused C=128 Softmax on the course NPU."""
import json
from pathlib import Path
import torch
from softmax_fused import softmax_fused
from course_validation import cpu_input, checked_input
from course_paths import OUTPUT_DIR


def run():
    results = []
    for M, blocks, stages in ((8, 1, 1), (9216, 36, 1), (9216, 36, 2)):
        kernel = softmax_fused(M, 8, blocks, stages)
        dest = (OUTPUT_DIR / 'softmax_fused')
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f'generated_{M}_{blocks}_{stages}.cpp').write_text(kernel.get_kernel_source())
        for case in ('random', 'zero', 'constant', 'large'):
            host = cpu_input((M, 128), case)
            got = kernel(checked_input(host)).cpu()
            ref = torch.softmax(host, dim=-1)
            assert torch.isfinite(got).all() and (got >= 0).all()
            torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(got.sum(-1), torch.ones(M), rtol=1e-5, atol=1e-6)
            item = dict(M=M, blocks=blocks, stages=stages, case=case,
                        max_abs_error=(got-ref).abs().max().item(), status='PASS')
            results.append(item)
            print(json.dumps(item), flush=True)
    (dest / 'validation.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    run()

"""Contract-based ST with CPU double oracles and actual local NPU execution."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

import row_softmax


OP_DIR = Path(__file__).resolve().parent
REPO_ROOT = OP_DIR.parents[1]
CASES = json.loads((REPO_ROOT / 'course_inputs/cases.json').read_text())
assert Path(row_softmax.__file__).resolve() == OP_DIR / 'row_softmax.py'


def reference(x_cpu):
    assert x_cpu.device.type == 'cpu'
    return torch.softmax(x_cpu.double(), dim=1).float()


def make_input(shape, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator).clamp(-80, 80)


def check_output(x, expected_cpu, before_cpu, y):
    torch.npu.synchronize(x.device)
    actual = y.cpu()
    assert y.shape == x.shape
    assert y.dtype == x.dtype == torch.float32
    assert y.device == x.device
    assert y.is_contiguous()
    assert y.data_ptr() != x.data_ptr()
    assert torch.equal(x.cpu().view(torch.int32), before_cpu.view(torch.int32)), 'input bits changed'
    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()
    # Exact task policy: every element, no percentage allowance.
    error = (actual.double() - expected_cpu.double()).abs()
    allowance = 1e-6 + 1e-4 * expected_cpu.double().abs()
    assert (error <= allowance).all(), f'max normalized error: {(error / allowance).max().item()}'
    row_error = (actual.double().sum(dim=1) - 1).abs()
    assert (row_error <= 1e-5).all(), f'row sum error: {row_error.max().item()}'
    return {
        'max_abs_error': error.max().item(),
        'max_normalized_error': (error / allowance).max().item(),
        'max_row_sum_error': row_error.max().item(),
    }


@pytest.mark.parametrize('case', CASES, ids=[case['case_id'] for case in CASES])
def test_specified(case, record_property):
    torch.npu.set_device(case['device_id'])
    x_cpu = make_input(case['shape'], case['seed'])
    expected = reference(x_cpu)
    x = x_cpu.to(f'npu:{case["device_id"]}')
    before = x.cpu().clone()
    y = row_softmax.run(x)
    metrics = check_output(x, expected, before, y)
    record_property('case_id', case['case_id'])
    record_property('seed', case['seed'])
    record_property('metrics', json.dumps(metrics))
    print(json.dumps({'case_id': case['case_id'], 'seed': case['seed'], **metrics}))
    if directory := os.environ.get('ROW_SOFTMAX_SOURCE_DIR'):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f'{case["case_id"]}.cpp').write_text(row_softmax.build(*case['shape']).get_kernel_source())


# These shapes exercise vector lengths beyond the supplied cases and the actual
# 72-AIV scheduling boundary (68, 72, 76), without inventing illegal N=65 tails.
ST_SHAPES = [(8, 256), (12, 320), (68, 448), (72, 768), (76, 1984), (128, 1024), (252, 2048), (256, 1984)]
PATTERNS = ['zero', 'positive80', 'negative80', 'alternating_extremes', 'first_max', 'last_max', 'tied_max', 'ramp', 'near_uniform', 'row_distinct']
NUMERIC_SHAPES = [(4, 128), (68, 192), (256, 2048)]
INVALID_SHAPES = [
    (0, 128),
    (1, 128),
    (3, 128),
    (5, 128),
    (255, 128),
    (257, 128),
    (260, 128),
    (4, 0),
    (4, 64),
    (4, 127),
    (4, 129),
    (4, 2047),
    (4, 2049),
    (4, 2112),
]


def exercise(x_cpu, label):
    torch.npu.set_device(0)
    x = x_cpu.to('npu:0')
    before = x.cpu().clone()
    y = row_softmax.run(x)
    metrics = check_output(x, reference(x_cpu), before, y)
    print(json.dumps({'st_case': label, **metrics}))
    return y


@pytest.mark.parametrize('shape', ST_SHAPES, ids=[f'm{m}_n{n}' for m, n in ST_SHAPES])
def test_legal_shape(shape):
    exercise(make_input(shape, 20260916), f'legal_shape_{shape}')


def patterned_input(shape, pattern):
    rows, cols = shape
    if pattern in ('zero', 'positive80', 'negative80'):
        return torch.full(shape, {'zero': 0.0, 'positive80': 80.0, 'negative80': -80.0}[pattern])
    if pattern == 'alternating_extremes':
        return torch.tensor([-80.0, 80.0]).repeat(rows, cols // 2)
    if pattern in ('first_max', 'last_max', 'tied_max'):
        x = torch.full(shape, -80.0)
        indices = {'first_max': [0], 'last_max': [cols - 1], 'tied_max': [0, cols // 2, cols - 1]}[pattern]
        x[:, indices] = 80.0
        return x
    if pattern == 'ramp':
        return torch.linspace(-80, 80, cols).repeat(rows, 1)
    if pattern == 'near_uniform':
        x = torch.full(shape, -0.0)
        x[:, 1::2] = torch.finfo(torch.float32).eps
        return x
    if pattern == 'row_distinct':
        x = torch.full(shape, -40.0)
        x[torch.arange(rows), (torch.arange(rows) * 67) % cols] = 40.0
        return x
    raise AssertionError(pattern)


@pytest.mark.parametrize('shape', NUMERIC_SHAPES, ids=[f'm{m}_n{n}' for m, n in NUMERIC_SHAPES])
@pytest.mark.parametrize('pattern', PATTERNS)
def test_numeric_boundary(shape, pattern):
    x = patterned_input(shape, pattern)
    expected = reference(x)
    # Sanity-check the independent oracle against an exact mathematical case.
    if pattern in ('zero', 'positive80', 'negative80'):
        assert torch.equal(expected, torch.full_like(x, 1.0 / shape[1]))
    exercise(x, f'{pattern}_{shape}')


@pytest.mark.parametrize('offset', [1, 8, 257], ids=['offset1', 'offset8', 'offset257'])
def test_contiguous_view_storage(offset):
    torch.npu.set_device(0)
    x_cpu = make_input((4, 128), 313)
    storage = torch.full((offset + x_cpu.numel() + 17,), -13.0, device='npu:0')
    x = storage[offset : offset + x_cpu.numel()].view(4, 128)
    x.copy_(x_cpu)
    before_storage = storage.cpu().clone()
    before = x.cpu().clone()
    assert x.is_contiguous() and x.storage_offset() == offset
    y = row_softmax.run(x)
    check_output(x, reference(x_cpu), before, y)
    assert torch.equal(storage.cpu().view(torch.int32), before_storage.view(torch.int32))
    # Verify storage independence by an actual write to the result.
    y.fill_(0)
    torch.npu.synchronize()
    assert torch.equal(storage.cpu().view(torch.int32), before_storage.view(torch.int32))


def test_repeat_and_shift():
    torch.npu.set_device(0)
    x_cpu = make_input((68, 192), 811)
    x = x_cpu.to('npu:0')
    y1 = row_softmax.run(x)
    held = y1.cpu().clone()
    y2 = row_softmax.run(x)
    check_output(x, reference(x_cpu), x_cpu, y2)
    assert torch.equal(y1.cpu(), held), 'later call changed earlier output'
    assert y1.data_ptr() != y2.data_ptr()
    shifted = x_cpu + torch.linspace(-60, 60, 68).unsqueeze(1)
    # Use actual rounded shifted input for the primary oracle.
    y3 = exercise(shifted, 'shifted_rows')
    torch.testing.assert_close(y3.cpu(), held, rtol=1e-4, atol=1e-6)
    exercise(make_input((68, 192), 812), 'changed_input_same_compiled_shape')


def test_backend_and_builder():
    torch.npu.set_device(0)
    compiled = row_softmax.build(4, 128)
    source = compiled.get_kernel_source()
    assert '__simd_vf__' in source and '__vector__' in source
    assert 'asc_copy_gm2ub' in source and 'asc_copy_ub2gm' in source
    x_cpu = make_input((4, 128), 919)
    x = x_cpu.to('npu:0')
    y = torch.full_like(x, float('nan'))
    compiled(x, y)
    check_output(x, reference(x_cpu), x_cpu, y)
    # Also exercise the public path with this compiled shape.
    check_output(x, reference(x_cpu), x_cpu, row_softmax.run(x))


@pytest.mark.parametrize('shape', INVALID_SHAPES, ids=[f'm{m}_n{n}' for m, n in INVALID_SHAPES])
def test_reject_shape(shape):
    x = torch.zeros(shape, device='npu:0', dtype=torch.float32)
    reason = 'rows' if shape[0] != 4 else 'cols'
    with pytest.raises(ValueError, match=reason):
        row_softmax.run(x)


@pytest.mark.parametrize('shape', [(), (512,), (4, 1, 128)], ids=['rank0', 'rank1', 'rank3'])
def test_reject_rank(shape):
    with pytest.raises(ValueError, match='rank'):
        row_softmax.run(torch.zeros(shape, dtype=torch.float32, device='npu:0'))


@pytest.mark.parametrize(
    'dtype',
    [torch.float16, torch.bfloat16, torch.float64, torch.int32, torch.bool, torch.complex64],
    ids=['float16', 'bfloat16', 'float64', 'int32', 'bool', 'complex64'],
)
def test_reject_dtype(dtype):
    with pytest.raises(TypeError, match='dtype.*float32'):
        row_softmax.run(torch.zeros((4, 128), dtype=dtype, device='npu:0'))


@pytest.mark.parametrize(
    'value',
    [float('nan'), float('inf'), -float('inf'), 80.00000762939453, -80.00000762939453],
    ids=['nan', 'posinf', 'neginf', 'above80', 'below_neg80'],
)
def test_reject_value(value):
    x = torch.zeros((4, 128), device='npu:0')
    x[-1, -1] = value
    before = x.cpu().view(torch.int32).clone()
    reason = 'finite' if not torch.isfinite(torch.tensor(value)) else '80'
    with pytest.raises(ValueError, match=reason):
        row_softmax.run(x)
    assert torch.equal(x.cpu().view(torch.int32), before)


@pytest.mark.parametrize('layout', ['transpose', 'stride2', 'expand'])
def test_reject_noncontiguous(layout):
    if layout == 'transpose':
        x = torch.zeros((128, 4), device='npu:0').t()
    elif layout == 'stride2':
        x = torch.zeros((4, 256), device='npu:0')[:, ::2]
    else:
        x = torch.zeros((1, 128), device='npu:0').expand(4, 128)
    assert x.shape == (4, 128) and not x.is_contiguous()
    with pytest.raises(ValueError, match='contiguous'):
        row_softmax.run(x)


@pytest.mark.parametrize('kind', ['cpu', 'sparse', 'not_tensor'])
def test_reject_interface(kind):
    if kind == 'cpu':
        x, reason = torch.zeros((4, 128)), 'NPU'
    elif kind == 'sparse':
        x, reason = torch.zeros((4, 128)).to_sparse(), 'strided'
    else:
        x, reason = [[0.0]], 'tensor'
    with pytest.raises(ValueError, match=reason):
        row_softmax.run(x)


@pytest.mark.parametrize(
    'rows,cols,reason',
    [(3, 128, 'rows'), (4, 129, 'cols'), (4.0, 128, 'rows'), (4, 128.0, 'cols')],
    ids=['rows3', 'cols129', 'float_rows', 'float_cols'],
)
def test_reject_builder(rows, cols, reason):
    # Prime the valid cache entry first, so the test detects invalid cache hits.
    row_softmax.build(4, 128)
    with pytest.raises(ValueError, match=reason):
        row_softmax.build(rows, cols)

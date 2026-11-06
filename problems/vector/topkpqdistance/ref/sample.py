import argparse
from pathlib import Path

import numpy as np
import reference


ROOT = Path(__file__).resolve().parent

OUTPUTS = [
    (
        'topk_distance.bin',
        [8],
        np.dtype('float32'),
        32,
        0.0001,
        0.0001,
        0.0001
    ),
    ('topk_ivf.bin', [8], np.dtype('int32'), 32, 0, 0, 0),
    ('topk_index.bin', [8], np.dtype('int32'), 32, 0, 0, 0),
]


def generate(directory):
    rng = np.random.default_rng(42)
    for name in ('input', 'output', 'golden'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    for item in OUTPUTS:
        (directory / 'output' / item[0]).unlink(missing_ok=True)
        (directory / 'golden' / item[0]).unlink(missing_ok=True)
    x0 = np.asarray(16, dtype=np.dtype('int32')).reshape([])
    (directory / 'input' / 'actual_count.bin').write_bytes(x0.tobytes())
    x1 = rng.uniform(1.0, 1000000.0, [16]).astype(np.dtype('float32'))
    (directory / 'input' / 'pq_distance.bin').write_bytes(x1.tobytes())
    x2 = rng.uniform(0.0, 1.0, [8]).astype(np.dtype('float32'))
    (directory / 'input' / 'grouped_extreme_distance.bin').write_bytes(x2.tobytes())
    x3 = rng.integers(0, 256, size=[16], dtype=np.dtype('int32'))
    (directory / 'input' / 'pq_ivf.bin').write_bytes(x3.tobytes())
    x4 = rng.integers(0, 100000000, size=[16], dtype=np.dtype('int32'))
    (directory / 'input' / 'pq_index.bin').write_bytes(x4.tobytes())
    result = reference.impl(x0, x1, x2, x3, x4, **{
        'order': 'ASC',
        'k': 8,
        'group_size': 2
    })
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError('reference output group count mismatch')
    values = [result[0], result[1], result[2]]
    for item, array in zip(OUTPUTS, values):
        name, shape, dtype = item[:3]
        if np.isscalar(array):
            array = np.asarray(array).reshape(shape)
        array = np.asarray(array)
        if array.dtype != dtype or list(array.shape) != shape:
            raise ValueError('reference dtype/shape mismatch: ' + name)
        (directory / 'golden' / name).write_bytes(array.tobytes())
    print('Sample data generated.')


def verify(directory):
    passed = True
    expected = {item[0] for item in OUTPUTS}
    for folder in ('output', 'golden'):
        actual = {p.name for p in (directory / folder).iterdir()}
        if actual != expected:
            print('VERIFY ERROR: unexpected or missing files in ' + folder)
            return 1
    for item in OUTPUTS:
        name, shape, dtype, size, rtol, atol, tol = item[:7]
        try:
            raw = (directory / 'output' / name).read_bytes()
            golden = (directory / 'golden' / name).read_bytes()
            if len(raw) != size or len(golden) != size:
                raise ValueError('exact byte size mismatch')
            a, b = [np.frombuffer(v, dtype=dtype) for v in (raw, golden)]
            if dtype.kind in 'biu':
                if rtol == 0 and atol == 0:
                    close = a == b
                else:
                    close = np.isclose(a, b, rtol=rtol, atol=atol)
            elif True:
                close = np.isclose(
                    a.astype(np.float64),
                    b.astype(np.float64),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True
                )
            bad = int(np.count_nonzero(~close))
            ok = bad <= tol * a.size
            print(
                ('PASS' if ok else 'Wrong Answer')
                + ': ' + name + f', mismatched={bad}/{a.size}'
            )
            passed &= ok
        except (OSError, ValueError) as error:
            print('VERIFY ERROR: ' + name + ': ' + str(error))
            passed = False
    print('PASSED' if passed else 'FAILED')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'verify'))
    parser.add_argument('--directory', type=Path, default=ROOT / 'build-npu')
    args = parser.parse_args()
    try:
        if args.action == 'generate':
            generate(args.directory)
        else:
            raise SystemExit(verify(args.directory))
    except Exception as error:
        print('SAMPLE ERROR: ' + str(error))
        raise SystemExit(2)

"""Compare the dynamic image insertion IR against PyTorch BF16 semantics."""

import argparse
from pathlib import Path

import numpy as np
import torch
import tvm
from image_embedding_ir import build_image_embedding_ir
from tvm import relax


def expected(hidden, image, start):
    result = hidden.copy()
    result[0, start : start + len(image)] = (
        torch.from_numpy(image).to(torch.bfloat16).float().numpy()
    )
    return result


def apply(vm, hidden, image, start):
    return vm["main"](
        tvm.runtime.tensor(hidden),
        tvm.runtime.tensor(image),
        tvm.runtime.tensor(np.asarray([start], np.int32)),
    ).numpy()


def assert_bits(actual, reference):
    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    with tvm.transform.PassContext(disabled_pass=["StaticPlanBlockMemory"]):
        executable = relax.build(build_image_embedding_ir(cpu=True), target="llvm")
    vm = relax.VirtualMachine(executable, tvm.cpu())
    rng = np.random.default_rng(42)
    for n, k, start in (
        (1, 1, 0),
        (2, 1, 0),
        (2, 1, 1),
        (280, 280, 0),
        (337, 256, 9),
        (337, 1, 336),
    ):
        hidden = rng.standard_normal((1, n, 1536), dtype=np.float32)
        image = rng.standard_normal((k, 1536), dtype=np.float32)
        assert_bits(apply(vm, hidden, image, start), expected(hidden, image, start))
    hidden = rng.standard_normal((1, 13, 1536), dtype=np.float32)
    first = rng.standard_normal((2, 1536), dtype=np.float32)
    second = rng.standard_normal((3, 1536), dtype=np.float32)
    actual = apply(vm, apply(vm, hidden, first, 1), second, 9)
    assert_bits(actual, expected(expected(hidden, first, 1), second, 9))
    special = np.zeros((1, 1536), np.float32)
    special.view(np.uint32).flat[:10] = [
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0xFFC00000,
        0x7F800001,
        0xFF800001,
        0x80000000,
        0x3F808000,
        0x3F818000,
        0xBF818000,
    ]
    assert_bits(
        apply(vm, np.zeros((1, 1, 1536), np.float32), special, 0),
        expected(np.zeros((1, 1, 1536), np.float32), special, 0),
    )
    if args.image:
        image = np.fromfile(args.image, np.float32).reshape(-1, 1536)
        hidden = rng.standard_normal((1, len(image) + 3, 1536), dtype=np.float32)
        assert_bits(apply(vm, hidden, image, 2), expected(hidden, image, 2))
    print(
        "Dynamic image insertion matches PyTorch BF16 bit-for-bit, including multiple images"
    )


if __name__ == "__main__":
    main()

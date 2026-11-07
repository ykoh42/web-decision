from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path
import sys

import tvm
from tvm.script import tirx as T


SOURCE_PROTOTYPE = Path(__file__).resolve().parent / "online_kernel.py"


@cache
def source_module():
    spec = importlib.util.spec_from_file_location(
        "kev_querytile_online_source", SOURCE_PROTOTYPE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SOURCE_PROTOTYPE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def online_cpu_reference(query, key, value, bias):
    """Run the source prototype's CPU recurrence with ONNX graph layouts."""
    return source_module().online_f32_differential(
        query,
        key.transpose(0, 1, 3, 2),
        value,
        bias[0, 0],
    )


@cache
def attention_primfunc():
    """Adapt the standalone online-softmax TIR to Kev's imported tensor layouts."""
    source = source_module().OnlineAttention["main"].script()

    replacements = {
        'k: T.Buffer((1, 16, seq_len, 128), "float16")':
            'k: T.Buffer((1, 16, 128, seq_len), "float16")',
        'bias: T.Buffer((1, seq_len, seq_len), "float16")':
            'bias: T.Buffer((1, 1, seq_len, seq_len), "float16")',
        'out: T.Buffer((1, 16, seq_len, 128), "float16"), seq_len: T.int32':
            'seq_len: T.int32, out: T.Buffer((1, 16, seq_len, 128), "float16")',
        'k[0, head, key, dim]': 'k[0, head, dim, key]',
        'bias[0, query_row, key]': 'bias[0, 0, query_row, key]',
    }
    for old, new in replacements.items():
        count = source.count(old)
        if count != 1:
            raise ValueError(f"Expected one source occurrence for {old!r}, found {count}")
        source = source.replace(old, new)

    func = tvm.script.from_source(source, extra_vars={"T": T})
    return (
        func.with_attr("global_symbol", "kev_attention_query_tile")
        .with_attr("tirx.is_scheduled", 1)
    )

"""Focused regression for the pinned Kanana text M-RoPE recomposition."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch
import tvm
from tvm.relax.frontend.torch import from_exported_program

SCRIPT_PATH = Path(__file__).with_name("prototype_text_tvm_relax.py")
SPEC = importlib.util.spec_from_file_location("kanana_text_tvm_relax", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


class RotaryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mrope_section = [24, 20, 20]
        self.register_buffer("inv_freq", torch.arange(64, dtype=torch.float32))


class RecompositionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary = RotaryModule()
        SCRIPT.install_static_mrope_recomposition(torch, self)

    def forward(self, freq):
        return self.rotary.recomposition_frequencies(freq)


def _reference_recomposition(freq, sections):
    axis_by_channel = []
    for channel in range(freq.shape[-1]):
        if channel < sections[1] * 3 and channel % 3 == 1:
            axis_by_channel.append(1)
        elif channel < sections[2] * 3 and channel % 3 == 2:
            axis_by_channel.append(2)
        else:
            axis_by_channel.append(0)
    indices = torch.tensor(axis_by_channel, dtype=torch.int64, device=freq.device)
    indices = indices.view(1, 1, 1, -1).expand(1, freq.shape[1], freq.shape[2], -1)
    selected = torch.gather(freq, 0, indices).squeeze(0)
    return torch.cat((selected, selected), dim=-1)


def _wgsl_sources(module):
    sources = []
    for imported in module.imports:
        sources.extend(_wgsl_sources(imported))
    try:
        sources.append(module.inspect_source("wgsl"))
    except tvm.error.InternalError:
        pass
    return sources


class StaticMropeRecompositionTest(unittest.TestCase):
    def test_preserves_values_and_avoids_boolean_where_masks(self):
        wrapper = RecompositionWrapper().eval()
        freq = torch.randn((3, 1, 5, 64), dtype=torch.float16)
        expected = _reference_recomposition(freq, wrapper.rotary.mrope_section)
        actual = wrapper.rotary.recomposition_frequencies(freq)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        exported = torch.export.export(wrapper, (freq,))
        relax_module = from_exported_program(
            exported,
            keep_params_as_input=True,
            run_ep_decomposition=True,
            unwrap_unit_return_tuple=True,
        )
        normalized, counts = SCRIPT.normalize_webgpu_dtypes(
            tvm, np, relax_module, max_sequence_length=20_480
        )
        self.assertFalse(counts["graphHasInt64TensorTypes"])
        self.assertNotIn("R.equal(", normalized.script())

        executable = tvm.relax.build(normalized, target="llvm")
        vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
        output = vm["main"](tvm.runtime.tensor(freq.numpy(), device=tvm.cpu())).numpy()
        np.testing.assert_array_equal(output, expected.numpy())

        webgpu_executable = tvm.relax.build(normalized, target="webgpu")
        wgsl = "\n".join(
            source for source in _wgsl_sources(webgpu_executable.mod) if source
        )
        self.assertNotIn("array<i8>", wgsl)


if __name__ == "__main__":
    unittest.main()

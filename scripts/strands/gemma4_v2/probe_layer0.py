"""Compare the pinned checkpoint's first text layer using native Relax."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
import tvm
from safetensors import safe_open
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.script import tirx as T


def make_unpack_matrix(size, words, rows, columns):
    @T.prim_func(s_tir=True)
    def unpack_matrix(
        weight: T.Buffer((words,), "uint32"),
        output: T.Buffer((rows, columns), "float32"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        with T.sblock("root"):
            for block in T.thread_binding((size + 255) // 256, thread="blockIdx.x"):
                for thread in T.thread_binding(256, thread="threadIdx.x"):
                    index = block * 256 + thread
                    if index < size:
                        word = weight[index // 2]
                        half = T.if_then_else(
                            index % 2 == 0,
                            word & T.uint32(65535),
                            word >> T.uint32(16),
                        )
                        output[index // columns, index % columns] = T.reinterpret(
                            "float32", half << T.uint32(16)
                        )

    return unpack_matrix


def make_unpack(shape):
    size = int(np.prod(shape))
    words = (size + 1) // 2

    if len(shape) == 2:
        rows, columns = shape

        return make_unpack_matrix(size, words, rows, columns)

    if len(shape) != 1:
        raise ValueError(f"Unsupported parameter shape {shape}")

    @T.prim_func(s_tir=True)
    def unpack(
        weight: T.Buffer((words,), "uint32"), output: T.Buffer((size,), "float32")
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        with T.sblock("root"):
            for block in T.thread_binding((size + 255) // 256, thread="blockIdx.x"):
                for thread in T.thread_binding(256, thread="threadIdx.x"):
                    index = block * 256 + thread
                    if index < size:
                        word = weight[index // 2]
                        half = T.if_then_else(
                            index % 2 == 0, word & T.uint32(65535), word >> T.uint32(16)
                        )
                        output[index] = T.reinterpret("float32", half << T.uint32(16))

    return unpack


def pack_parameters(module, parameters, input_count):
    original = module["main"]
    print(
        "Packing graph arguments",
        len(original.params),
        input_count,
        len(parameters),
        flush=True,
    )
    module = tvm.IRModule(
        {
            key: value
            for key, value in module.functions.items()
            if key.name_hint != "main"
        },
        attrs=module.attrs,
        global_infos=module.global_infos,
    )
    builder = relax.BlockBuilder(module)
    float_main = builder.add_func(
        original.with_attr("global_symbol", "float_main"), "float_main"
    )
    kernels = {}
    tensors = []
    values = []
    for index, parameter in enumerate(parameters):
        array = parameter.numpy()
        bits = array.view(np.uint32).reshape(-1)
        if np.any(bits & np.uint32(65535)):
            raise ValueError(f"Parameter {index} is not exactly BF16-representable")
        halves = (bits >> np.uint32(16)).astype(np.uint16)
        if len(halves) % 2:
            halves = np.pad(halves, (0, 1))
        values.append(halves.view(np.uint32))
        shape = array.shape
        name = "unpack_" + "_".join(str(dimension) for dimension in shape)
        if shape not in kernels:
            kernels[shape] = builder.add_func(
                make_unpack(shape).with_attr("global_symbol", name),
                name,
            )
        tensors.append(shape)
    variables = [
        relax.Var(f"input_{index}", parameter.ty)
        for index, parameter in enumerate(original.params[:input_count])
    ] + [
        relax.Var(f"packed_{index}", relax.TensorType(value.shape, "uint32"))
        for index, value in enumerate(values)
    ]
    with builder.function("packed_main", variables):
        with builder.dataflow():
            decoded = []
            for variable, shape in zip(variables[input_count:], tensors):
                decoded.append(
                    builder.emit(
                        relax.call_tir(
                            kernels[shape],
                            (variable,),
                            relax.TensorType(shape, "float32"),
                        )
                    )
                )
            result = builder.emit(
                relax.Call(float_main, variables[:input_count] + decoded)
            )
            result = builder.emit_output(result)
        builder.emit_func_output(result)
    return builder.get(), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tvm-source", type=Path)
    parser.add_argument("--emsdk", type=Path)
    parser.add_argument("--packed-webgpu", action="store_true")
    args = parser.parse_args()
    if args.output and (not args.tvm_source or not args.emsdk):
        parser.error("--output requires --tvm-source and --emsdk")
    config = transformers.AutoConfig.from_pretrained(str(args.checkpoint)).text_config
    fixture = torch.load(args.capture, map_location="cpu", weights_only=True)
    split = fixture["state_length"]

    def attention(_module, query, key, value, _mask, **_kwargs):
        scores = query.float() @ key.float().transpose(-1, -2)
        state = (
            torch.softmax(scores[:, :, :split, :split], dim=-1)
            @ value[:, :, :split].float()
        )
        question = torch.softmax(scores[:, :, split:, :], dim=-1) @ value.float()
        return torch.cat((state, question), dim=2).to(query.dtype).transpose(1, 2), None

    transformers.AttentionInterface.register("web_decision_v2_probe", attention)
    config._attn_implementation = "web_decision_v2_probe"
    layer = Gemma4TextDecoderLayer(config, args.layer).eval()
    prefix = f"model.language_model.layers.{args.layer}."
    expected_weights = set(layer.state_dict())
    with safe_open(args.checkpoint / "model.safetensors", framework="pt") as file:
        weights = {
            key[len(prefix) :]: file.get_tensor(key).float()
            for key in file.keys()
            if key.startswith(prefix) and key[len(prefix) :] in expected_weights
        }
    layer.load_state_dict(weights, strict=True)

    class Layer(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, hidden, per_layer, cosine, sine, *shared):
            shared_kv = (
                {self.block.self_attn.layer_type: (shared[0], shared[1])}
                if shared
                else {}
            )
            output = self.block(
                hidden,
                per_layer,
                shared_kv_states=shared_kv,
                position_embeddings=(cosine, sine),
                attention_mask=None,
            )
            if self.block.self_attn.store_full_length_kv:
                key, value = shared_kv[self.block.self_attn.layer_type]
                return torch.cat(
                    (output.reshape(-1), key.reshape(-1), value.reshape(-1))
                )
            return output

    graph = Layer(layer).eval()
    inputs = (
        fixture["hidden"],
        fixture["per_layer_input"],
        *fixture["position_embeddings"],
    )
    if layer.self_attn.is_kv_shared_layer:
        inputs += tuple(fixture["shared_kv"][layer.self_attn.layer_type])
    reference = fixture["output"].numpy()
    if layer.self_attn.store_full_length_kv:
        key, value = fixture["shared_kv"][layer.self_attn.layer_type]
        reference = np.concatenate(
            (reference.reshape(-1), key.numpy().reshape(-1), value.numpy().reshape(-1))
        )
    with torch.no_grad():
        bf16_output = (
            graph.to(torch.bfloat16)(*(value.to(torch.bfloat16) for value in inputs))
            .float()
            .numpy()
        )
    print(
        "CPU BF16 vs upstream BF16 max error:",
        np.max(np.abs(bf16_output - reference)),
        flush=True,
    )
    graph.float()
    with torch.no_grad():
        expected = graph(*inputs).numpy()
    difference = np.abs(expected - reference)
    print(
        "FP32 vs upstream BF16 max/mean error:",
        difference.max(),
        difference.mean(),
        flush=True,
    )
    exported = torch.export.export(graph, inputs, strict=False)
    print("Torch export passed", flush=True)
    module = from_exported_program(
        exported, unwrap_unit_return_tuple=True, keep_params_as_input=True
    )
    module, parameters = relax.frontend.detach_params(module)
    print("Native TVM import passed", flush=True)
    vm = relax.VirtualMachine(relax.build(module, target="llvm"), tvm.cpu())
    actual = vm["main"](
        *(tvm.runtime.tensor(value.numpy()) for value in inputs), *parameters["main"]
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    print("TVM CPU max error:", np.max(np.abs(actual - expected)), flush=True)
    if args.output:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen35"))
        from qwen35_tvm_build import (
            export_tvmjs,
            verify_emsdk_checkout,
            verify_tvm_checkout,
        )

        verify_tvm_checkout(args.tvm_source)
        verify_emsdk_checkout(args.emsdk)
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        packed_values = None
        if args.packed_webgpu:
            module, packed_values = pack_parameters(
                module, parameters["main"], len(inputs)
            )
        executable = relax.build(module, target=target)
        args.output.mkdir(parents=True, exist_ok=True)
        export_tvmjs(
            executable, args.output / "layer.wasm", args.emsdk, args.tvm_source
        )
        records = []
        for index, tensor in enumerate(
            [
                *(value.numpy() for value in inputs),
                *(
                    packed_values
                    if packed_values is not None
                    else (value.numpy() for value in parameters["main"])
                ),
            ]
        ):
            raw = tensor.tobytes()
            name = f"input-{index}.bin"
            (args.output / name).write_bytes(raw)
            records.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        (args.output / "inputs.json").write_text(json.dumps(records) + "\n")
        (args.output / "output-shape.json").write_text(
            json.dumps(list(expected.shape)) + "\n"
        )
        expected.tofile(args.output / "expected.bin")
        print("WebGPU layer export passed", flush=True)


if __name__ == "__main__":
    main()

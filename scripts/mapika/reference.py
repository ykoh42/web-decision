"""Record prompts, token IDs and optional eager output from the pinned Mapika source."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reference_cases import fixtures

REPOSITORY = "Mapika/decider-2b"
REVISION = "7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3"
UPSTREAM_REVISION = "ab45f9147de7a2da974f68c77cc8ed60908b379c"
MAX_STATE_TOKENS = 32768
MAX_ROW_TOKENS = MAX_STATE_TOKENS + 4096
MAX_REQUEST_TOKENS = 1048576
MAX_ROWS = 1024
MAX_IMAGES = 4
SMALL_FILES = {
    "decider_config.json": "bbe0f382362b23a5ab9115e23d01a1b0511bf334c9355092a3b4d2ea0ddb5e1d",
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250",
}
VISION_FILES = {
    "vision/config.json": "1eb0581a3232412f874acbeb2725292a22be2feebdce04b15363f4255464c07e",
    "vision/processor_config.json": "9ad79dcc3fc51f5eaa519b43adcb3ff5b748a5a0af959237ad7b68b710b9e041",
    "vision/tokenizer.json": SMALL_FILES["tokenizer.json"],
    "vision/tokenizer_config.json": SMALL_FILES["tokenizer_config.json"],
}
WEIGHT_FILES = {
    "model.safetensors": "1265b014ce710ef219eb137792705bc0e15a59f1c68460cf26394ae5416876e1",
    "vision/visual.safetensors": "bf7a6edb1196c9478f63736079aef71d4b0faa98c3c37fcc09f0239cc53d1c97",
}


def verify_files(snapshot, pins):
    for name, expected in pins.items():
        with (snapshot / name).open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Incorrect SHA-256 for Mapika {name}")


def use_source(source):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    if git("rev-parse", "HEAD") != UPSTREAM_REVISION or git("status", "--porcelain"):
        raise ValueError(
            "Use a clean Mapika/decider v1.10.0 checkout at the pinned revision"
        )
    sys.path.insert(0, str(source))


def assemble_request(request, config):
    from decider import systemone as system
    from decider.images import decode_assets, place_images

    state = system.render_state(request["state"])
    images = []
    if request.get("assets"):
        assets = decode_assets(request["assets"], MAX_IMAGES)
        state, images = place_images(
            state, request["questions"], assets, MAX_IMAGES, image_reference_text
        )
    rendered = {
        key: system.render_question(question)
        for key, question in request["questions"].items()
    }
    independent = request["independent"]
    flat, index = system.plan_rows(rendered, config["isolated_levels"] and independent)
    rows = [[row] for row in flat] if independent else [flat]
    if len(rows) > MAX_ROWS:
        raise ValueError(
            f"too many questions: {len(rows)} scoring rows exceeds {MAX_ROWS}"
        )
    return SimpleNamespace(
        state=state, images=images, rendered=rendered, index=index, rows=rows
    )


def image_reference_text(index, count):
    # Pinned decider.vision_reader.reference_text needs the same values but imports torch.
    return "the image" if count == 1 else f"image {index + 1}"


def add_images(tokenizer, processor, items, images):
    from decider.vision_reader import VisionReader

    marker_ids = tokenizer.convert_tokens_to_ids(
        ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
    )
    vision = SimpleNamespace(
        ip=processor,
        merge=processor.merge_size,
        **dict(zip(("vs", "pad", "ve"), marker_ids)),
    )
    vision.image_inputs = lambda inputs: VisionReader.image_inputs(vision, inputs)
    return VisionReader.add_images(vision, items, images)


def prepare_request(request, prepared, tokenizer, processor, config):
    from decider import systemone as system
    from decider import temperature
    from decider.prompt_fast import build_rows, unique_tokens

    rows = [
        [(row["question"], row["options"]) for row in group] for group in prepared.rows
    ]
    items, context_size = build_rows(
        tokenizer, prepared.state, rows, max_ctx_tokens=MAX_STATE_TOKENS
    )
    types = system.row_types(prepared.rendered, prepared.index)
    for item, kinds in zip(
        items, [[kind] for kind in types] if request["independent"] else [types]
    ):
        item["types"] = kinds
    pixels = (
        add_images(tokenizer, processor, items, prepared.images)
        if prepared.images
        else None
    )
    lengths = [len(item["ids"]) for item in items]
    if max(lengths, default=0) > MAX_ROW_TOKENS or sum(lengths) > MAX_REQUEST_TOKENS:
        raise ValueError(
            "too many tokens: Mapika serving row or request budget exceeded"
        )
    (base, by_type), _ = temperature.from_config(config)
    temperatures = temperature.for_items(base, by_type, items)
    tokens = sum(lengths) if prepared.images else unique_tokens(items, context_size)
    return SimpleNamespace(
        items=items,
        pixels=pixels,
        temperatures=temperatures,
        usage={"input_tokens": tokens, "output_tokens": 0},
        context_size=context_size,
        raw_context_size=len(
            tokenizer.encode("Context:\n" + prepared.state, add_special_tokens=False)
        ),
    )


def text_logits(model, inputs):
    import torch
    from decider.model import collate

    with torch.inference_mode():
        for item in inputs.items:
            batch = collate([item], model.tok.pad_token_id)
            yield model(batch).cpu()


def vision_logits(reader, inputs):
    from decider import engine

    captured = []
    original = engine.read_slots

    def observe(logits, rows, slots, counts, temperatures, group_sizes):
        captured.append(logits[rows, slots].cpu())
        return original(logits, rows, slots, counts, temperatures, group_sizes)

    with patch.object(engine, "read_slots", observe):
        probabilities = reader.score(inputs.items, inputs.pixels, inputs.temperatures)
    if len(captured) != len(inputs.items):
        raise RuntimeError("The pinned vision scorer must forward every row")
    return zip(captured, probabilities, strict=True)


def score_items(model, reader, inputs):
    from decider import temperature

    if inputs.pixels is not None:
        yield from vision_logits(reader, inputs)
        return
    for idx, logits in enumerate(text_logits(model, inputs)):
        selected = temperature.item_slice(inputs.temperatures, idx, idx + 1)
        yield (
            logits,
            temperature.scaled_softmax(
                logits, temperature.slot_temperatures(selected, [inputs.items[idx]])
            ),
        )


def record_scores(prepared, inputs, rows, model, reader, config):
    from decider import systemone as system

    distributions = []
    for row, (logits, probabilities) in zip(
        rows, score_items(model, reader, inputs), strict=True
    ):
        counts = row["optionCounts"]
        row["logits"] = [
            values[:count].tolist()
            for values, count in zip(logits, counts, strict=True)
        ]
        row["probabilities"] = [
            values[:count].tolist()
            for values, count in zip(probabilities, counts, strict=True)
        ]
        distributions.extend(row["probabilities"])
    return {
        "model": "decider-" + config["version"],
        "answers": system.assemble(prepared.rendered, prepared.index, distributions),
        "usage": inputs.usage,
    }


def record_positions(rows, reader, pixels):
    import torch

    for row in rows:
        text = list(range(len(row["inputIds"])))
        row["textPositionIds"] = text
        if pixels is None:
            row["positionIds"] = [text, text, text]
            continue
        ids = torch.tensor([row["inputIds"]], dtype=torch.long)
        positions, _ = reader.vlm.model.get_rope_index(
            input_ids=ids,
            mm_token_type_ids=(ids == reader.pad).long(),
            image_grid_thw=pixels["image_grid_thw"],
            attention_mask=torch.ones_like(ids),
        )
        row["positionIds"] = positions.squeeze(1).tolist()


def record_pixels(processor_output):
    pixels = processor_output["pixel_values"]
    raw = pixels.float().contiguous().numpy().astype("<f4", copy=False).tobytes()
    return {
        "shape": list(pixels.shape),
        "encoding": "float32-little-endian-base64",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def run_case(case, config, tokenizer, processor, model, reader):
    from decider import temperature

    result = dict(case)
    try:
        prepared = assemble_request(case["request"], config)
        if tokenizer is not None:
            inputs = prepare_request(
                case["request"], prepared, tokenizer, processor, config
            )
    except ValueError as error:
        return {**result, "status": "rejected", "message": str(error)}

    result.update(
        status="accepted",
        renderedState=prepared.state,
        renderedQuestions=prepared.rendered,
        plannedRows=prepared.rows,
        questionIndex=prepared.index,
        imageSizes=[list(img.size) for img in prepared.images],
    )
    if tokenizer is None:
        return result
    rows = [
        {
            "inputIds": item["ids"],
            "slots": item["slots"],
            "optionCounts": item["nopts"],
            "questionTypes": item["types"],
            "temperatures": temperature.slot_temperatures(
                temperature.item_slice(inputs.temperatures, i, i + 1), [item]
            ),
        }
        for i, item in enumerate(inputs.items)
    ]
    result.update(
        rows=rows,
        usage=inputs.usage,
        contextSize=inputs.context_size,
        rawContextSize=inputs.raw_context_size,
    )
    if inputs.pixels is not None:
        result["imageGridThw"] = inputs.pixels["image_grid_thw"].tolist()
    if model is not None:
        record_positions(rows, reader, inputs.pixels)
        if inputs.pixels is not None:
            result["pixelValues"] = record_pixels(inputs.pixels)
        result["result"] = record_scores(prepared, inputs, rows, model, reader, config)
    return result


def load_models(snapshot, needs_vision, dtype_name):
    import torch
    from decider.model import DecisionModel
    from decider.vision_reader import VisionReader

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    model = DecisionModel(str(snapshot), dtype=dtype, grad_ckpt=False).eval()
    if not needs_vision:
        return model, None
    engine = SimpleNamespace(
        m=model,
        tok=model.tok,
        dev="cpu",
        W=model.lm.lm_head.weight[model.letters].detach(),
    )
    return model, VisionReader(engine, str(snapshot), dtype=dtype)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("render", "prepare", "infer"), default="prepare"
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.stage == "infer" and not args.case:
        parser.error("--stage infer requires explicit --case selection")
    return args


def selected_cases(names):
    cases = fixtures()
    if not names:
        return cases
    available = {case["id"]: case for case in cases}
    return [available[name] for name in names]


def main():
    args = parse_arguments()
    os.environ["HF_HUB_OFFLINE"] = "1"
    use_source(args.upstream_source.resolve(strict=True))
    cases = selected_cases(args.case)
    snapshot = args.snapshot.resolve(strict=True)
    pins = dict(SMALL_FILES)
    needs_vision = any(case["request"].get("assets") for case in cases)
    if needs_vision:
        pins.update(VISION_FILES)
    if args.stage == "infer":
        pins["model.safetensors"] = WEIGHT_FILES["model.safetensors"]
        if needs_vision:
            pins.update(WEIGHT_FILES)
    verify_files(snapshot, pins)
    config = json.loads((snapshot / "decider_config.json").read_text(encoding="utf-8"))
    tokenizer = processor = model = reader = None
    if args.stage != "render":
        from transformers import AutoProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if needs_vision:
            processor = AutoProcessor.from_pretrained(
                snapshot / "vision", local_files_only=True
            ).image_processor
    if args.stage == "infer":
        model, reader = load_models(snapshot, needs_vision, args.dtype)
    report = {
        "format": "web-decision-mapika12-source-reference-v1",
        "stage": args.stage,
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "deciderRevision": UPSTREAM_REVISION,
            "files": pins,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "pillow")
        }
        if tokenizer is not None
        else {},
        "execution": {"backend": "pytorch", "device": "cpu", "dtype": args.dtype}
        if model is not None
        else None,
        "cases": [
            run_case(case, config, tokenizer, processor, model, reader)
            for case in cases
        ],
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(encoded + "\n")


if __name__ == "__main__":
    main()

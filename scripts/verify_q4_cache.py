#!/usr/bin/env python3
"""Verify every compiled TVM q4 tensor against the source checkpoint bytes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import onnx
from safe_output import create_new_file, open_directory, require_absent

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = 394


def digest_part(path, offset, length):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(offset)
        remaining = length
        while remaining:
            data = source.read(min(1024 * 1024, remaining))
            if not data:
                raise ValueError(f"unexpected end of {path}")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def compare_tensor(tensor, record, data_path, source_folder, cache_folder):
    entry = {item.key: item.value for item in tensor.external_data}
    source = source_folder / entry["location"]
    offset = int(entry["offset"])
    length = int(entry["length"])
    mismatch = (
        int(record["nbytes"]) != length
        or record["format"] != "raw"
        or digest_part(source, offset, length)
        != digest_part(cache_folder / data_path, int(record["byteOffset"]), length)
    )
    return length, mismatch


def verify_tensors(tensors, records, source_folder, cache_folder):
    mismatches = []
    total = 0
    for index, tensor in enumerate(tensors):
        data_path, record = records[f"param_{index}"]
        length, mismatch = compare_tensor(
            tensor, record, data_path, source_folder, cache_folder
        )
        total += length
        if mismatch:
            mismatches.append({"param": index, "source": tensor.name})
    return total, mismatches


def report_location(seq_len, dynamic):
    suffix = "dynamic" if dynamic else f"seq{seq_len}"
    if dynamic:
        name = "q4-cache-verification-dynamic.json"
    elif seq_len == 69:
        name = "q4-cache-verification.json"
    else:
        name = f"q4-cache-verification-seq{seq_len}.json"
    return suffix, name


def open_report(report_name):
    report_fd = open_directory(ROOT, "reports")
    try:
        require_absent(report_fd, report_name)
    except FileExistsError:
        os.close(report_fd)
        raise SystemExit(
            f"Refusing to overwrite {ROOT / 'reports' / report_name}"
        ) from None
    return report_fd


def build_report(total, mismatches, count):
    return {
        "kind": "q4-cache-byte-verification",
        "source": "original model_q4f16.onnx_data",
        "parameters": count,
        "totalBytesChecked": total,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "method": "SHA-256 of each tensor byte range versus its TVM cache record",
        "note": "A uint32 view repacks no data; the cache bytes match the source packed weights and scales.",
    }


def verify_cache(suffix):
    source_folder = ROOT / "models/kev-0.6b-onnx/onnx"
    cache_folder = ROOT / f"models/tvm-kev-q4-{suffix}"
    model = onnx.load(
        source_folder / f"model_q4f16_direct_{suffix}_u32.onnx",
        load_external_data=False,
    )
    tensors = [
        tensor
        for tensor in model.graph.initializer
        if tensor.name.startswith(("q4_packed_", "q4_scale_"))
    ]
    manifest = json.loads((cache_folder / "tensor-cache.json").read_text())
    records = {
        record["name"]: (shard["dataPath"], record)
        for shard in manifest["records"]
        for record in shard["records"]
    }
    assert len(tensors) == manifest["metadata"]["numParams"] == EXPECTED_PARAMETERS
    total, mismatches = verify_tensors(tensors, records, source_folder, cache_folder)
    return build_report(total, mismatches, len(tensors))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=69)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify and print without writing a report",
    )
    args = parser.parse_args(argv)
    suffix, report_name = report_location(args.seq_len, args.dynamic)
    report_fd = None if args.check_only else open_report(report_name)
    try:
        report = verify_cache(suffix)
        if report_fd is not None:
            create_new_file(
                report_fd,
                report_name,
                lambda stream: stream.write(
                    (json.dumps(report, indent=2) + "\n").encode()
                ),
            )
        print(json.dumps(report, indent=2))
        return 1 if report["mismatches"] else 0
    finally:
        if report_fd is not None:
            os.close(report_fd)


if __name__ == "__main__":
    raise SystemExit(main())

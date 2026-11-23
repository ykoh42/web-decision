"""Generate the fixed checkpoint scoring asset from macOS arm64 NumPy 2.4.6.

Run with the pinned environment: python generate_scoring_asset.py decision_config.json output.bin
The reference is this NumPy build, not a claim that every platform evaluates identically.
"""

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EXPECTED = "8976812e52bc591dee1a1b891d427b16eb0bf641ea61299a518276d35e6da3ad"


def read_exact(stream, count):
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            raise RuntimeError("Portable scoring stream ended early")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def generate(config_path, output_path):
    config = json.loads(config_path.read_text())
    if (
        sys.platform != "darwin"
        or platform.machine() != "arm64"
        or np.__version__ != "2.4.6"
        or config["temperature"] != 1.5456
        or config["final_logit_softcapping"] != 30
    ):
        raise RuntimeError(
            "Use macOS arm64, NumPy 2.4.6, and the pinned Gemma v2 decision configuration"
        )
    bits = np.arange(65536, dtype=np.uint32) << np.uint32(16)
    with np.errstate(invalid="ignore", over="ignore"):
        scores = (
            np.tanh(bits.view(np.float32) / np.float32(30))
            * np.float32(30)
            / np.float32(1.5456)
        )
    unique = np.unique(scores[np.isfinite(scores)])
    unique.sort()
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "scores.f32le"
        unique.astype("<f4").tofile(source)
        stream_script = Path(__file__).with_name("scoring-reference-stream.mts")
        process = subprocess.Popen(
            ["node", "--import", "tsx", str(stream_script), str(source)],
            stdout=subprocess.PIPE,
            bufsize=0,
            cwd=Path(__file__).resolve().parents[3],
        )
        parts = []
        for maximum in range(len(unique)):
            shifts = unique[: maximum + 1] - unique[maximum]
            portable = np.frombuffer(
                read_exact(process.stdout, len(shifts) * 4), dtype="<u4"
            )
            reference = np.exp(shifts).view(np.uint32)
            parts.append(np.unique(shifts[portable != reference].view(np.uint32)))
        if process.wait() != 0:
            raise RuntimeError("Portable scoring stream failed")
    keys = np.unique(np.concatenate(parts))
    values = np.exp(keys.view(np.float32)).view(np.uint32)
    table = np.column_stack((keys, values)).astype("<u4")
    result = scores.astype("<f4").tobytes() + table.tobytes()
    digest = hashlib.sha256(result).hexdigest()
    if digest != EXPECTED:
        raise RuntimeError(f"Reference differs from the pinned asset: {digest}")
    output_path.write_bytes(result)
    print(
        json.dumps(
            {
                "bytes": len(result),
                "sha256": digest,
                "scores": len(unique),
                "corrections": len(keys),
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: generate_scoring_asset.py decision_config.json output.bin"
        )
    generate(Path(sys.argv[1]), Path(sys.argv[2]))

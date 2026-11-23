"""Compare rebuild bytes without claiming that the builds were independent."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def describe(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        header = file.read(8)
        if header != b"\0asm\x01\0\0\0":
            raise ValueError(f"{path} is not a WebAssembly version 1 module")
        digest.update(header)
        size += len(header)
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument(
        "--expected-sha256",
        help="Artifact digest from the curated manifest, if available",
    )
    args = parser.parse_args()
    if args.expected_sha256 is not None and (
        len(args.expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.expected_sha256)
    ):
        parser.error("--expected-sha256 must be a lowercase SHA-256 digest")
    try:
        if args.first.samefile(args.second):
            raise ValueError("The two paths refer to the same file")
        first, second = describe(args.first), describe(args.second)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    matched = first["bytes"] == second["bytes"] and first["sha256"] == second["sha256"]
    expected_match = args.expected_sha256 is None or (
        first["sha256"] == args.expected_sha256
        and second["sha256"] == args.expected_sha256
    )
    print(
        json.dumps(
            {
                "first": first,
                "second": second,
                "sameBytes": matched,
                "expectedSha256": args.expected_sha256,
                "expectedMatch": expected_match if args.expected_sha256 else None,
            },
            indent=2,
        )
    )
    return 0 if matched and expected_match else 1


if __name__ == "__main__":
    raise SystemExit(main())

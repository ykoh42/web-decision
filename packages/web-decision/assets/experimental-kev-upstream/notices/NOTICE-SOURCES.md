# Candidate notice source map

All staged notice files are copied byte-for-byte from the pinned local toolchain trees except `MODEL-APACHE-2.0.txt`, which is downloaded from the cited immutable Hugging Face revision, and `PROVENANCE.md`, which is authored for this candidate.

- `MODEL-APACHE-2.0.txt`: `Qwen/Qwen3-0.6B-Base` at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; 11,343 bytes; SHA-256 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`. Source: https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
- Emscripten, musl, libc++, libc++abi, and Emscripten compiler-rt files: Emscripten 4.0.23 source tree at emsdk revision `d7b4130688bfa348392b921ab04015aa15c4f154`.
- TVM, TVM-FFI, DLPack, and TVM compiler-rt files: TVM revision `7b5acb32bd271342a24130f4ef439ff6c2e52c6c`, including TVM-FFI submodule `daf594da8e6950fdc02ae35fcf4e0ab2e59a4979`.
- `COMPILER-RT-HEADER-NOTICE.txt`: leading copyright/permission comment extracted from `3rdparty/compiler-rt/builtin_fp16.h` in that TVM tree.
- `PROVENANCE.md`: candidate-specific summary of pins, build provenance, test scope, and unresolved distribution gates.

This file is included in the npm package at `assets/experimental-kev-upstream/notices/NOTICE-SOURCES.md`. The separate prepared asset-pack review used a `candidate-pins.json` to verify its notice files, but that packager file is not included in the npm archive and is not a runtime dependency. These source records do not clear model terms or redistribution rights.

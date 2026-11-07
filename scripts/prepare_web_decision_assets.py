#!/usr/bin/env python3
"""Package the verified, locally built Kev Q4 TVM artifacts for a web server."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from kev06_asset_manifest import KEV06_MANIFEST_SHA256, canonical_manifest_sha256

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True, help='New output directory; must not already exist')
args = parser.parse_args()
if args.output.is_symlink():
    raise SystemExit(f'Refusing to overwrite existing path: {args.output}')
output = args.output.resolve()
if output.exists():
    raise SystemExit(f'Refusing to overwrite existing path: {output}')
source = ROOT / 'models/kev-0.6b-onnx'
cache = ROOT / 'models/tvm-kev-q4-dynamic'
wasm = ROOT / 'public/tvm-kev-q4-dynamic.wasm'
build = json.loads((ROOT / 'reports/tvm-q4-build-dynamic.json').read_text())
if build['status'] != 'success' or build['inputShape'] != [1, 'sequence_length'] or build['numberOfParameters'] != 394:
    raise SystemExit('The verified dynamic build report is missing or incompatible')
subprocess.run([sys.executable, str(ROOT / 'scripts/verify_q4_cache.py'), '--dynamic', '--check-only'], cwd=ROOT, check=True)
cache_manifest = json.loads((cache / 'tensor-cache.json').read_text())
if cache_manifest['metadata']['numParams'] != 394:
    raise SystemExit('Unexpected number of model parameters')
sources = {'model.wasm': wasm, 'tokenizer.json': source / 'tokenizer.json',
           'tokenizer_config.json': source / 'tokenizer_config.json', 'model_config.json': source / 'config.json',
           'weights/tensor-cache.json': cache / 'tensor-cache.json'}
for shard in cache_manifest['records']:
    name = shard['dataPath']
    if Path(name).name != name or not name.endswith('.bin'):
        raise SystemExit(f'Unsafe shard name: {name}')
    sources[f'weights/{name}'] = cache / name


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


checkpoint_file = source / 'onnx/model_q4f16.onnx_data'
if digest(checkpoint_file) != '777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b':
    raise SystemExit('Source checkpoint does not match the pinned SHA-256')
expected_source = {
    'onnx/model_q4f16.onnx': '26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8',
    'config.json': '40ce1c875acf016ab7742ecdc45d3bad0fe8c51d3ead0ca2263ec5eb4e96f42d',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5',
}
for name, expected in expected_source.items():
    if digest(source / name) != expected:
        raise SystemExit(f'Source model file does not match the pinned SHA-256: {name}')
if build.get('tvmVersion') != '0.27.0.post1' or build.get('wasmSha256') != digest(wasm):
    raise SystemExit('The WebAssembly executable does not match the pinned TVM build report; rebuild it.')
if build.get('embeddedBuildRoot') is not False or str(ROOT).encode() in wasm.read_bytes():
    raise SystemExit('The WebAssembly executable contains an absolute build directory; rebuild it with the preview setup script.')
if build.get('tensorCacheManifestSha256') != digest(cache / 'tensor-cache.json'):
    raise SystemExit('The tensor cache does not match the TVM build report; rebuild it.')

# Validate notice inputs before creating an output directory. These notices are
# separate from inference files and do not claim complete license clearance.
tvm = ROOT / '.toolchains/apache-tvm-src'
emscripten = ROOT / '.toolchains/emsdk/upstream/emscripten'
notice_sources = {
    'PROVENANCE.md': ROOT / 'scripts/asset-notices/PROVENANCE.md',
    'MODEL-APACHE-2.0.txt': ROOT / 'scripts/asset-notices/QWEN-LICENSE-APACHE-2.0.txt',
    'EMSCRIPTEN-LICENSE.txt': emscripten / 'LICENSE',
    'MUSL-COPYRIGHT.txt': emscripten / 'system/lib/libc/musl/COPYRIGHT',
    'LIBCXX-LICENSE.txt': emscripten / 'system/lib/libcxx/LICENSE.TXT',
    'LIBCXXABI-LICENSE.txt': emscripten / 'system/lib/libcxxabi/LICENSE.TXT',
    'EMSCRIPTEN-COMPILER-RT-LICENSE.txt': emscripten / 'system/lib/compiler-rt/LICENSE.TXT',
    'TVM-LICENSE.txt': tvm / 'LICENSE',
    'TVM-NOTICE.txt': tvm / 'NOTICE',
    'TVM-FFI-LICENSE.txt': tvm / '3rdparty/tvm-ffi/LICENSE',
    'TVM-FFI-NOTICE.txt': tvm / '3rdparty/tvm-ffi/NOTICE',
    'DLPACK-LICENSE.txt': tvm / '3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE',
    'COMPILER-RT-LICENSE.txt': tvm / 'licenses/LICENSE.builtin_fp16.txt',
}
expected_notice_hashes = {
    'MODEL-APACHE-2.0.txt': '832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e',
    'EMSCRIPTEN-LICENSE.txt': '620a78084fc7ca97c0b5dea9abf891f3ffcadfdbf305276f099c9c4e12fc1d86',
    'MUSL-COPYRIGHT.txt': 'f9bc4423732350eb0b3f7ed7e91d530298476f8fec0c6c427a1c04ade22655af',
    'LIBCXX-LICENSE.txt': '539dd7aed86e8a4f12cbdd0e6c50c189c7d74847e4fecc64ce2c6ee3a01da38b',
    'LIBCXXABI-LICENSE.txt': 'e2b35be49f7284a45b7baca8fc7b3ab7440e7902392b2528a457816b5bb2a15c',
    'EMSCRIPTEN-COMPILER-RT-LICENSE.txt': '1a8f1058753f1ba890de984e48f0242a3a5c29a6a8f2ed9fd813f36985387e8d',
}
for name, path in notice_sources.items():
    if not path.is_file(): raise SystemExit(f'Missing required notice source: {name}')
    if name in expected_notice_hashes and digest(path) != expected_notice_hashes[name]:
        raise SystemExit(f'Unexpected notice source: {name}')
header = (tvm / '3rdparty/compiler-rt/builtin_fp16.h').read_bytes()
if not header.startswith(b'/*') or b'*/' not in header:
    raise SystemExit('Unexpected compiler-rt header notice')

output.mkdir(parents=True)
files = {}
for name, path in sources.items():
    dest = output / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    source_hash = digest(path)
    if digest(dest) != source_hash:
        raise SystemExit(f'Copy failed integrity check: {name}')
    files[name] = {'sha256': source_hash, 'bytes': dest.stat().st_size}
manifest = {
    'formatVersion': 1,
    'model': 'kev-0.6b-q4f16',
    'backend': 'tvm-webgpu',
    'experimental': True,
    'maxStateTokens': 8192,
    'maxBranchTokens': 8192,
    'parameterCount': 394,
    'tvmVersion': build['tvmVersion'],
    'checkpoint': {'repository': 'onnx-community/kev-0.6b-ONNX', 'revision': '5a827091b3d58fe4fdfc02a5a5b2ce364047d678',
                   'weightSha256': '777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b'},
    'files': files,
}
if canonical_manifest_sha256(manifest) != KEV06_MANIFEST_SHA256:
    raise SystemExit('Prepared Kev 0.6B manifest does not match the reviewed package pin.')
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
# Keep distribution notices beside the assets. They are deliberately outside
# `manifest.files`: the inference loader totals that list as bytes it fetches.
# The model and executable bytes, manifest and cache keys stay unchanged.
notice_dir = output / 'notices'
notice_dir.mkdir()
for name, path in notice_sources.items():
    destination = notice_dir / name
    shutil.copyfile(path, destination)
    if digest(destination) != digest(path):
        raise SystemExit(f'Notice copy failed integrity check: {name}')
# Preserve the specific header's copyright and permission notice, not only the
# more general compiler-rt license file shipped in TVM's source tree.
header_notice = header[:header.index(b'*/') + 2] + b'\n'
(notice_dir / 'COMPILER-RT-HEADER-NOTICE.txt').write_bytes(header_notice)
print(f'Prepared {len(files)} verified files at {output}; {sum(f["bytes"] for f in files.values()):,} bytes', flush=True)

#!/usr/bin/env python3
"""Record pinned and resolved build versions without paths or personal input."""
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def run(*args):
    return subprocess.check_output(args,cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
with (root/'reports/tvm-q4-build-dynamic.json').open() as f: build=json.load(f)
packages=['apache-tvm','apache-tvm-ffi','numpy','onnx','onnxruntime','onnxsim','ml_dtypes','protobuf','typing_extensions','packaging','flatbuffers']
report={'python':platform.python_version(),'platform':platform.system(),'machine':platform.machine(),
        'packages':{p:importlib.metadata.version(p) for p in packages},
        'node':run('node','--version'),'npm':run('npm','--version'),'pnpm':run('pnpm','--version'),
        'emscripten':run('emcc','--version').splitlines()[0],
        'emsdkRevision':run('git','-C','.toolchains/emsdk','rev-parse','HEAD'),
        'tvmRevision':run('git','-C','.toolchains/apache-tvm-src','rev-parse','HEAD'),
        'tvmSubmodules':run('git','-C','.toolchains/apache-tvm-src','submodule','status','--recursive','3rdparty/tvm-ffi').splitlines(),
        'wasmSha256':build['wasmSha256'],'tensorCacheManifestSha256':build['tensorCacheManifestSha256'],
        'note':'Records resolved versions for this build; it is not a signature or a guarantee of byte-identical builds.'}
out=root/'reports/preview-build-environment.json'
if out.exists():raise SystemExit(f'Refusing to overwrite {out}')
out.write_text(json.dumps(report,indent=2)+'\n');print(f'Recorded build environment: {out.relative_to(root)}')

#!/usr/bin/env python3
"""Compile Kev's actual packed q4 weights with TVM WebGPU custom operators."""
import argparse,hashlib,json,time,traceback
import os
from pathlib import Path
import onnx,tvm
from tvm import relax
from tvm.contrib import tvmjs
from tvm.relax.frontend.onnx import from_onnx
import tvm_selective_params,tvm_narrow_int64,tvm_q4_operators
from safe_output import atomic_write,create_temporary_directory,open_directory,promote_directory,promote_file,remove_tree,require_absent
ROOT=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--seq-len',type=int,default=69);parser.add_argument('--dynamic',action='store_true');parser.add_argument('--diagnostic-dir',type=Path,help='New directory for intermediate IR snapshots');args=parser.parse_args();seq=args.seq_len
if args.diagnostic_dir:
 args.diagnostic_dir.mkdir(parents=True,exist_ok=False)
if args.dynamic:import tvm_dynamic_onnx
if not 1<=seq<=1024:raise SystemExit('seq-len must be in 1..1024 for this prototype')
suffix='dynamic' if args.dynamic else f'seq{seq}'
path=ROOT/f'models/kev-0.6b-onnx/onnx/model_q4f16_direct_{suffix}_u32.onnx'
root_fd=os.open(ROOT,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);models_fd=None;public_fd=None;reports_fd=None;stage_fd=None
try:
 models_fd=open_directory(ROOT,'models')
 public_fd=open_directory(ROOT,'public');reports_fd=open_directory(ROOT,'reports')
 wasm_name=f'tvm-kev-q4-{suffix}.wasm';cache_name=f'tvm-kev-q4-{suffix}';require_absent(public_fd,wasm_name);require_absent(models_fd,cache_name)
 stage_name,stage_fd=create_temporary_directory(root_fd,'.tvm-q4-stage')
except BaseException:
 for fd in (stage_fd,reports_fd,public_fd,models_fd,root_fd):
  if fd is not None:os.close(fd)
 raise
# TVM needs path strings for output. The checkout must not be mutated by a
# same-UID process while TVM writes into this private stage; final publication
# below is anchored to the open destination directory descriptors.
stage_root=ROOT/stage_name
staged_wasm=stage_root/wasm_name;staged_cache=stage_root/cache_name
wasm_output=ROOT/'public'/wasm_name;cache_output=ROOT/'models'/cache_name
report={'kind':'tvm-kev-q4-webgpu-build','tvmVersion':tvm.__version__,'model':str(path.relative_to(ROOT)),'inputShape':[1,'sequence_length' if args.dynamic else seq],'weightPrecision':'original q4 packed bytes, f16 scales','accumulation':'float32','target':'webgpu / wasm32','stages':[],'notes':['Only a uint32 view of the original bytes is used; no global FP16 weight expansion.','Shape and int32/boolean handling are restricted to this prototype.']}
start=time.perf_counter()
def stage(name):report['stages'].append({'name':name,'elapsedSeconds':time.perf_counter()-start});print(name,flush=True)
try:
 stage('Loading packed ONNX graph');model=onnx.load(path,load_external_data=True);onnx.checker.check_model=lambda _:None
 shape_dict=None if args.dynamic else {'input_ids':[1,seq],'attention_mask':[1,seq]}
 stage('Importing q4 operators into Relax');mod=from_onnx(model,shape_dict=shape_dict,dtype_dict={'input_ids':'int32','attention_mask':'int32'},keep_params_in_input=True);del model
 mod,params=relax.frontend.detach_params(mod);weights=params['main'];report['numberOfParameters']=len(weights);report['parameterBytes']=sum(int(x.numpy().nbytes) for x in weights)
 if args.diagnostic_dir:(args.diagnostic_dir/'import-ir.py').write_text(mod.script())
 if args.dynamic:
  text=mod.script();atomic_write(reports_fd,'tvm-q4-dynamic-import-ir.py',lambda output:output.write(text.encode()))
 target=tvm.target.Target('webgpu',host={'kind':'llvm','mtriple':'wasm32-unknown-unknown-wasm'})
 stage('Scheduling packed WebGPU kernels')
 if args.dynamic:
  from tvm.relax.backend.gpu_generic.pipeline import library_dispatch_passes,legalize_passes,dataflow_lower_passes,finalize_passes
  lowered=mod
  with target:
   for pipeline_pass in library_dispatch_passes(target)+legalize_passes(target)+dataflow_lower_passes(target)+finalize_passes(target):
    print('Dynamic pipeline:',pipeline_pass.info.name,flush=True)
    lowered=pipeline_pass(lowered)
 else:
  with target: lowered=relax.get_default_pipeline(target)(mod)
 if args.dynamic:
  from tvm_kev_shape_order import normalize
  lowered=normalize(lowered)
  report['shapeStoreOrdering']='canonical, with model-specific independence checks'
 if args.diagnostic_dir:
  (args.diagnostic_dir/'lowered-ir.py').write_text(lowered.script())
 q4=[]
 for gv,func in lowered.functions.items():
  if 'packed_q4_matmul' in gv.name_hint:
   script=func.script();q4.append({'name':gv.name_hint,'usesSharedMemory':('scope="shared"' in script),'hasPackedUint32':('"uint32"' in script),'hasGlobalDecodedBuffer':('q4_decoded' in script and 'scope="global"' in script)})
 report['q4KernelChecks']=q4
 if len(q4)<5 or any(not x['hasPackedUint32'] or x['hasGlobalDecodedBuffer'] for x in q4):raise ValueError('Packed kernel scheduling checks failed')
 stage('Compiling TVM WebGPU executable');ex=relax.build(lowered,target=target,relax_pipeline=None)
 stage('Linking WebAssembly runtime');ex.export_library(str(staged_wasm),fcompile=tvmjs.create_tvmjs_wasm)
 stage('Writing packed tensor cache')
 tvmjs.dump_tensor_cache(((f'param_{i}',v) for i,v in enumerate(weights)),str(staged_cache),encode_format='raw',show_progress=False,meta_data={'numParams':len(weights),'source':'original q4f16 checkpoint','shape':[1,'sequence_length' if args.dynamic else seq]})
 def sha256(path):
  digest=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
  return digest.hexdigest()
 report.update(status='success',wasm=str(wasm_output.relative_to(ROOT)),wasmBytes=staged_wasm.stat().st_size,wasmSha256=sha256(staged_wasm),embeddedBuildRoot=str(ROOT).encode() in staged_wasm.read_bytes(),tensorCache=str(cache_output.relative_to(ROOT)),tensorCacheManifestSha256=sha256(staged_cache/'tensor-cache.json'))
 promote_directory(stage_fd,cache_name,models_fd,cache_name)
 try:promote_file(stage_fd,wasm_name,public_fd,wasm_name)
 except BaseException:
  remove_tree(models_fd,cache_name)
  raise
except BaseException as e:report.update(status='failed',errorType=type(e).__name__,error=str(e),traceback=traceback.format_exc())
report['elapsedSeconds']=time.perf_counter()-start;report_name='tvm-q4-build-dynamic.json' if args.dynamic else ('tvm-q4-build.json' if seq==69 else f'tvm-q4-build-seq{seq}.json');report_text=json.dumps(report,indent=2)+'\n'
try:atomic_write(reports_fd,report_name,lambda output:output.write(report_text.encode()))
finally:
 os.close(stage_fd)
 try:remove_tree(root_fd,stage_name)
 finally:
  os.close(root_fd);os.close(models_fd);os.close(public_fd);os.close(reports_fd)
print(json.dumps(report,indent=2),flush=True)
if report['status']!='success':raise SystemExit(1)

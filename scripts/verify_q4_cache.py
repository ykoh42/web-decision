#!/usr/bin/env python3
"""Verify every compiled TVM q4 tensor against the source checkpoint bytes."""
import argparse,hashlib,json
import os
from pathlib import Path
import onnx
from safe_output import create_new_file,open_directory,require_absent
ROOT=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--seq-len',type=int,default=69);parser.add_argument('--dynamic',action='store_true');parser.add_argument('--check-only',action='store_true',help='Verify and print without writing a report');args=parser.parse_args();seq=args.seq_len
suffix='dynamic' if args.dynamic else f'seq{seq}'
report_name='q4-cache-verification-dynamic.json' if args.dynamic else ('q4-cache-verification.json' if seq==69 else f'q4-cache-verification-seq{seq}.json')
out=ROOT/'reports'/report_name
reports_fd=None
if not args.check_only:
 reports_fd=open_directory(ROOT,'reports')
 try:require_absent(reports_fd,report_name)
 except FileExistsError:
  os.close(reports_fd);raise SystemExit(f'Refusing to overwrite {out}')
folder=ROOT/'models/kev-0.6b-onnx/onnx';cache=ROOT/f'models/tvm-kev-q4-{suffix}'
model=onnx.load(folder/f'model_q4f16_direct_{suffix}_u32.onnx',load_external_data=False)
quant=[t for t in model.graph.initializer if t.name.startswith(('q4_packed_','q4_scale_'))]
manifest=json.loads((cache/'tensor-cache.json').read_text());records={r['name']:(shard['dataPath'],r) for shard in manifest['records'] for r in shard['records']}
assert len(quant)==manifest['metadata']['numParams']==394

def digest_part(path,offset,length):
 h=hashlib.sha256()
 with path.open('rb') as f:
  f.seek(offset);remaining=length
  while remaining:
   b=f.read(min(1024*1024,remaining))
   if not b:raise ValueError(f'unexpected end of {path}')
   h.update(b);remaining-=len(b)
 return h.hexdigest()

mismatches=[];total=0
for i,t in enumerate(quant):
 entry={x.key:x.value for x in t.external_data};src=folder/entry['location'];offset=int(entry['offset']);length=int(entry['length']);data_path,record=records[f'param_{i}'];total+=length
 if int(record['nbytes'])!=length or record['format']!='raw' or digest_part(src,offset,length)!=digest_part(cache/data_path,int(record['byteOffset']),length):mismatches.append({'param':i,'source':t.name})
report={'kind':'q4-cache-byte-verification','source':'original model_q4f16.onnx_data','parameters':len(quant),'totalBytesChecked':total,'mismatchCount':len(mismatches),'mismatches':mismatches,'method':'SHA-256 of each tensor byte range versus its TVM cache record','note':'A uint32 view repacks no data; the cache bytes match the source packed weights and scales.'}
if not args.check_only:
 try:create_new_file(reports_fd,report_name,lambda stream:stream.write((json.dumps(report,indent=2)+'\n').encode()))
 finally:os.close(reports_fd)
print(json.dumps(report,indent=2))
if mismatches:raise SystemExit(1)

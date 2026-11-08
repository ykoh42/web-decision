#!/usr/bin/env python3
"""Specialize Kev q4 ONNX without ever expanding checkpoint weights.

During shape-only simplification, quantized operators are represented by
shape-only FP16 graph inputs. They have no tensor contents and are never
allocated. After simplification, the original packed operators and external
q4 bytes/scales are restored. A second graph views the same bytes as uint32 for
TVM's WebGPU kernels.
"""
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import os
import numpy as np,onnx,onnxsim
from onnx import TensorProto,external_data_helper,helper,numpy_helper
from safe_output import atomic_write,atomic_write_serialized_message,open_directory
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'models/kev-0.6b-onnx/onnx';source=folder/'model_q4f16.onnx';started=time.perf_counter()
parser=argparse.ArgumentParser();parser.add_argument('--seq-len',type=int,default=69);parser.add_argument('--dynamic',action='store_true');args=parser.parse_args();seq=args.seq_len
if not 1<=seq<=1024:raise SystemExit('seq-len must be in 1..1024 for this prototype')
model_output_fd=open_directory(ROOT,'models/kev-0.6b-onnx/onnx');report_output_fd=open_directory(ROOT,'reports')
original=onnx.load(source,load_external_data=False);model=copy.deepcopy(original)
initializers={t.name:t for t in original.graph.initializer};ops={};removed=set();placeholders=[];nodes=[];counts={}
constants=[numpy_helper.from_array(np.array([-1],np.int64),'__expanded_norm_axis'),numpy_helper.from_array(np.array(1e-6,np.float32),'__expanded_norm_epsilon')]
for index,node in enumerate(original.graph.node):
 counts[node.op_type]=counts.get(node.op_type,0)+1
 if node.op_type=='MatMulNBits' and node.domain=='com.microsoft':
  attrs={a.name:helper.get_attribute_value(a) for a in node.attribute};n,k,block=attrs['N'],attrs['K'],attrs['block_size'];assert len(node.input)==3 and attrs['bits']==4 and k%block==0
  name=f'__expanded_weight_{index}';placeholders.append(helper.make_tensor_value_info(name,TensorProto.FLOAT16,[k,n]));nodes.append(helper.make_node('MatMul',[node.input[0],name],list(node.output),name=node.name+'_expanded'));removed.update(node.input[1:]);ops[node.name]=node
 elif node.op_type=='GatherBlockQuantized' and node.domain=='com.microsoft':
  attrs={a.name:helper.get_attribute_value(a) for a in node.attribute};assert len(node.input)==3 and attrs['block_size']==32 and attrs['gather_axis']==0 and attrs['quantize_axis']==1
  rows,cols=initializers[node.input[0]].dims;name=f'__expanded_embedding_{index}';placeholders.append(helper.make_tensor_value_info(name,TensorProto.FLOAT16,[rows,cols]));nodes.append(helper.make_node('Gather',[name,node.input[1]],list(node.output),name=node.name+'_expanded',axis=0));removed.update([node.input[0],node.input[2]]);ops[node.name]=node
 elif node.op_type=='SimplifiedLayerNormalization':
  attrs={a.name:helper.get_attribute_value(a) for a in node.attribute};assert attrs.get('axis',-1)==-1 and abs(attrs.get('epsilon',1e-5)-1e-6)<1e-10 and len(node.output)==1
  p=f'__expanded_norm_{index}';x,w=node.input
  nodes.extend([helper.make_node('Cast',[x],[p+'_x'],to=TensorProto.FLOAT),helper.make_node('Cast',[w],[p+'_w'],to=TensorProto.FLOAT),helper.make_node('Mul',[p+'_x',p+'_x'],[p+'_square']),helper.make_node('ReduceMean',[p+'_square','__expanded_norm_axis'],[p+'_mean'],keepdims=1),helper.make_node('Add',[p+'_mean','__expanded_norm_epsilon'],[p+'_variance']),helper.make_node('Sqrt',[p+'_variance'],[p+'_root']),helper.make_node('Div',[p+'_x',p+'_root'],[p+'_normalized']),helper.make_node('Mul',[p+'_normalized',p+'_w'],[p+'_scaled']),helper.make_node('Cast',[p+'_scaled'],list(node.output),to=TensorProto.FLOAT16)])
 else:nodes.append(copy.deepcopy(node))
assert len(placeholders)==197 and len(ops)==197
kept=[]
for t in original.graph.initializer:
 if t.name in removed:continue
 t=copy.deepcopy(t)
 if t.data_location==TensorProto.EXTERNAL:
  info={x.key:x.value for x in t.external_data};assert int(info['length'])<1024*1024
  external_data_helper.load_external_data_for_tensor(t,str(folder));t.ClearField('external_data');t.data_location=TensorProto.DEFAULT
 kept.append(t)
del model.graph.initializer[:];model.graph.initializer.extend(kept+constants)
del model.graph.node[:];model.graph.node.extend(nodes)
model.graph.input.extend(placeholders)
for inp in model.graph.input:
 if inp.name in ('input_ids','attention_mask'):
  batch_dim,seq_dim=inp.type.tensor_type.shape.dim
  batch_dim.ClearField('dim_param');batch_dim.dim_value=1
  if args.dynamic:
   seq_dim.ClearField('dim_value');seq_dim.dim_param='sequence_length'
  else:
   seq_dim.ClearField('dim_param');seq_dim.dim_value=seq
print(f'Simplifying shapes using {len(placeholders)} content-free placeholders; no dense weights are loaded.',flush=True)
simplified,checked=onnxsim.simplify(model,check_n=0,skip_fuse_bn=True)
assert checked
placeholder_names={x.name for x in placeholders};inputs={x.name for x in simplified.graph.input};assert placeholder_names<=inputs
selected=[n for n in simplified.graph.node if n.name.endswith('_Q4_expanded')];assert len(selected)==197

def restore(as_u32):
 m=copy.deepcopy(simplified);newnodes=[];names=set();rename={};counter=0
 for node in m.graph.node:
  if not node.name.endswith('_Q4_expanded'):newnodes.append(node);continue
  source_node=ops[node.name[:-9]];q=copy.deepcopy(source_node)
  if source_node.op_type=='MatMulNBits':q.input[0]=node.input[0];q.output[0]=node.output[0];indexes=[1,2]
  else:q.input[1]=node.input[1];q.output[0]=node.output[0];indexes=[0,2]
  for j in indexes:
   old=source_node.input[j];role='packed' if j==(1 if source_node.op_type=='MatMulNBits' else 0) else 'scale'
   if old not in rename:rename[old]=f'q4_{role}_{len(rename):03d}'
   q.input[j]=rename[old];names.add(old)
  q.name=f'q4_{source_node.op_type}_{counter:03d}';counter+=1;newnodes.append(q)
 keep=list(m.graph.initializer)
 for name in sorted(names,key=lambda n:rename[n]):
  t=copy.deepcopy(initializers[name]);t.name=rename[name]
  if as_u32 and t.name.startswith('q4_packed_'):
   ext={x.key:x.value for x in t.external_data};length=int(ext['length']);assert length%4==0
   if t.data_type==TensorProto.INT4:rows,cols=t.dims;assert cols%8==0;t.dims[:]=[rows,cols//8]
   else:assert t.data_type==TensorProto.UINT8;rows,groups,size=t.dims;assert (groups*size)%4==0;t.dims[:]=[rows,groups*size//4]
   t.data_type=TensorProto.UINT32;assert math.prod(t.dims)*4==length
  keep.append(t)
 del m.graph.node[:];m.graph.node.extend(newnodes)
 del m.graph.initializer[:];m.graph.initializer.extend(keep)
 inps=[x for x in m.graph.input if x.name not in placeholder_names];del m.graph.input[:];m.graph.input.extend(inps)
 used={x for n in m.graph.node for x in n.input};assert not (used&placeholder_names)
 suffix='dynamic' if args.dynamic else f'seq{seq}'
 name=f'model_q4f16_direct_{suffix}_u32.onnx' if as_u32 else f'model_q4f16_direct_{suffix}.onnx';atomic_write_serialized_message(model_output_fd,name,m);return name
ort=restore(False);tvm=restore(True)
report={'kind':'q4-direct-specialization','source':str(source.relative_to(ROOT)),'onnxGraph':ort,'tvmGraph':tvm,'inputShape':[1,'sequence_length' if args.dynamic else seq],'quantizedOperators':len(ops),'rmsnormOperatorsExpanded':counts['SimplifiedLayerNormalization'],'placeholderCount':len(placeholders),'simplifiedNodeCount':len(simplified.graph.node),'denseWeightFilesCreated':0,'checkpointWeightBytesReadByConverter':0,'elapsedSeconds':time.perf_counter()-started,'note':'Shape-only graph inputs carry no values. Both outputs reuse the original packed external weights and scales.'}
report_name='q4-direct-dynamic.json' if args.dynamic else ('q4-direct-specialization.json' if seq==69 else f'q4-direct-specialization-seq{seq}.json')
report_text=json.dumps(report,indent=2)+'\n';atomic_write(report_output_fd,report_name,lambda output:output.write(report_text.encode()));os.close(model_output_fd);os.close(report_output_fd);print(json.dumps(report,indent=2),flush=True)

"""Experimental TVM Relax converters for Kev's two packed ONNX q4 operators.

The input packed tensor is a uint32 view of the checkpoint's bytes, with eight
4-bit values in each word. We unpack only into a workgroup tile inside the
matmul kernel; no global dense-weight tensor is materialized.
"""
from tvm import te,tirx
from tvm.relax.frontend.onnx import onnx_frontend as frontend


def matmul_te(x,packed,scales,block_size=32):
    b,m,k=x.shape;k=int(k);n=int(packed.shape[0])
    assert int(packed.shape[1])==k//8 and int(scales.shape[1])==k//block_size
    red=te.reduce_axis((0,k),'k')
    def unpack(col,ix):
        q=tirx.bitwise_and(tirx.shift_right(packed[col,ix//8],tirx.Cast('uint32',(ix%8)*4)),tirx.const(15,'uint32'))
        return (tirx.Cast('float32',q)-tirx.const(8,'float32'))*tirx.Cast('float32',scales[col,ix//block_size])
    decoded=te.compute((n,k),unpack,name='q4_decoded')
    accum=te.compute((b,m,n),lambda batch,row,col:te.sum(tirx.Cast('float32',x[batch,row,red])*decoded[col,red],axis=red),name='q4_acc')
    return te.compute((b,m,n),lambda batch,row,col:tirx.Cast('float16',accum[batch,row,col]),name='q4_matmul_output')


def gather_te(packed,ids,scales,block_size=32):
    batch,length=ids.shape;rows,words=map(int,packed.shape);hidden=words*8
    assert int(scales.shape[0])==rows and int(scales.shape[1])==hidden//block_size
    def value(b,t,k):
        token=ids[b,t];q=tirx.bitwise_and(tirx.shift_right(packed[token,k//8],tirx.Cast('uint32',(k%8)*4)),tirx.const(15,'uint32'))
        signed=tirx.if_then_else(q<tirx.const(8,'uint32'),tirx.Cast('int32',q),tirx.Cast('int32',q)-tirx.const(16,'int32'))
        return tirx.Cast('float16',signed)*scales[token,k//block_size]
    return te.compute((batch,length,hidden),value,name='q4_gather_output')


class MatMulNBits(frontend.OnnxOpConverter):
    @classmethod
    def _impl_v1(cls,bb,inputs,attrs,params):
        assert len(inputs)==3 and attrs['bits']==4 and attrs['block_size']==32
        assert int(inputs[0].ty.shape[-1])==attrs['K'] and int(inputs[1].ty.shape[0])==attrs['N']
        return bb.emit_te(matmul_te,inputs[0],inputs[1],inputs[2],primfunc_name_hint='packed_q4_matmul')


class GatherBlockQuantized(frontend.OnnxOpConverter):
    @classmethod
    def _impl_v1(cls,bb,inputs,attrs,params):
        assert len(inputs)==3 and attrs['block_size']==32 and attrs['gather_axis']==0 and attrs['quantize_axis']==1
        return bb.emit_te(gather_te,inputs[0],inputs[1],inputs[2],primfunc_name_hint='packed_q4_gather')

original_map=frontend._get_convert_map

def patched_map():
    mapping=original_map()
    mapping['MatMulNBits']=MatMulNBits
    mapping['GatherBlockQuantized']=GatherBlockQuantized
    return mapping

frontend._get_convert_map=patched_map

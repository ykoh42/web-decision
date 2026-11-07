#!/usr/bin/env python3
"""Exercise the fail-closed assumptions of the Kev shape ordering."""
import unittest
import tvm
from tvm_kev_shape_order import normalize

def module(indices, dependency=False, unsupported=False):
 lines=['@I.ir_module','class Module:','    @T.prim_func(s_tir=True)',
        '    def shape_func(H: T.Buffer((T.int64(13),), "int64")):',
        '        T.func_attr({"tirx.is_host_func": True})']
 for i in indices:
  source=1 if dependency and i==12 else 0
  op='+' if unsupported and i==12 else '*'
  lines.append(f'        H[T.int64({i})] = T.int64(2) {op} H[T.int64({source})]')
 return tvm.script.from_source('\n'.join(lines)+'\n')

class CanonicalShapeTest(unittest.TestCase):
 def test_sorts_independent_stores_without_changing_expressions(self):
  mod=module(list(range(12,0,-1)));before={int(s.indices[0]):str(s.value) for s in mod['shape_func'].body.seq}
  normalize(mod);after=list(mod['shape_func'].body.seq)
  self.assertEqual([int(s.indices[0]) for s in after],list(range(1,13)))
  self.assertEqual({int(s.indices[0]):str(s.value) for s in after},before)
 def test_rejects_store_depending_on_another_output(self):
  with self.assertRaisesRegex(ValueError,'unknown dependencies'):normalize(module(list(range(1,13)),dependency=True))
 def test_rejects_duplicate_or_missing_slots(self):
  with self.assertRaisesRegex(ValueError,'indexes'):normalize(module(list(range(2,13))+[2]))
 def test_rejects_unrecognized_expression(self):
  with self.assertRaisesRegex(ValueError,'unknown dependencies'):normalize(module(list(range(1,13)),unsupported=True))
 def test_rejects_multiple_buffer_parameters(self):
  lines=['@I.ir_module','class Module:','    @T.prim_func(s_tir=True)',
         '    def shape_func(H: T.Buffer((T.int64(13),), "int64"), J: T.Buffer((T.int64(13),), "int64")):',
         '        T.func_attr({"tirx.is_host_func": True})']
  lines.extend(f'        H[T.int64({i})] = T.int64(2) * H[T.int64(0)]' for i in range(1,13))
  with self.assertRaisesRegex(ValueError,'parameter'):normalize(tvm.script.from_source('\n'.join(lines)+'\n'))

if __name__=='__main__':unittest.main()

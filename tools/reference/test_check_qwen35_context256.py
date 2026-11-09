import copy
import hashlib
import struct
import unittest
from unittest import mock
import check_qwen35_context256 as check


def partitions():
    return [dict(chunk=n,calls=check.partition_calls(n),full_logits_bits_exact=True,
                 checkpoint_state_bits_exact=True,warmed_rust_allocations=0,forward_ns=1) for n in [32,17,1]]


class Partitions(unittest.TestCase):
    def test_checkpoint_boundary_splits_irregular_chunks_without_truncation(self):
        self.assertEqual(check.partition_calls(17),([17]*7+[9])*2)
        self.assertEqual(check.partition_calls(32),[32]*8)
        self.assertEqual(check.partition_calls(1),[1]*256)
        check.validate_partitions(partitions())

    def test_unknown_or_boolean_partition_does_not_qualify(self):
        for value in [True,False,0,16,33,'17',17.0]:
            with self.assertRaises(ValueError):check.partition_calls(value)

    def test_false_bit_or_allocation_flags_and_changed_schedules_reject(self):
        for key,value in [('full_logits_bits_exact',False),('checkpoint_state_bits_exact',1),
                          ('warmed_rust_allocations',False),('calls',[32]*8),('forward_ns',True)]:
            rows=partitions();rows[1][key]=value
            with self.assertRaises(ValueError):check.validate_partitions(rows)

    def test_all_three_schedules_are_required_once(self):
        rows=partitions()
        for bad in [rows[:-1],rows+[rows[-1]],list(reversed(rows))]:
            with self.assertRaises(ValueError):check.validate_partitions(bad)


class Arrays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wanted={(None,f'unit_stage{i}',None):1 for i in range(1051)}
        raw=bytearray();rows=[]
        for profile,fmt,width in [('ideal_fp64','<d',8),('staged_fp32','<f',4)]:
            for n in [128,256]:
                for i in range(1051):
                    block=struct.pack(fmt,i/8)
                    rows.append(dict(checkpoint=n,profile=profile,token=n-1,layer=None,stage=f'unit_stage{i}',
                                     count=1,offset=len(raw),bytes=width,sha256=hashlib.sha256(block).hexdigest()))
                    raw.extend(block)
        cls.raw=bytes(raw);cls.rows=rows

    def parse(self,rows,raw=None):
        with mock.patch.object(check.prior,'expected',return_value=self.wanted), \
             mock.patch.object(check.producer,'plan',return_value=([1]*256,0,{128:1051,256:1051})):
            return check.reference_rows(dict(arrays=rows),self.raw if raw is None else raw)

    def test_each_profile_and_checkpoint_keeps_its_full_unique_stage_table(self):
        result=self.parse(self.rows)
        self.assertEqual(set(result),{(p,n) for p in ['ideal_fp64','staged_fp32'] for n in [128,256]})
        self.assertEqual(float(result['staged_fp32',256][None,'unit_stage1050',None][0]),131.25)

    def test_missing_duplicate_and_mislocated_checkpoint_arrays_reject(self):
        for field,value in [('stage','unit_stage0'),('checkpoint',True),('token',True),('profile','staged_fp32'),('offset',0)]:
            rows=copy.deepcopy(self.rows);rows[1][field]=value
            with self.assertRaises(ValueError):self.parse(rows)
        with self.assertRaises(ValueError):self.parse(self.rows[:-1])

    def test_nonfinite_values_still_reject_after_metadata_hash_is_resealed(self):
        rows=copy.deepcopy(self.rows);raw=bytearray(self.raw);block=struct.pack('<d',float('nan'))
        raw[:8]=block;rows[0]['sha256']=hashlib.sha256(block).hexdigest()
        with self.assertRaises(ValueError):self.parse(rows,bytes(raw))

    def test_trailing_or_mutated_body_cannot_pass_stage_hashes(self):
        with self.assertRaises(ValueError):self.parse(self.rows,self.raw+b'x')
        raw=bytearray(self.raw);raw[-1]^=1
        with self.assertRaises(ValueError):self.parse(self.rows,bytes(raw))


class OwnedProcess(unittest.TestCase):
    def setUp(self):
        self.wrapper=dict(pid=21,group=21,parent_pid=20)
        self.child=dict(pid=22,group=21,parent_pid=21)

    def test_instrument_child_matches_its_parent_and_group(self):
        check.owned_reference_process(self.child,self.wrapper)

    def test_unrelated_child_or_parent_cannot_inherit_the_group_claim(self):
        for field,value in [('parent_pid',99),('group',99),('pid',21),('pid',20)]:
            row={**self.child,field:value}
            with self.assertRaises(ValueError):check.owned_reference_process(row,self.wrapper)

    def test_boolean_or_nonpositive_identity_cannot_match_an_integer(self):
        for field in self.child:
            for value in [True,False,0,-1,'21',21.0]:
                row={**self.child,field:value}
                with self.assertRaises(ValueError):check.owned_reference_process(row,self.wrapper)

    def test_complete_typed_child_schema_is_required(self):
        for row in [None,[],{**self.child,'unknown':22},{'pid':22,'group':21}]:
            with self.assertRaises(ValueError):check.owned_reference_process(row,self.wrapper)


class RuntimeOrigins(unittest.TestCase):
    def setUp(self):
        self.pin=dict(bytes=1,sha256='a'*64)
        self.runtime=dict(files={path:self.pin.copy() for path in ['/unit/numpy/root.py','/unit/numpy/core.so','/unit/mlx/core.so','/unit/mlx/nn.py']})
        self.native=dict(files={'/unit/Python':self.pin.copy()})
        self.imports={'numpy':'/unit/numpy/root.py','mlx.core':'/unit/mlx/core.so','mlx.nn':'/unit/mlx/nn.py'}
        self.endpoint=dict(numpy_modules={'numpy':'/unit/numpy/root.py','numpy._core._multiarray_umath':'/unit/numpy/core.so'},
                           non_system_native_images={'/unit/Python':self.pin.copy()},system_native_images=['/usr/lib/libSystem.B.dylib'])

    def validate(self):check.runtime_origins(self.endpoint,self.imports,self.runtime,self.native)

    def test_exact_numpy_root_core_and_native_images_are_admitted(self):
        self.validate()

    def test_nonempty_fake_module_map_is_not_actual_numpy_coverage(self):
        self.endpoint['numpy_modules']={'other':'/unit/numpy/root.py'}
        with self.assertRaises(ValueError):self.validate()

    def test_admitted_file_cannot_impersonate_another_module_root(self):
        self.endpoint['numpy_modules']['numpy']='/unit/numpy/core.so'
        with self.assertRaises(ValueError):self.validate()

    def test_typed_native_pin_cannot_use_bool_as_a_byte_count(self):
        self.endpoint['non_system_native_images']['/unit/Python']['bytes']=True
        with self.assertRaises(ValueError):self.validate()

    def test_unknown_image_and_mislabeled_system_image_reject(self):
        self.endpoint['non_system_native_images']['/unit/rogue']=self.pin
        with self.assertRaises(ValueError):self.validate()
        del self.endpoint['non_system_native_images']['/unit/rogue']
        self.endpoint['system_native_images']=['/unit/rogue']
        with self.assertRaises(ValueError):self.validate()

    def test_malformed_endpoint_inventory_has_a_validation_error(self):
        for field,value in [('numpy_modules',True),('non_system_native_images',[]),('system_native_images',False)]:
            old=self.endpoint[field];self.endpoint[field]=value
            with self.assertRaises(ValueError):self.validate()
            self.endpoint[field]=old


if __name__=='__main__':unittest.main()

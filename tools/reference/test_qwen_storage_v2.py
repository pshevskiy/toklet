"""Independent v2 stripe reconstruction and provenance boundary checks."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import weakref
from unittest import mock

import numpy as np
import qwen_storage_v2 as storage
import qwen_storage_v2_oracle as oracle


class GroupEightStorage(unittest.TestCase):
    def test_manual_bytes_reconstruct_and_invalid_versions_metadata_padding_reject(self):
        root=Path(__file__).resolve().parents[2]/'tests/fixtures/q4-v2'
        data=(root/'rows.q4').read_bytes();expected=json.loads((root/'manifest.json').read_text())['reconstructed_rows']
        np.testing.assert_array_equal(storage.decode_q4(data,2,17),expected)
        for offset in (0,8,12,32,40,48,76,85,88):
            corrupt=bytearray(data);corrupt[offset]^=1
            with self.assertRaises(ValueError):storage.decode_q4(corrupt,2,17)
        for offset,bits in ((64,0x8000),(64,0x7c00),(66,0x7e00)):
            corrupt=bytearray(data);corrupt[offset:offset+2]=bits.to_bytes(2,'little')
            with self.assertRaises(ValueError):storage.decode_q4(corrupt,2,17)
        for data2 in (data[:63],data[:-1],data+b'\0'):
            with self.assertRaises(ValueError):storage.decode_q4(data2,2,17)

    def test_stripes_cover_partial_groups_alignment_and_decode_row_boundary(self):
        for rows,columns in ((1,7),(2,8),(2,9),(129,65)):
            stride=((columns+7)//8*8+63)//64*64
            data=bytearray(struct.pack('<8sIIQQQQ16x',b'TOKQ4V2\0',2,8,rows,columns,stride,rows*stride))
            expected=[]
            for row in range(rows):
                body=bytearray(stride)
                for col in range(columns):
                    start=col//8*8;struct.pack_into('<ee',body,start,.125,-.5)
                    body[start+4+col%8//2]|=(col%16)<<(4*(col%2))
                data.extend(body);expected.append([.125*(col%16)-.5 for col in range(columns)])
            np.testing.assert_array_equal(storage.decode_q4(data,rows,columns),expected)

    def test_exact_consumed_manifest_bytes_are_hashed_before_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest';before=b'{"old":true}';after=b'{"new":true}';path.write_bytes(before)
            class Replaced(io.BytesIO):
                def __exit__(self,*args):
                    with open(path,'wb') as stream:stream.write(after)
                    return super().__exit__(*args)
            pin=hashlib.sha256(before).hexdigest()
            with mock.patch.object(Path,'open',return_value=Replaced(before)):
                data,identity=storage.bounded_bytes(path,100,pin)
            self.assertEqual((data,identity),(before,pin));self.assertNotEqual(storage.sha(path),identity)

    def test_tied_half_table_is_single_cached_allocation_and_descriptor_endpoint_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.safetensors';words=np.asarray([0x3f80,0xbf80,0,0x3e80]*1025,dtype='<u2')
            header=json.dumps({'model.embed_tokens.weight':{'dtype':'BF16','shape':[1025,4],'data_offsets':[0,words.nbytes]}}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+words.tobytes())
            with storage.Weights(path,storage.sha(path)) as reader:
                table=reader['embed_tokens'];self.assertIs(table,reader['embed_tokens']);np.testing.assert_array_equal(table,np.tile([1,-1,0,.25],(1025,1)));reader.verify()
                saved=path.with_name('old');path.rename(saved);path.write_bytes(saved.read_bytes());saved.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'descriptor endpoint'):reader.verify()
            self.assertFalse(reader.cached)

    def test_publication_requires_completed_native_memory_and_explicit_packed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'output';out.mkdir()
            with mock.patch.object(oracle.native_usage,'usage',return_value={'lifetime_max_phys_footprint':7*1024**3}),self.assertRaises(ValueError):oracle.publish_manifest(out,{}, {'native_samples':[]})
            self.assertFalse((out/'manifest.json').exists());self.assertTrue((out/'.manifest.pending').exists())
            with mock.patch.object(oracle.storage,'bounded_bytes') as read,self.assertRaises(ValueError):oracle.export(Path('Qwen2.5-1.5B-Instruct'),Path('unused'),None,out,Path(tmp)/'receipt')
            read.assert_not_called()

    def test_prior_expanded_stripe_dies_before_next_allocation(self):
        rows,columns=129,9;stride=64
        data=struct.pack('<8sIIQQQQ16x',b'TOKQ4V2\0',2,8,rows,columns,stride,rows*stride)+bytes(rows*stride)
        expanded_refs=[];real_empty=np.empty
        class Codes(np.ndarray):
            def astype(self,*args,**kwargs):
                self_outer.assertTrue(all(ref() is None for ref in expanded_refs),'previous expanded stripe overlaps next')
                kwargs['subok']=False;result=super().astype(*args,**kwargs)
                expanded_refs.append(weakref.ref(result));return result
        self_outer=self
        def empty(shape,*args,**kwargs):
            value=real_empty(shape,*args,**kwargs)
            return value.view(Codes) if len(shape)==3 else value
        with mock.patch.object(storage.np,'empty',empty):actual=storage.decode_q4(data,rows,columns)
        self.assertEqual(len(expanded_refs),2);self.assertTrue(all(ref() is None for ref in expanded_refs))
        np.testing.assert_array_equal(actual,np.zeros((rows,columns)))

    def test_packed_manifest_version_exclusions_inventory_and_header_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'model.safetensors';config=root/'config.json';config.write_bytes(b'{}')
            name='model.layers.0.self_attn.q_proj.weight';words=np.asarray(list(range(8))+[0],dtype='<f4')
            bf16=(words.view(np.uint32)>>16).astype('<u2').tobytes()*2
            header=json.dumps({name:{'dtype':'BF16','shape':[2,9],'data_offsets':[0,len(bf16)]}}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+bf16)
            packed=root/'packed';packed.mkdir();payload=bytearray(64)
            struct.pack_into('<ee',payload,0,1,0);payload[4:8]=bytes([0x10,0x32,0x54,0x76])
            data=struct.pack('<8sIIQQQQ16x',b'TOKQ4V2\0',2,8,2,9,64,128)+payload*2
            (packed/(name+'.q4')).write_bytes(data)
            manifest={'format':'toklet-q4-model-v2','quantizer':'toklet-q4-affine-f32-source-half-metadata-rne-g8-v2','config_sha256':storage.sha(config),
                      'sources':[{'file':path.name,'sha256':storage.sha(path)}],
                      'matrices':[{'name':name,'file':name+'.q4','rows':2,'columns':9,'packed_bytes':len(data),'packed_sha256':hashlib.sha256(data).hexdigest(),'source_f32_sha256':hashlib.sha256(words.tobytes()*2).hexdigest()}],
                      'exclusions':['norms','qkv_biases','tied_embedding_output'],'excluded_dtype':'F16','packed_matrix_bytes':len(data)}
            saved=json.loads(json.dumps(manifest))
            for mutation in ('version','quantizer','exclusions','source','duplicate','missing','remap','shape'):
                altered=json.loads(json.dumps(saved))
                if mutation=='version':altered['format']='toklet-q4-model-v1'
                if mutation=='quantizer':altered['quantizer']='toklet-q4-affine-f32-source-half-metadata-rne-v1'
                if mutation=='exclusions':altered['excluded_dtype']='F32'
                if mutation=='source':altered['sources'][0]['sha256']='0'*64
                if mutation=='duplicate':altered['matrices'].append(dict(altered['matrices'][0]))
                if mutation=='missing':altered['matrices']=[]
                if mutation=='remap':altered['matrices'][0]['file']='../other'
                if mutation=='shape':altered['matrices'][0]['columns']=8
                (packed/'manifest.json').write_text(json.dumps(altered))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):storage.Weights(path,storage.sha(path),packed)
            (packed/'manifest.json').write_text(json.dumps(saved))
            with storage.Weights(path,storage.sha(path),packed) as reader:
                np.testing.assert_array_equal(reader['layers.0.q_proj.weight'],np.tile(words,(2,1)))
                reader.verify();config.write_bytes(b'{"changed":true}')
                with self.assertRaisesRegex(ValueError,'config endpoint'):reader.verify()
            config.write_bytes(b'{}');mixed=bytearray(data);mixed[:8]=b'TOKQ4V1\0';mixed[8:12]=(1).to_bytes(4,'little');mixed[12:16]=(64).to_bytes(4,'little')
            (packed/(name+'.q4')).write_bytes(mixed);saved['matrices'][0]['packed_sha256']=hashlib.sha256(mixed).hexdigest();(packed/'manifest.json').write_text(json.dumps(saved))
            with storage.Weights(path,storage.sha(path),packed) as reader,self.assertRaises(ValueError):reader['layers.0.q_proj.weight']


if __name__=='__main__':unittest.main()

"""Authored hostile/archive and static-export fixtures; no downloaded code runs."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import mlx_architectures as m

REV = 'a' * 40


def git_metadata(raw):
    # Independent fixture writer: Git tree bytes are recursively serialized from
    # authored tar files; the parser consumes the resulting flat API descriptor.
    hierarchy={}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            parts=member.name.split('/')[1:];node=hierarchy
            for part in parts[:-1]:node=node.setdefault(part,{})
            data=archive.extractfile(member).read() if member.isfile() else b''
            node[parts[-1]]=(data,'100644' if member.isfile() else '120000')
    rows=[]
    def tree(node,prefix=''):
        entries=[]
        for name,value in node.items():
            path=prefix+name
            if isinstance(value,dict):
                pin=tree(value,path+'/');mode='040000';kind='tree'
            else:
                raw,mode=value;pin=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();kind='blob'
            row=dict(path=path,mode=mode,type=kind,sha=pin)
            if kind=='blob':row['size']=len(raw)
            rows.append(row);entries.append((name,value,pin,mode))
        data=b''
        for name,value,pin,mode in sorted(entries,key=lambda item:(item[0]+('/' if isinstance(item[1],dict) else '')).encode()):
            data+=mode.lstrip('0').encode()+b' '+name.encode()+b'\0'+bytes.fromhex(pin)
        return hashlib.sha1(b'tree '+str(len(data)).encode()+b'\0'+data).hexdigest()
    root=tree(hierarchy)
    return dict(sha=root,truncated=False,tree=rows),dict(sha=REV,commit=dict(tree=dict(sha=root)))


class Inventory(unittest.TestCase):
    def fixture(self, mutation=None):
        output = io.BytesIO()
        prefix = 'mlx-lm-' + REV
        files = [('mlx_lm/utils.py', b'MODEL_REMAPPING={"mistral":"llama"}\nMODEL_ARCHITECTURE_REMAPPING={("a","A"):"b"}\n'),
                 ('mlx_lm/models/llama.py', b'class Model: pass\nclass ModelArgs: pass\n'),
                 ('mlx_lm/models/qwen2/__init__.py', b'from .qwen2 import Model\nfrom .config import ModelConfig\n'),
                 ('mlx_lm/models/qwen2/qwen2.py', b'class Model: pass\n'),
                 ('mlx_lm/models/base.py', b'class BaseModel: pass\n'), ('LICENSE', b'authored fixture')]
        if mutation == 'omitOne': files = [row for row in files if row[0] != 'mlx_lm/models/qwen2/qwen2.py']
        if mutation == 'missing': files = [('LICENSE', b'authored fixture')]
        if mutation == 'execution': files[1] = (files[1][0], b'raise RuntimeError("must never execute")\nclass Model: pass\n')
        if mutation == 'aliases': files[0] = (files[0][0], b'MODEL_REMAPPING={1:"llama"}\n')
        if mutation == 'syntax': files[1] = (files[1][0], b'class ???')
        if mutation == 'traversal': files.append(('../outside.py', b'authored'))
        if mutation == 'duplicate': files.append(files[1])
        if mutation == 'wrongRoot': prefix = 'other-' + REV
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            for name, raw in files:
                member = tarfile.TarInfo(prefix + '/' + name);member.size = len(raw)
                if mutation == 'symlink' and name.endswith('llama.py'):
                    member.type = tarfile.SYMTYPE;member.linkname = '/outside';member.size = 0
                archive.addfile(member, io.BytesIO(raw))
        return output.getvalue()

    def invoke(self, raw, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.tar.gz';path.write_bytes(raw)
            tree,commit=kwargs['metadata'] if 'metadata' in kwargs else git_metadata(raw)
            return m.inventory(path, kwargs.get('expected', hashlib.sha256(raw).hexdigest()), 'mlx-lm', kwargs.get('revision', REV),tree,commit)

    def test_exact_tree_exports_aliases_and_nested_helpers(self):
        result = self.invoke(self.fixture())
        self.assertEqual(set(result['root_entry_points']), {'llama', 'qwen2'})
        self.assertFalse(result['model_modules']['mlx_lm/models/qwen2/qwen2.py']['root_entry_point'])
        self.assertEqual(result['loader_aliases']['MODEL_ARCHITECTURE_REMAPPING'], [dict(source=['a', 'A'], target='b')])
        json.dumps(result, allow_nan=False)
    def test_untrusted_module_body_is_parsed_and_never_executed(self):
        result = self.invoke(self.fixture('execution'))
        self.assertIn('llama', result['root_entry_points'])
    def test_path_alias_syntax_special_member_and_completeness_failures_reject(self):
        for mutation in ['traversal', 'duplicate', 'wrongRoot', 'symlink', 'missing', 'aliases', 'syntax']:
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, SyntaxError)):
                self.invoke(self.fixture(mutation))
    def test_hash_and_immutable_revision_are_required(self):
        for kwargs in [dict(expected='f' * 64), dict(expected='not-a-hash'), dict(revision='main')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.invoke(self.fixture(), **kwargs)
    def test_archive_expansion_source_and_member_count_bounds(self):
        for name in ['MAX_ARCHIVE', 'MAX_EXPANDED', 'MAX_SOURCE', 'MAX_MEMBERS']:
            with self.subTest(name=name), patch.object(m, name, 1), self.assertRaises(ValueError): self.invoke(self.fixture())
    def test_missing_one_model_or_changed_git_descriptor_rejects(self):
        for mutation in ['omitOne','truncatedTree','blob','root','mode']:
            raw=self.fixture('omitOne') if mutation=='omitOne' else self.fixture();tree,commit=git_metadata(self.fixture())
            if mutation=='truncatedTree':tree['truncated']=True
            elif mutation=='blob':tree['tree'][0]['sha']='f'*40
            elif mutation=='root':commit['commit']['tree']['sha']='f'*40
            elif mutation=='mode':tree['tree'][0]['mode']='100755'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.invoke(raw,metadata=(tree,commit))
    def test_oversized_compressed_pax_header_is_bounded_before_tar_parser(self):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz',format=tarfile.PAX_FORMAT) as archive:
            member=tarfile.TarInfo('mlx-lm-'+REV+'/LICENSE');member.pax_headers={'comment':'x'*32768};member.size=1
            archive.addfile(member,io.BytesIO(b'x'))
        raw=stream.getvalue();tree,commit=git_metadata(self.fixture())
        with patch.object(m,'MAX_EXPANDED',2048),patch.object(m.tarfile,'open') as parse,self.assertRaisesRegex(ValueError,'Decompressed tar'):
            self.invoke(raw,metadata=(tree,commit))
        parse.assert_not_called()
    def test_known_empty_git_tree_identity(self):
        root='4b825dc642cb6eb9a060e54bf8d69288fbee4904'
        selected,actual=m.tree_selection(dict(sha=root,truncated=False,tree=[]),dict(sha=REV,commit=dict(tree=dict(sha=root))),'mlx-lm',REV)
        self.assertEqual(selected,{});self.assertEqual(actual,root)
    def test_truncated_archive_cannot_produce_inventory(self):
        with self.assertRaises((EOFError, tarfile.TarError)): self.invoke(self.fixture()[:60])
    def test_path_replacement_cannot_change_consumed_archive(self):
        raw = self.fixture();original = tarfile.open;tree,commit=git_metadata(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.tar.gz';path.write_bytes(raw)
            def opening(*args, **kwargs):
                path.write_bytes(b'path replaced after admitted read')
                return original(*args, **kwargs)
            with patch.object(m.tarfile, 'open', side_effect=opening):
                result = m.inventory(path, hashlib.sha256(raw).hexdigest(), 'mlx-lm', REV,tree,commit)
            self.assertEqual(result['archive']['sha256'], hashlib.sha256(raw).hexdigest())


if __name__ == '__main__': unittest.main()

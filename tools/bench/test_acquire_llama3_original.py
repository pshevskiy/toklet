import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import acquire_llama3_original as subject


class AcquisitionTests(unittest.TestCase):
    def test_original_and_mirror_require_all_frozen_identities(self):
        rows = [{'rfilename': name, 'blobId': blob, 'size': 10} for name, blob in subject.BLOBS.items()]
        rows.append({'rfilename': 'model.safetensors', 'blobId': subject.POINTER, 'size': subject.WEIGHT_BYTES,
                     'lfs': {'size': subject.WEIGHT_BYTES, 'sha256': subject.WEIGHT_SHA}})
        document = {'sha': subject.REVISION, 'siblings': rows}
        subject.inventory(document, subject.REVISION)
        for index in range(len(rows)):
            changed = copy.deepcopy(document)
            changed['siblings'][index]['blobId'] = '0' * 40
            with self.assertRaises(ValueError):
                subject.inventory(changed, subject.REVISION)
        changed = copy.deepcopy(document); changed['siblings'].append(rows[0])
        with self.assertRaises(ValueError):
            subject.inventory(changed, subject.REVISION)

    def test_exclusive_bounded_stream_rejects_overflow_and_wrong_digest(self):
        class Response(io.BytesIO):
            status = 200
            def geturl(self):
                return 'https://huggingface.co/fixture'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sink'
            with patch.object(subject.urllib.request, 'urlopen', return_value=Response(b'12345')):
                with self.assertRaises(ValueError):
                    subject.transfer('https://huggingface.co/fixture', path, 4)
            self.assertLessEqual(path.stat().st_size, 4)
            with self.assertRaises(FileExistsError):
                subject.transfer('https://huggingface.co/fixture', path, 5)
            path = Path(directory) / 'digest'
            with patch.object(subject.urllib.request, 'urlopen', return_value=Response(b'12345')):
                with self.assertRaises(ValueError):
                    subject.transfer('https://huggingface.co/fixture', path, 5, 5, '0' * 64)

    def test_payload_reserve_is_checked_after_metadata_and_during_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'payload'
            with patch.object(subject.shutil, 'disk_usage', return_value=type('Space', (), {'free': 9})()):
                with self.assertRaises(ValueError):
                    subject.transfer('https://huggingface.co/fixture', path, 5, 5, disk_reserve=5)
            self.assertFalse(path.exists())
            class Response(io.BytesIO):
                status = 200
                def geturl(self):
                    return 'https://huggingface.co/fixture'
            mib = 1024**2
            space = lambda free: type('Space', (), {'free': free})()
            with patch.object(subject.shutil, 'disk_usage', side_effect=[space(15*mib), space(6*mib)]), patch.object(subject.urllib.request, 'urlopen', return_value=Response(bytes(10*mib))):
                with self.assertRaises(ValueError):
                    subject.transfer('https://huggingface.co/fixture', path, 10*mib, 10*mib, disk_reserve=5*mib)
            self.assertEqual(path.stat().st_size, 8*mib)

    def test_git_blob_is_independent_known_git_object(self):
        self.assertEqual(subject.git_blob(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')

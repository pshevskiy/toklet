"""Portable structural/clock/provenance controls for discarded-payload witnesses."""
import copy
import hashlib
import io
import unittest

import minicpm5_parent as source
import check_minicpm5_parent as reader
import test_minicpm5_parent as fixtures


def case():
    prefix, header, raw = fixtures.fixture()
    shard = ('fixture', len(raw), hashlib.sha256(raw).hexdigest(), prefix, header,
             {'norm': bytes.fromhex('803f00c0')})
    result = source.authenticate(io.BytesIO(raw), *shard[3:5], shard[1], shard[2], shard[5], io.StringIO())
    return shard, result


class Witnesses(unittest.TestCase):
    def test_literal_complete_tensor_and_block_witness(self):
        shard, result = case()
        reader.validate_result(result, shard)
        self.assertEqual(result['tensors'][0]['sha256'],
                         hashlib.sha256(bytes.fromhex('803f00c0')).hexdigest())

    def test_whole_identity_coverage_and_exact_norm_digest_cannot_be_relabelled(self):
        shard, original = case()
        changes = [(['bytes'], True), (['sha256'], '0' * 64), (['exact_child_norms'], False),
                   (['tensors', 0, 'sha256'], '0' * 64), (['extra_claim'], True),
                   (['tensors', 0, 'exact_child_norm'], 1), (['tensors', 0, 'shape'], [2.])]
        for path, value in changes:
            changed = copy.deepcopy(original);node = changed
            for key in path[:-1]:node = node[key]
            node[path[-1]] = value
            with self.assertRaises(ValueError):reader.validate_result(changed, shard)
        changed = copy.deepcopy(original);changed['tensors'].pop()
        with self.assertRaises(ValueError):reader.validate_result(changed, shard)

    def test_block_holes_overlaps_boolean_offsets_and_bad_single_block_hash_reject(self):
        shard, original = case()
        for field, value in [('offset', 0), ('offset', True), ('bytes', 3),
                             ('bytes', 4.), ('sha256', 'z' * 64), ('sha256', '0' * 64)]:
            changed = copy.deepcopy(original);changed['tensors'][0]['blocks'][0][field] = value
            with self.assertRaises(ValueError):reader.validate_result(changed, shard)
        changed = copy.deepcopy(original);changed['tensors'][0]['blocks'] *= 2
        with self.assertRaises(ValueError):reader.validate_result(changed, shard)

    def test_sequential_closed_transport_with_exact_command_and_hash(self):
        shard, _ = case();record, error = fixtures.Transfers().run_transfer()
        self.assertIsNone(error)
        owner = dict(started_ns=record['started_ns'] - 1, completed_ns=record['completed_ns'] + 1)
        reader.validate_transfer(record, shard, owner)

    def test_transport_clock_and_resource_can_never_exceed_owning_lifetime(self):
        shard, _ = case();original, _ = fixtures.Transfers().run_transfer()
        owner = dict(started_ns=original['started_ns'] - 1, completed_ns=original['completed_ns'] + 1)
        for field, value in [('started_ns', True), ('started_ns', owner['started_ns'] - 1),
                             ('completed_ns', owner['completed_ns'] + 1),
                             ('final_wait4_peak_rss_bytes', 256 * 1024 ** 2 + 1)]:
            changed = copy.deepcopy(original);changed[field] = value
            with self.assertRaises(ValueError):reader.validate_transfer(changed, shard, owner)

    def test_partial_insecure_unclosed_or_payload_retaining_transport_rejects(self):
        shard, _ = case();original, _ = fixtures.Transfers().run_transfer()
        owner = dict(started_ns=original['started_ns'] - 1, completed_ns=original['completed_ns'] + 1)
        for field, value in [('passed', 1), ('exit_code', False), ('exit_code', 22),
                             ('http_code', 206), ('ssl_verify_result', False),
                             ('ssl_verify_result', 1), ('payload_retained', True),
                             ('size_download', shard[1] - 1)]:
            changed = copy.deepcopy(original);changed[field] = value
            with self.assertRaises(ValueError):reader.validate_transfer(changed, shard, owner)

    def test_unbounded_or_changed_download_command_is_not_inferred(self):
        shard, _ = case();original, _ = fixtures.Transfers().run_transfer()
        owner = dict(started_ns=original['started_ns'] - 1, completed_ns=original['completed_ns'] + 1)
        for index, value in [(0, 'curl'), (1, '--insecure'), (5, '=http'), (13, '99999999999'),
                             (20, 'saved-parent.safetensors'), (-1, 'https://huggingface.co/main')]:
            changed = copy.deepcopy(original);changed['command'][index] = value
            with self.assertRaises(ValueError):reader.validate_transfer(changed, shard, owner)


if __name__ == '__main__':unittest.main()

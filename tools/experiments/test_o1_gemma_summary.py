import unittest
from pathlib import Path
import re
from o1_gemma_summary import comparison


class OverheadAnalysisTests(unittest.TestCase):
    def rows(self):
        return [dict(trial=i, warmup=i < 3, mode=mode, ttft_ms=latency,
                     total_ms=10 * latency, tokens=[1, 2], history=4, committed=3, pending=1)
                for i in range(13) for mode, latency in [('disabled', 100), ('summary', 102)]]

    def test_known_two_percent_cost_is_not_hidden_by_units(self):
        result = comparison(self.rows())
        for value in result.values():
            self.assertAlmostEqual(value['median_ratio_overhead_pct'], 2)
            self.assertAlmostEqual(value['paired_median_pct'], 2)
            self.assertGreater(value['paired_median_bootstrap_95_pct'][0], 1)

    def test_mismatched_work_is_rejected(self):
        rows = self.rows()
        rows[-1]['pending'] = 0
        with self.assertRaises(ValueError):
            comparison(rows)

    def test_xcode_summary_source_membership(self):
        project = (Path(__file__).resolve().parents[2] / 'apple/TokletHarness.xcodeproj/project.pbxproj').read_text()
        file_id = re.search(r'(\w{24}) = \{ isa = PBXFileReference;[^\n]*path = "Harness/NativeRequestSummary.swift"', project).group(1)
        build_id = re.search(r'(\w{24}) = \{ isa = PBXBuildFile; fileRef = ' + file_id, project).group(1)
        self.assertRegex(project, r'PBXSourcesBuildPhase;[^\n]*files = \([^)]*' + build_id)
        self.assertRegex(project, r'PBXGroup; children = \([^)]*' + file_id)


if __name__ == '__main__':
    unittest.main()

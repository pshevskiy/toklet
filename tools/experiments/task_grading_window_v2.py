"""One frozen larger v2 grading window; start only after owned model exit."""
import argparse
from pathlib import Path
import sys
import task_grading_window as shared
import task_grade_admission_v2 as admission
import metal_tasks_v2 as producer


def run(args):
    return shared.run(args,collection_admission=admission,idle_admission=admission.admit_idle,
                      extra_source_paths=(Path(__file__).resolve(),Path(admission.__file__).resolve(),Path(producer.__file__).resolve()),
                      extra_grading_modules=(admission,producer,producer.cli,producer.cli.shared,producer.loss,
                                             producer.loss.shared,producer.loss.large,producer.loss.large.quality_rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--idle-proof',type=Path,required=True)
    parser.add_argument('--prefix',required=True)
    sys.exit(run(parser.parse_args()))

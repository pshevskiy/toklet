"""Original literal grading window for one completed Online collection."""
import argparse
from pathlib import Path
import sys

import m6_task_admission as admission
import task_grading_window as shared


def run(args):
    producer = admission.producer
    return shared.run(args, collection_admission=admission, idle_admission=admission.admit_idle,
                      extra_source_paths=(Path(__file__).resolve(), Path(admission.__file__).resolve(), Path(producer.__file__).resolve()),
                      extra_grading_modules=(admission, producer, producer.context, producer.shared,
                                             producer.shared.cli, producer.cli_v2, producer.context.loss,
                                             producer.context.loss.shared, producer.context.loss.large,
                                             producer.context.loss.large.quality_rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--idle-proof', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    sys.exit(run(p.parse_args()))

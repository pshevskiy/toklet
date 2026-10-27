"""Versioned, archived-logit Qwen3 output-fidelity screen; no model execution.

The contract and exact runtime/source identities must be frozen externally before
this entry point computes anything. This does not replace historical stage gates.
"""
from contextlib import ExitStack
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys

import check_qwen3_context512_native as native
from qwen3_context_precision import Owner

FORMAT = "toklet-qwen3-archived-output-screen-v1"
BOUND = 0.05


def observe(actual, expected):
    """All saved FP32 entries, widened before max/exp/fsum/log/subtraction.

    NumPy FP64 exp is elementwise; math.fsum accumulates its complete result.
    No softmax probabilities are rounded to Half/BF16, clipped or discarded.
    """
    import numpy as np
    if (actual.dtype != np.dtype("<f4") or expected.dtype != np.dtype("<f4")
            or actual.ndim != 1 or expected.shape != actual.shape
            or not 2 <= actual.size <= 151936
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("Matched finite saved FP32 rows required")
    profiles, logs = [], []
    for row in (actual, expected):
        wide = row.astype(np.float64)
        greedy = int(np.argmax(wide))  # First/lowest ID, including exact ties.
        shifted = wide - wide[greedy]
        log_normalizer = math.log(math.fsum(np.exp(shifted)))
        logp = shifted - log_normalizer
        if not np.isfinite(logp).all():
            raise ValueError("Nonfinite normalized log probabilities")
        profiles.append(dict(greedy=greedy, log_normalizer=log_normalizer))
        logs.append(logp)
    error = np.abs(logs[0] - logs[1])
    index = int(np.argmax(error))
    maximum = float(error[index])
    failures = int(np.count_nonzero(error > BOUND))
    return dict(maximum_absolute_log_probability_error=maximum,
                maximum_vocabulary_id=index,
                native_log_probability=float(logs[0][index]),
                reference_log_probability=float(logs[1][index]),
                failed_vocabulary_entries=failures,
                native_greedy=profiles[0]["greedy"],
                reference_greedy=profiles[1]["greedy"],
                greedy_equal=profiles[0]["greedy"] == profiles[1]["greedy"],
                native_log_normalizer=profiles[0]["log_normalizer"],
                reference_log_normalizer=profiles[1]["log_normalizer"],
                passed=failures == 0 and profiles[0]["greedy"] == profiles[1]["greedy"])


def document(path, pin, stack, audits):
    owner = Owner(path, pin)
    audits.append(owner.audit)
    stack.enter_context(owner)
    if pin["bytes"] > 8 * 1024**2:
        raise ValueError("Screen metadata exceeds bound")
    owner.file.seek(0)
    return native.decode(owner.file.read(pin["bytes"]))


def runtime_endpoint(path, pin, identity, audits):
    if Owner.identity(path.lstat()) != identity:
        raise ValueError("Screen runtime/source path endpoint differs")
    runtime_file(path, pin, audits, identity)


def runtime_file(path, pin, audits, expected=None):
    # The array Owner deliberately rejects empty captures. Installed NumPy also
    # contains legitimate empty package/type markers; authenticate those exactly.
    if pin["bytes"] == 0:
        audit = dict(path=str(path), checksum_passes=0, endpoint_checked=False, endpoint_error=None)
        audits.append(audit)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb", buffering=0) as stream:
                value = os.fstat(stream.fileno());identity = Owner.identity(value)
                if (type(pin["bytes"]) is not int or not stat.S_ISREG(value.st_mode) or value.st_size != 0
                        or pin["sha256"] != hashlib.sha256(b"").hexdigest()
                        or expected is not None and identity != expected or stream.read(1)
                        or Owner.identity(path.lstat()) != identity
                        or Owner.identity(os.fstat(stream.fileno())) != identity):
                    raise ValueError("Empty runtime marker identity differs")
                audit.update(checksum_passes=1, endpoint_checked=True)
                return identity
        except BaseException as error:
            audit["endpoint_error"] = type(error).__name__ + ": " + str(error)
            raise
    owner = Owner(path, pin)
    audits.append(owner.audit)
    with owner:
        if expected is not None and owner.before != expected:
            raise ValueError("Screen runtime/source descriptor endpoint differs")
        return owner.before


def runtime_origins(root, contract):
    """Bind the executing numerical imports and non-system native images.

    Uses the same public dyld inventory as text_reference_runtime, without
    importing that text producer or its unrelated third-party packages.
    """
    import ctypes
    admitted = {str((root / name).resolve()) for name in contract["screen_runtime_and_sources"]}
    selected = contract["required_import_origins"]
    origins = {}
    for name, module in list(sys.modules.items()):
        if name in selected or name == "numpy" or name.startswith("numpy."):
            origin = getattr(module, "__file__", None)
            if origin is None or str(Path(origin).resolve()) not in admitted:
                raise ValueError("Unadmitted numerical module origin: " + name)
            origins[name] = str(Path(origin).resolve())
    if any(origins.get(name) != str((root / path).resolve()) for name, path in selected.items()):
        raise ValueError("Actual numerical/helper import differs")
    if (str(Path(__file__).resolve()) not in admitted
            or str(Path(sys.executable).resolve()) != contract["python_executable"]
            or not sys.flags.no_user_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute()):
        raise ValueError("Screen source/interpreter/cold execution differs")
    library = ctypes.CDLL(None)
    count = library._dyld_image_count;count.argtypes = [];count.restype = ctypes.c_uint32
    image = library._dyld_get_image_name;image.argtypes = [ctypes.c_uint32];image.restype = ctypes.c_char_p
    images = []
    for index in range(count()):
        path = str(Path(image(index).decode()).resolve())
        if path.startswith(("/usr/lib/", "/System/")):
            continue
        if path not in admitted:
            raise ValueError("Unadmitted native runtime image: " + path)
        images.append(path)
    if contract["python_shared_runtime"] not in images:
        raise ValueError("Pinned shared Python runtime is not loaded")
    return dict(modules=origins, non_system_native_images=sorted(images))


def validate_contract(value):
    if (value["format"] != FORMAT or value["positions"] != 512
            or type(value["positions"]) is not int or value["vocabulary"] != 151936
            or type(value["vocabulary"]) is not int
            or value["criterion"] != dict(maximum_absolute_log_probability_error=BOUND,
                inclusive=True, exact_lowest_id_greedy=True, nonfinite_rejected=True)
            or any(value["criterion"][key] is not True for key in
                   ("inclusive", "exact_lowest_id_greedy", "nonfinite_rejected"))
            or value["native_recipe"] != dict(name="chunk128", chunk=128, segments=[512])
            or type(value["native_recipe"]["chunk"]) is not int
            or any(type(x) is not int for x in value["native_recipe"]["segments"])
            or [case["id"] for case in value["cases"]] != ["original", "future_tail"]):
        raise ValueError("Frozen screen geometry/criterion differs")
    for case in value["cases"]:
        wanted = [9707, 11, 1879, 0] * 128
        if case["id"] == "future_tail":
            wanted[-1] = 17
        if case["tokens"] != wanted or any(type(x) is not int for x in case["tokens"]):
            raise ValueError("Frozen screen token positions differ")
    return value


def screen(root, contract_path, contract_pin, journal, report):
    import numpy as np
    root = Path(root)
    report.update(complete=False, accepted=False, parents=[], cases=[],
                  journal=dict(bytes=0, rows=0), validation_opened=False,
                  native_executed=False, historical_canonical_qualified=False,
                  current_engine_or_app_qualified=False)
    digest = hashlib.sha256()
    with ExitStack() as stack:
        contract = validate_contract(document(contract_path, contract_pin, stack, report["parents"]))
        # Check the complete pinned runtime/source closure twice without keeping
        # hundreds of package descriptors open. Reject replaced paths/inodes too.
        # This screen imports no MLX and opens no validation inputs.
        for path, pin in contract["screen_runtime_and_sources"].items():
            identity = runtime_file(root / path, pin, report["parents"])
            stack.callback(runtime_endpoint, root / path, pin, identity, report["parents"])
        report["initial_origins"] = runtime_origins(root, contract)
        def origins_endpoint():
            report["endpoint_origins"] = runtime_origins(root, contract)
        stack.callback(origins_endpoint)
        fixture = document(root / contract["reference_manifest"]["path"],
                           contract["reference_manifest"]["identity"], stack, report["parents"])
        if fixture["plan"]["cases"] != {case["id"]: case["tokens"] for case in contract["cases"]}:
            raise ValueError("Archived reference token positions differ")
        with Path(journal).open("xb") as output:
            try:
                for case in contract["cases"]:
                    summary = dict(case=case["id"], rows=0, complete=False,
                                   failed_rows=0, failed_vocabulary_entries=0,
                                   greedy_disagreements=0, worst=None)
                    report["cases"].append(summary)
                    with ExitStack() as case_stack:
                        # The frame journal is JSONL, not a JSON document.
                        journal_owner = Owner(root / case["native_journal"]["path"], case["native_journal"]["identity"])
                        report["parents"].append(journal_owner.audit)
                        case_stack.enter_context(journal_owner)
                        frames = list(native.frames(root / case["native_journal"]["path"],
                            contract["native_recipe"], fixture))
                        frames = [frame for frame in frames if frame["name"] == "logits"]
                        if frames != case["native_logit_frames"] or len(frames) != 4:
                            raise ValueError("Frozen native logit frames differ")
                        owners = {}
                        for name in ("native", "reference"):
                            item = case[name]
                            owner = Owner(root / item["path"], item["identity"])
                            report["parents"].append(owner.audit)
                            owners[name] = case_stack.enter_context(owner)
                        reference = fixture["records"][case["id"] + ".fp64"]
                        item = reference["arrays"]["logits"]
                        if (case["reference"]["identity"] != {k: reference[k] for k in ("bytes", "sha256")}
                                or item != case["reference_logit_array"]
                                or item["shape"] != [512, 151936] or item["dtype"] != "<f4"):
                            raise ValueError("Frozen FP64/Half reference differs")
                        owners["reference"].file.seek(item["offset"])
                        reference_digest = hashlib.sha256()
                        for frame in frames:
                            owners["native"].file.seek(frame["offset"])
                            frame_digest = hashlib.sha256()
                            for position in range(frame["start"], frame["start"] + frame["rows"]):
                                if position != summary["rows"]:
                                    raise ValueError("Noncontiguous native row coverage")
                                rows = {}
                                for name in owners:
                                    raw = owners[name].file.read(151936 * 4)
                                    if len(raw) != 151936 * 4:
                                        raise ValueError("Truncated saved logit row")
                                    (frame_digest if name == "native" else reference_digest).update(raw)
                                    rows[name] = np.frombuffer(raw, dtype="<f4")
                                value = dict(case=case["id"], position=position,
                                             **observe(rows["native"], rows["reference"]))
                                raw = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
                                written = output.write(raw)
                                if type(written) is int and 0 <= written <= len(raw):
                                    digest.update(raw[:written]);report["journal"]["bytes"] += written
                                if written != len(raw):
                                    raise OSError("Incomplete screen row journal write")
                                report["journal"]["rows"] += 1;summary["rows"] += 1
                                summary["failed_rows"] += int(not value["passed"])
                                summary["failed_vocabulary_entries"] += value["failed_vocabulary_entries"]
                                summary["greedy_disagreements"] += int(not value["greedy_equal"])
                                if (summary["worst"] is None or value["maximum_absolute_log_probability_error"]
                                        > summary["worst"]["maximum_absolute_log_probability_error"]):
                                    summary["worst"] = value
                            if frame_digest.hexdigest() != frame["sha256"]:
                                raise ValueError("Consumed native logit frame differs")
                        if summary["rows"] != 512 or reference_digest.hexdigest() != item["sha256"]:
                            raise ValueError("Consumed reference logit array/coverage differs")
                    summary["complete"] = True
                    print(json.dumps({k: v for k, v in summary.items() if k != "worst"}), flush=True)
            finally:
                report["journal"]["sha256"] = digest.hexdigest()
    report["complete"] = True
    report["accepted"] = all(case["failed_rows"] == 0 for case in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--contract-bytes", type=int, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(passed=False, primary_error=None, report={})
    def interrupted(number, _):
        # Let ExitStack authenticate endpoints and main retain the partial result.
        # Further cancellation must not interrupt this short cleanup.
        for item in (signal.SIGTERM, signal.SIGINT):
            signal.signal(item, signal.SIG_IGN)
        raise InterruptedError("Screen interrupted by signal " + str(number))
    handlers = {item: signal.signal(item, interrupted) for item in (signal.SIGTERM, signal.SIGINT)}
    try:
        screen(Path.cwd(), args.contract,
               dict(bytes=args.contract_bytes, sha256=args.contract_sha), args.journal, result["report"])
        result["passed"] = True  # A completed rejection is a valid screen outcome.
    except BaseException as error:
        result["primary_error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        try:
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False);stream.write("\n")
        finally:
            for item, handler in handlers.items():
                signal.signal(item, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())

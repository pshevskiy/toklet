"""Explicit local Metal/state probes against the frozen independent Qwen logits.

Records a failed numerical gate without changing it. External throughput records
are not substitutes for Toklet's numerical or held-out quality qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np
import llama_prepare as setup
import prepare
import mlx_baseline as common


def reference_logits(full, item, anchor):
    expected_file = full / item["file"]
    if (item["sha256"] != anchor["full_array_sha256"]
            or prepare.digest(expected_file) != item["sha256"]
            or item["shape"] != anchor["shape"]
            or item["bytes"] != anchor["full_array_bytes"]
            or expected_file.stat().st_size != anchor["full_array_bytes"]
            or item["reference_envelope"] != anchor["reference_envelope"]):
        raise ValueError("Independent full logits identity differs")
    return np.fromfile(expected_file, dtype="<f4").reshape(anchor["shape"])[-1].astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite probe evidence")
    revision = setup.source_identity()
    build = setup.checked_build()
    converted = prepare.load_json(setup.MODEL / "manifest.json")
    binary = prepare.ROOT / build["binary"]
    model = setup.MODEL / "model.gguf"
    if (build["llama_revision"] != revision or converted["llama_revision"] != revision
            or prepare.digest(binary) != build["binary_sha256"]
            or prepare.digest(model) != converted["gguf_sha256"]):
        raise ValueError("External artifact/build identity changed")
    full = prepare.ROOT / "artifacts/qwen-forward-v2"
    fixture = prepare.load_json(full / "manifest.json")
    anchors = prepare.load_json(prepare.ROOT / "tests/fixtures/qwen-v2/manifest.json")
    for key in ("model", "cases", "policy", "exporter_sha256", "argmax"):
        if fixture[key] != anchors[key]:
            raise ValueError(f"Independent fixture identity differs: {key}")
    source = prepare.load_json(prepare.SOURCES)["model"]["weights"]
    prepare.validate_source(source)
    if fixture["model"]["weights"] != source or converted["source_sha256"] != source["sha256"]:
        raise ValueError("Probe and converted source differ")
    with tempfile.TemporaryDirectory(prefix="toklet-llama-probe-") as directory, tempfile.TemporaryFile(mode="w+t") as errors:
        worker = subprocess.Popen([str(binary), str(model)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=errors, text=True)
        watchdog = threading.Timer(300, worker.kill)
        watchdog.start()
        try:
            def read():
                line = worker.stdout.readline()
                if not line:
                    errors.seek(0)
                    raise ValueError(errors.read())
                return json.loads(line)

            def send(request, event):
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
                result = read()
                if result.get("event") != event:
                    raise ValueError(result)
                return result

            ready = read()
            if ready.get("event") != "ready":
                raise ValueError(ready)
            probes = []
            for name, case in fixture["cases"].items():
                item = fixture["arrays"][f"cases.{name}.logits"]
                anchor = anchors["arrays"][f"cases.{name}.logits"]
                expected = reference_logits(full, item, anchor)
                output = Path(directory) / f"{name}.f32le"
                result = send({"action": "probe", "tokens": case["tokens"],
                               "position_offset": case["offset"], "path": str(output)}, "prepared")
                if result["evaluated_tokens"] != 0 or result["pending_tokens"] != 0:
                    raise ValueError("Offset probe did not reset session")
                actual = np.fromfile(output, dtype="<f4").astype(np.float64)
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError("Invalid completed probe logits")
                difference = np.abs(actual - expected)
                gate = anchor["reference_envelope"]["atol"] + anchors["policy"]["rtol"] * np.abs(expected)
                target = int(fixture["argmax"][name][-1])
                def nll(logits):
                    maximum = logits.max()
                    return float(maximum + np.log(np.exp(logits - maximum).sum()) - logits[target])
                probes.append({"case": name, "max_absolute_error": float(difference.max()),
                               "max_tolerance_fraction": float((difference / gate).max()),
                               "gate_passed": bool((difference <= gate).all()),
                               "expected_greedy": target, "actual_greedy": int(actual.argmax()),
                               "teacher_forced_nll_delta": nll(actual) - nll(expected)})
            state = send({"action": "prepare", "tokens": [9707, 11, 358, 2776]}, "prepared")
            if (state["evaluated_tokens"], state["pending_tokens"]) != (4, 0):
                raise ValueError("Preparation state differs")
            send({"action": "stage", "tokens": [4460]}, "prepared")
            for request in ({"action": "stage", "tokens": [-1]},
                            {"action": "stage", "tokens": [2**64-1]},
                            {"action": "reset", "tokens": [1]*1921, "continuation": [1]*128},
                            {"action": "stage", "tokens": [1], "unknown": True}):
                send(request, "error")
            state = send({"action": "continue", "continuation": [1, 2]}, "sample")["sample"]
            if (state["reused_tokens"], state["prefill_tokens"], state["evaluated_tokens"], state["pending_tokens"]) != (4, 1, 6, 1):
                raise ValueError("Rejection or continuation changed the valid prefix")
            state = send({"action": "prompt", "tokens": [9707, 11, 358, 2776, 4460, 1, 2], "continuation": [3]}, "sample")["sample"]
            if (state["reused_tokens"], state["prefill_tokens"]) != (6, 1):
                raise ValueError("Exact pending prompt did not reuse prefix")
            state = send({"action": "prompt", "tokens": [9707], "continuation": [4]}, "sample")["sample"]
            if state["reused_tokens"] != 0 or state["prefill_tokens"] != 1:
                raise ValueError("Shortened prompt did not replay")
            tokens = prepare.read_tokens(prepare.CONTRACT / "prompt_1920.u32le")
            fixed = prepare.read_tokens(prepare.CONTRACT / "continuation_128.u32le")
            state = send({"action": "prepare", "tokens": tokens + fixed}, "prepared")
            if (state["evaluated_tokens"], state["pending_tokens"]) != (2048, 0):
                raise ValueError("Real-model final capacity position did not commit")
            send({"action": "stage", "tokens": [1]}, "error")
            peak = common.footprint_peak(worker.pid)
            if peak is None or peak > 6*1024**3:
                raise ValueError("Missing or excessive physical footprint")
            worker.stdin.close()
            if worker.wait(timeout=20) != 0:
                raise ValueError("Worker shutdown failed")
            errors.seek(0)
            diagnostics = errors.read()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            prepare.write_json(args.output, {"llama_revision": revision, "binary_sha256": build["binary_sha256"],
                "worker_sha256": build["worker_sha256"], "checker_sha256": prepare.digest(Path(__file__)),
                "converted_model": converted, "ready": ready, "numerical_probes": probes,
                "all_numerical_gates_passed": all(p["gate_passed"] for p in probes),
                "state_checks_passed": True, "completed_capacity": 2048,
                "peak_process_bytes": peak, "device": common.command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
                "vendor_diagnostics": diagnostics,
                "scope": "Last full-vocabulary logits for four short frozen cases on the small-batch path; functional cache/admission checks and real 2048-position execution. No 128-token GEMM numerical, held-out quality or throughput qualification. phys_footprint_peak is a kernel charge counter that omits clean mapped weights; total mapped/shared physical peak is unavailable."})
            print(json.dumps(probes, indent=2))
        finally:
            watchdog.cancel()
            if worker.poll() is None:
                worker.kill()
            try:
                worker.wait(timeout=20)
            finally:
                worker.stdin.close()
                worker.stdout.close()


if __name__ == "__main__":
    main()

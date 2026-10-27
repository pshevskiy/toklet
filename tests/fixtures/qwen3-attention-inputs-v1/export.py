"""Extract bounded own-input accuracy witnesses from authenticated saved captures.

No model or reference graph executes. Inputs are native layer0 values, not an
independent whole-model oracle. Run from the repository root; existing outputs
are never overwritten. The Rust test computes its own FP64 attention reference.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/reference"))
from qwen3_context_precision import Owner, MANIFEST_SHA
import check_qwen3_context512_native as native

HERE = Path(__file__).resolve().parent

def load(path, sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Frozen metadata differs")
    return native.decode(raw)


def export():
    if any((HERE / name).exists() for name in ("inputs.f16le", "manifest.json")):
        raise FileExistsError("Never overwrite the saved attention witness")
    manifest = load(ROOT / "tests/fixtures/qwen3-context512-v1/manifest.json", MANIFEST_SHA)
    contract = load(ROOT / "tests/fixtures/qwen3-output-screen-v1/contract.json",
                    "0df7e712e7d397d8f334a194a187c0c26938656ab86867fe3a1a3ce624e21721")
    case = contract["cases"][0]
    records, payload, audits = [], bytearray(), []
    with ExitStack() as stack:
        owners = {}
        for name in ("native", "native_journal"):
            item = case[name]
            owner = Owner(ROOT / item["path"], item["identity"])
            owners[name] = stack.enter_context(owner)
            audits.append(owner.audit)
        frames = list(native.frames(ROOT / case["native_journal"]["path"], contract["native_recipe"], manifest))
        for name, width, rows in (("q_rotary", 2048, [0, 41, 65]),
                                  ("k_rotary", 1024, list(range(66))),
                                  ("v_projection", 1024, list(range(66)))):
            frame = next(f for f in frames if f["start"] == 0 and f["name"] == "layers.0." + name)
            if frame["columns"] != width:
                raise ValueError("Witness width differs")
            stream = owners["native"].file
            stream.seek(frame["offset"])
            raw = stream.read(frame["bytes"])
            if hashlib.sha256(raw).hexdigest() != frame["sha256"]:
                raise ValueError("Source frame differs")
            record = dict(name=name, rows=rows, shape=[len(rows), width], offset=len(payload), source_frame=frame)
            selected = bytearray()
            for row in rows:
                stream.seek(frame["offset"] + row * width * 4)
                raw = stream.read(width * 4)
                if len(raw) != width * 4:
                    raise ValueError("Truncated row")
                for (value,) in struct.iter_unpack("<f", raw):
                    half = struct.pack("<e", value)
                    if struct.pack("<f", struct.unpack("<e", half)[0]) != struct.pack("<f", value):
                        raise ValueError("Native input is not exact Half widening")
                    selected.extend(half)
            record.update(bytes=len(selected), sha256=hashlib.sha256(selected).hexdigest())
            payload.extend(selected)
            records.append(record)
    output = dict(format="toklet-qwen3-attention-own-input-v1", layer=0, positions=[0, 41, 65],
                  query_heads=16, kv_heads=8, head_dim=128, reference="Test-local FP64 arithmetic with FP32 score/probability storage and Half context; not a whole-model oracle",
                  source=case["native"], source_journal=case["native_journal"], arrays=records,
                  bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), owners=audits)
    (HERE / "inputs.f16le").write_bytes(payload)
    (HERE / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("bytes", "sha256")}))


if __name__ == "__main__":
    export()

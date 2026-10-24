#!/usr/bin/env python3
"""Check mobile UI controllers/rendering with an explicitly supplied native archive.

The output folder must be fresh. Sources are snapshotted so concurrent engine
and interface work cannot silently change what a successful check describes.
This does not contact an iPhone or claim physical-device qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Mac arm64 libtoklet.a with the current ABI")
    parser.add_argument("output", type=Path, help="fresh output folder")
    parser.add_argument("--model", type=Path, help="optional local supported model for GPU/controller checks")
    parser.add_argument("--fixture", type=Path, help="optional local native tiny fixture for CPU worker/polling checks")
    parser.add_argument("--switch-model", type=Path, help="optional second local model (Qwen) for post-release memory admission checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    archive = args.archive.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((root / "apple/Harness").glob("*.swift"))
    sources += [root / "apple/tests/swift/WorkerChecks.swift", root / "apple/tests/swift/MobileMain.swift"]
    headers = list((root / "apple/include").glob("*"))
    pins = {str(path.relative_to(root)): sha(path) for path in [*sources, *headers] if path.is_file()}
    snapshot = output / "source-snapshot"
    for name, pin in pins.items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise RuntimeError("Source changed during snapshot: " + name)
        copied = snapshot / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(data)
    native_pin = sha(archive)
    records = []
    environment = dict(os.environ)
    developer = Path("/Applications/Xcode.app/Contents/Developer")
    if developer.is_dir():
        environment["DEVELOPER_DIR"] = str(developer)

    def run(label, command):
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True, timeout=300, check=False)
        (output / (label + ".stdout")).write_bytes(result.stdout)
        (output / (label + ".stderr")).write_bytes(result.stderr)
        records.append(dict(label=label, command=list(map(str, command)), exit_code=result.returncode))
        print(label, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-8000:])

    flags = ["-swift-version", "6", "-strict-concurrency=complete", "-warnings-as-errors", "-parse-as-library", "-O",
             "-I", str(snapshot / "apple/include"), "-module-cache-path", str(output / "ModuleCache")]
    mac_sources = [str(snapshot / path.relative_to(root)) for path in sources if path.name != "HarnessApp.swift"]
    mobile_sources = [str(snapshot / path.relative_to(root)) for path in sources if path.name != "MobileMain.swift"]
    links = ["-framework", "Metal", "-framework", "MetalPerformanceShaders", "-framework", "Security",
             "-lobjc", "-liconv", "-lresolv", "-lc++"]
    error = None
    try:
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], env=environment, text=True).strip()
        run("compile", ["xcrun", "swiftc", "-sdk", sdk, "-target", "arm64-apple-macos15.0", *flags,
                        *mac_sources, str(archive), *links, "-o", str(output / "mobile-check")])
        run("regressions", [str(output / "mobile-check")])
        if args.fixture:
            fixture = output / "tiny-fixture"
            fixture.mkdir()
            for name in ("config.json", "model.safetensors"):
                source = args.fixture.resolve(strict=True) / name
                data = source.read_bytes()
                if len(data) > 1024 * 1024:
                    raise RuntimeError("Native tiny fixture exceeds its size bound")
                (fixture / name).write_bytes(data)
                pins[str(source.relative_to(root))] = hashlib.sha256(data).hexdigest()
            run("worker-revision", [str(output / "mobile-check"), "--polling", str(fixture)])
        run("render", [str(output / "mobile-check"), "--render", str(output / "previews")])
        if args.model:
            run("local-model", [str(output / "mobile-check"), "--model", str(args.model.resolve(strict=True))])
            if args.switch_model:
                run("replacement-admission", [str(output / "mobile-check"), "--switching", str(args.model.resolve(strict=True)), str(args.switch_model.resolve(strict=True))])
        for platform, target in [("iphoneos", "arm64-apple-ios18.0"), ("iphonesimulator", "arm64-apple-ios18.0-simulator")]:
            sdk = subprocess.check_output(["xcrun", "--sdk", platform, "--show-sdk-path"], env=environment, text=True).strip()
            run(platform, ["xcrun", "swiftc", "-sdk", sdk, "-target", target, *flags, "-emit-module",
                           "-module-name", "TokletMobileCheck", "-emit-module-path", str(output / (platform + ".swiftmodule")),
                           *mobile_sources])
        if sha(archive) != native_pin:
            raise RuntimeError("Native archive changed during check")
    except (OSError, subprocess.SubprocessError, RuntimeError) as failure:
        error = str(failure)
    report = dict(passed=error is None, source_pins=pins, native_archive=str(archive), native_sha256=native_pin,
                  commands=records, error=error, physical_device_tested=False)
    (output / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()

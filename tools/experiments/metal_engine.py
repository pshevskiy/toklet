"""Explicit resident Qwen qualification, separate child and idle footprint reads.

These samples establish correctness/repetition/memory evidence, not speed parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os
import selectors
import signal
import time

ROOT = Path(__file__).resolve().parents[2]
import native_usage


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def kill_owned_group(child):
    try:
        os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin may report EPERM for a group containing only its zombie leader.
        # Reap that owned leader and retry; live-leader permission failures remain errors.
        if child.poll() is None:
            raise
        try:
            os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass


class Transport:
    """Bounded nonblocking pipes, with no reader threads or stream-close locks."""
    def __init__(self, child, raw, stderr, stdout_limit=8*1024*1024, event_limit=4096,
                 request_timeout=30, lifetime_timeout=180):
        if not 1 <= stdout_limit <= 16*1024*1024 or not 1 <= event_limit <= 32768:
            raise ValueError("transport limits outside admitted diagnostic bounds")
        self.stdout_limit, self.event_limit = stdout_limit, event_limit
        if (type(request_timeout) not in (int,float) or type(lifetime_timeout) not in (int,float)
                or not 0<request_timeout<=lifetime_timeout<=86400):
            raise ValueError("transport time limits outside bounded diagnostic scope")
        self.request_timeout,self.lifetime_timeout=request_timeout,lifetime_timeout
        self.child, self.raw, self.stderr = child, raw, stderr
        self.selector = selectors.DefaultSelector()
        for name, stream in [("stdout",child.stdout),("stderr",child.stderr)]:
            os.set_blocking(stream.fileno(),False)
            self.selector.register(stream,selectors.EVENT_READ,name)
        self.pending = bytearray()
        self.stdout_bytes = self.stderr_bytes = self.events = 0
        self.stdout_closed = False
        self.started = time.monotonic()
    def poll(self, timeout):
        for key,_ in self.selector.select(timeout):
            data = os.read(key.fileobj.fileno(),65536)
            if not data:
                self.selector.unregister(key.fileobj)
                if key.data == "stdout": self.stdout_closed = True
                continue
            if key.data == "stderr":
                allowed = 1024*1024-self.stderr_bytes
                self.stderr.write(data[:allowed]);self.stderr.flush()
                self.stderr_bytes += len(data)
                if self.stderr_bytes > 1024*1024:
                    raise ValueError("child stderr exceeds 1MiB bound")
            else:
                self.stdout_bytes += len(data)
                if self.stdout_bytes > self.stdout_limit:
                    raise ValueError("child stdout exceeds admitted byte bound")
                self.pending.extend(data)
                if max(map(len,self.pending.split(b"\n"))) > 65536:
                    raise ValueError("child output exceeds bounded line limit")
    def read(self):
        deadline = min(time.monotonic()+self.request_timeout,self.started+self.lifetime_timeout)
        while True:
            if b"\n" in self.pending:
                line,_,tail = self.pending.partition(b"\n")
                self.pending = bytearray(tail)
                self.events += 1
                if self.events > self.event_limit: raise ValueError("child event count exceeds bound")
                line = line.decode("utf-8")+"\n"
                self.raw.write(line);self.raw.flush()
                return json.loads(line)
            if self.stdout_closed:
                raise ValueError("child ended unexpectedly or emitted truncated JSON")
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError("child startup/request event timed out")
            self.poll(remaining)
    def finish(self):
        deadline = time.monotonic()+20
        while self.child.poll() is None:
            if time.monotonic() >= deadline: raise TimeoutError("child EOF shutdown timed out")
            self.poll(min(.1,deadline-time.monotonic()))
        self.poll(0)
        return self.child.wait(timeout=1)
    def close(self):
        self.selector.close()


def run(binary, output):
    checkpoint = ROOT / "models/Qwen2.5-0.5B-Instruct/model.safetensors"
    identity = digest(checkpoint)
    if identity != "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe":
        raise ValueError("checkpoint identity differs")
    result = {"binary_sha256": digest(binary), "checkpoint_sha256": identity,
              "scope": "Resident GPU correctness and bounded repeated request evidence; no comparative performance claim", "runs": []}
    paths = [output, output.with_suffix(".stderr"), output.with_suffix(".raw.jsonl")]
    if any(path.exists() for path in paths):
        raise ValueError("evidence destination already exists")
    source_paths = sorted(set([Path(__file__), Path(native_usage.__file__), ROOT/"Cargo.toml", ROOT/"Cargo.lock", ROOT/"tests/metal.rs"] + list((ROOT/"src").rglob("*.rs")) + list((ROOT/"src").rglob("*.metal"))))
    assets = [checkpoint] + sorted(checkpoint.parent.glob("*.json"))
    provenance = {str(path.relative_to(ROOT)): digest(path) for path in source_paths + assets}
    result.update(provenance=provenance, engine_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), build="cargo build --offline --release --features metal --bin toklet; CARGO_TARGET_DIR=artifacts/m4-metal-engine-v1", hardware=subprocess.check_output(["/usr/bin/uname","-m"],text=True).strip(), platform=subprocess.check_output(["/usr/bin/sw_vers"],text=True).strip())
    child = None
    started = time.monotonic()
    try:
      with paths[0].open("x") as report, paths[1].open("xb") as stderr, paths[2].open("x") as raw:
        try:
          child = subprocess.Popen([str(binary), "--model", str(checkpoint.parent), "--backend", "metal", "--context-capacity", "256", "--prefill-chunk", "32", "--interactive"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
          transport = Transport(child,raw,stderr)
          read = transport.read
          def send(request):
              child.stdin.write((json.dumps(request)+"\n").encode("utf-8"))
              child.stdin.flush()
          ready = read()
          if ready.get("event") != "ready" or ready.get("backend") != "metal":
              raise ValueError(ready)
          result["ready"] = ready
          for i in range(13):
              send({"action": "generate", "tokens": [9707, 11, 358, 2776], "max_new_tokens": 8})
              tokens = []
              while True:
                  event = read()
                  if event["event"] == "token":
                      tokens.append(event["token_id"])
                  elif event["event"] == "text":
                      pass
                  elif event["event"] == "done":
                      break
                  else:
                      raise ValueError(event)
              if not tokens or tokens[0] != 4460 or (i and tokens != result["runs"][0]["tokens"]):
                  raise ValueError("frozen first greedy ID or repeated generation differs")
              result["runs"].append({"phase": "warmup" if i < 3 else "sample", "tokens": tokens, "statistics": event["statistics"]})
              send({"action": "reset"})
              if read().get("event") != "reset":
                  raise ValueError("reset boundary differs")
              if i in (2, 12):
                  peak = native_usage.usage(child.pid)["lifetime_max_phys_footprint"]
                  if peak is None or peak > 6 * 1024**3:
                      raise ValueError("idle footprint unavailable/outside admission")
                  result[f"idle_peak_after_{i+1}_requests"] = peak
          send({"action": "generate", "prompt": "Explain why the sky is blue in two sentences.", "max_new_tokens": 96})
          text, tokens = [], []
          while True:
              event = read()
              if event["event"] in ("token", "text"):
                  text.append(event["text"])
                  if event["event"] == "token":
                      tokens.append(event["token_id"])
              elif event["event"] == "done":
                  break
              else:
                  raise ValueError(event)
          if event["statistics"]["generation"]["reason"]["kind"] != "stop_token":
              raise ValueError("complete chat did not reach configured EOS")
          result["chat"] = {"tokens": tokens, "text": "".join(text), "statistics": event["statistics"]}
          result["idle_usage_final"] = native_usage.usage(child.pid)
          child.stdin.close()
          if transport.finish() != 0:
              raise ValueError("child failed")
        except Exception as error:
          result["error"] = str(error)
          raise
        finally:
          if child is not None:
              # Own fresh group, including descendants holding pipes after leader exit.
              kill_owned_group(child)
              child.wait(timeout=10)
              if 'transport' in locals(): transport.close()
              for stream in (child.stdin,child.stdout,child.stderr):
                  if stream and not stream.closed: stream.close()
          result["endpoint_provenance"] = {str(path.relative_to(ROOT)): digest(path) for path in source_paths + assets}
          result["endpoint_binary_sha256"] = digest(binary)
          result["identities_unchanged"] = result["endpoint_provenance"] == provenance and result["endpoint_binary_sha256"] == result["binary_sha256"]
          if not result["identities_unchanged"]:
              result["error"] = "qualification identities changed"
          result["memory_caveat"] = "Public libproc v4 native charge counters may omit clean/shared pages; numerical admission counts persistent physical allocations once."
          report.write(json.dumps(result,indent=2)+"\n");report.flush()
    finally:
        if child is not None and child.poll() is None:
            kill_owned_group(child)
            child.wait(timeout=10)
    if not result["identities_unchanged"]:
        raise ValueError("qualification identities changed")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.binary.resolve(), args.output.resolve())

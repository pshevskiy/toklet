# M6 supervised long runs

The remaining serial queue runs in its own process session, so a chat-tool
interruption leaves the numerical collector attached to its worker. Explicit
pause and launcher publication failures use one graceful SIGINT to the verified
owned queue group. The supervisor waits for the interrupted collector, whose
existing cleanup kills and reaps its separate-session numerical worker. No
numerical kernel, request, quality gate or collector timeout changes.

## Preserved interruption

Thirteen of the original 23 queued roles completed before the chat execution
interruption: the successful larger32K quality retry, six profiles and six small
4K/8K/16K boundary/confirmation roles. Their records remain unchanged. The
in-progress small32K ordinary attempt lost its queue and collector while its
separate-session worker remained alive. Read-only process inventory identified
that owned worker as PID/PGID 61156 with the exact admitted command. The root
agent stopped that orphan with SIGTERM and verified its absence before any
further tests or measurements.

The original attempt remains incomplete and unqualified: its JSON report has
zero bytes, raw events contain only the 1,056-byte ready event, and stderr is
empty. These files and the original queue/status/log are preserved byte for
byte, with a separate interruption record. No completed numerical sample or
clean exit is inferred. The fresh retry uses small/long/32768/v2; the untouched
remaining nine roles retain their original planned v1 destinations.

## Ownership and failure behavior

The launcher reserves exclusive owner and log paths, starts one serial
supervisor in a separate session, records its PID/group and source identity,
and closes both metadata/output files inside the cleanup boundary. The queue
executes one collector at a time and stops on a failed result. Each collector
retains the unchanged 12-hour event and 24-hour worker lifetime policies, owned
worker cleanup, memory checks and final full-capacity recomputation.

A metadata write, flush or file-close failure triggers graceful interruption,
allowing Python cleanup to run. Signal or wait failures add diagnostics to the
original error instead of replacing it. The queue joins an interrupted
collector for up to 120 seconds; launcher cleanup waits up to 180 seconds.
Any cleanup failure requires verifying owned process absence before a retry.
These are error-path waits, without changing numerical measurement lifetimes.

Eight CPU-only tests pass: detached ownership, duplicate prevention, metadata
failure, signal/wait failure diagnostics, both file-close failure paths,
collector cleanup joins, preservation of an interrupted cleanup error, and an
actual bounded three-process signal fixture. The fixture creates a queue,
collector and separate-session worker, sends group SIGINT, and verifies worker
absence plus the collector's cleanup marker before supervisor exit 130.
An earlier mock-test fixture used an unresolved macOS temporary-directory alias;
resolving the fixture path fixed that expectation, with no policy change. Its
failed log remains preserved.

The [qualification receipt](../../benchmarks/results/2026-10-16-m6-supervision-qualification-v1.json)
binds three current source files and 11 exact published evidence copies.
Astra independently accepts the source and all eight tests. Astra accepted final publication after all source/evidence pins, exact copies,
interruption preservation, role selection, report and links verified with no
remaining findings. All review execution and hashing finished before launch.
The receipt records this prelaunch CPU qualification; it makes no claim that
the actual detached numerical queue had already started. This qualifies supervision behavior; actual GPU collections,
full boundary coverage and the final M6 decision remain separate.

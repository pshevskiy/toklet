import Foundation
import CToklet

struct NativeSettings: Sendable {
    var directory: String
    var packedDirectory = ""
    var metal = false
    var capacity: UInt32 = 32
    var chunk: UInt32 = 3
    var numericalBudget: UInt64 = 128 * 1024 * 1024

    func options() -> TokletOptions {
        TokletOptions(abi_version: TOKLET_ABI_VERSION,
                      struct_size: UInt32(MemoryLayout<TokletOptions>.size),
                      capacity: capacity, chunk: chunk, memory_limit_bytes: numericalBudget)
    }
}

enum NativeCommand: Sendable {
    case load(NativeSettings)
    // Raw IDs only. autoCancelAfter is a deterministic diagnostic interruption.
    case generate(operation: UInt32, tokens: [UInt32], maximum: UInt32,
                  stops: [UInt32], autoCancelAfter: Int = 0)
    case score(target: UInt32)
    case reset, suspend, replay, unload
}

// The native elapsed clock starts inside generation; callback uptime is the
// monotonic process clock observed at the Swift callback. Neither uses UI polls.
struct NativeTokenTiming: Codable, Sendable {
    let index: UInt32
    let nativeElapsedSeconds: Double
    let callbackUptimeNanoseconds: UInt64
}

// Scalar diagnostic only; the owning engine keeps all full logits resident.
struct NativeDistributionScore: Codable, Equatable, Sendable {
    let target: UInt32
    let greedy: UInt32
    let vocabulary: UInt32
    let evaluated: UInt32
    let targetLogit: Double
    let greedyLogit: Double
    let negativeLogLikelihood: Double
}

struct NativeSnapshot: Sendable {
    var busy = false
    var closed = false
    var hasModel = false
    var hasSession = false
    var phase = "Idle"
    var status: UInt32 = 0
    var message = ""
    var ids: [UInt32] = [] // At most 128, retained until the next command.
    var tokenTimings: [NativeTokenTiming] = [] // Same bounded emitted-token order.
    var reason: UInt32 = 0
    // Meaningful only for reason 2; hidden EOS remains in sampled history.
    var stopToken: UInt32 = .max
    var history: UInt32 = 0
    var evaluated: UInt32 = 0
    var pending: UInt32 = 0
    var savedHistory = 0
    var admittedBytes: UInt64 = 0
    var score: NativeDistributionScore?
    var workerFinished = false
}

private func nativeError(_ code: UInt32, _ error: inout TokletError) -> NSError {
    let text = withUnsafeBytes(of: &error.message) { bytes in
        String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
    return NSError(domain: "TokletNative", code: Int(code),
                   userInfo: [NSLocalizedDescriptionKey: text.isEmpty ? "Native status \(code)" : text])
}
private func requireOK(_ code: UInt32, _ error: inout TokletError) throws {
    if code != TOKLET_OK { throw nativeError(code, &error) }
}
private func invalid(_ text: String) -> NSError {
    NSError(domain: "TokletHarness", code: Int(TOKLET_INVALID),
            userInfo: [NSLocalizedDescriptionKey: text])
}

// @unchecked Sendable is confined to this synchronization boundary. Condition
// protects every shared field, including the independent atomic cancel handle's
// lifetime. Model/session pointers live only in Engine on the dedicated Thread.
// Call close() explicitly. waitUntilClosed() is for non-UI checks, never the UI.
final class NativeWorker: @unchecked Sendable {
    private let condition = NSCondition()
    private var queued: NativeCommand?
    private var closing = false
    private var view = NativeSnapshot()
    private var cancelFlag: OpaquePointer?
    private var thread: Thread?
    // Cold diagnostic completion observer for deterministic non-UI checks.
    // Runs after FFI returns, outside the condition lock, around publication;
    // the after observer receives only whether a score was published. Neither
    // receives native owners. Production callers leave both nil.
    private let beforeScorePublication: (@Sendable () -> Void)?
    private let afterScorePublication: (@Sendable (Bool) -> Void)?

    init(beforeScorePublication: (@Sendable () -> Void)? = nil,
         afterScorePublication: (@Sendable (Bool) -> Void)? = nil) throws {
        self.beforeScorePublication = beforeScorePublication
        self.afterScorePublication = afterScorePublication
        var error = TokletError()
        try requireOK(toklet_cancel_create(&cancelFlag, &error), &error)
        view.ids.reserveCapacity(128)
        view.tokenTimings.reserveCapacity(128)
        let thread = Thread { [weak self] in self?.run() }
        thread.name = "Toklet inference"
        self.thread = thread
        thread.start()
    }

    func snapshot() -> NativeSnapshot {
        condition.lock(); defer { condition.unlock() }
        return view
    }

    @discardableResult
    func submit(_ command: NativeCommand, initiallyCancelled: Bool = false) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard !closing, !view.closed, !view.busy, let flag = cancelFlag else { return false }
        // Clear only while idle, under the same lock as cancel/enqueue. A cancel
        // arriving after this point survives until the request observes it.
        _ = toklet_cancel_set(flag, initiallyCancelled ? 1 : 0, nil)
        queued = command
        view.busy = true
        view.phase = "Queued"
        view.score = nil
        view.stopToken = .max
        view.ids.removeAll(keepingCapacity: true)
        view.tokenTimings.removeAll(keepingCapacity: true)
        condition.signal()
        return true
    }

    // Non-UI diagnostics can wait for command/queued-cleanup completion without
    // retaining repeated snapshots or estimating completion from a polling tick.
    // NSCondition uses a wall-clock deadline; event timestamps stay monotonic.
    func waitUntilIdle(seconds: TimeInterval) -> NativeSnapshot? {
        guard seconds.isFinite, seconds >= 0, seconds <= 3600 else { return nil }
        condition.lock(); defer { condition.unlock() }
        let end = Date(timeIntervalSinceNow: seconds)
        while view.busy {
            if !condition.wait(until: end) { return view.busy ? nil : view }
        }
        return view
    }

    func cancel() {
        condition.lock(); defer { condition.unlock() }
        if let flag = cancelFlag { _ = toklet_cancel_set(flag, 1, nil) }
    }

    // Lifecycle cleanup supersedes a queued request and runs after active work
    // returns. Active native resources are never released concurrently.
    func interrupt(unload: Bool) {
        condition.lock(); defer { condition.unlock() }
        guard !closing, !view.closed else { return }
        if let flag = cancelFlag { _ = toklet_cancel_set(flag, 1, nil) }
        // Once a full unload is queued, a later background event cannot weaken it.
        if case .unload? = queued {} else { queued = unload ? .unload : .suspend }
        view.busy = true
        view.phase = "Cancelling for lifecycle cleanup"
        view.score = nil
        view.stopToken = .max
        condition.signal()
    }

    func close() {
        condition.lock(); defer { condition.unlock() }
        guard !view.closed else { return }
        closing = true; queued = nil
        view.score = nil
        view.stopToken = .max
        if let flag = cancelFlag { _ = toklet_cancel_set(flag, 1, nil) }
        view.busy = true
        condition.signal()
    }

    func waitUntilClosed(seconds: TimeInterval) -> Bool {
        condition.lock(); defer { condition.unlock() }
        let end = Date(timeIntervalSinceNow: seconds)
        while !view.closed {
            if !condition.wait(until: end) { return view.closed }
        }
        return true
    }

    fileprivate func record(_ event: TokletTokenEvent, autoCancelAfter: Int) -> UInt32 {
        condition.lock(); defer { condition.unlock() }
        guard view.ids.count < 128, event.index == UInt32(view.ids.count),
              event.elapsed_seconds.isFinite, event.elapsed_seconds >= 0,
              event.elapsed_seconds >= (view.tokenTimings.last?.nativeElapsedSeconds ?? 0) else { return 1 }
        let timing = NativeTokenTiming(index:event.index,nativeElapsedSeconds:event.elapsed_seconds,
                                       callbackUptimeNanoseconds:DispatchTime.now().uptimeNanoseconds)
        view.ids.append(event.token_id)
        view.tokenTimings.append(timing)
        if autoCancelAfter > 0, view.ids.count == autoCancelAfter, let flag = cancelFlag {
            _ = toklet_cancel_set(flag, 1, nil)
        }
        return 0
    }

    private func run() {
        precondition(!Thread.isMainThread)
        // This object's model/session fields never leave this OS thread.
        let engine = Engine()
        while true {
            condition.lock()
            while queued == nil && !closing { condition.wait() }
            if closing { condition.unlock(); break }
            let command = queued!
            queued = nil
            let flag = cancelFlag!
            view.phase = "Running"
            condition.unlock()
            var failure: NSError?
            do { try autoreleasepool { try engine.perform(command, flag: flag, worker: self) } }
            catch { failure = error as NSError }
            if case .score = command { beforeScorePublication?() }
            condition.lock()
            view.hasModel = engine.model != nil
            view.hasSession = engine.session != nil
            view.savedHistory = engine.savedCount
            view.history = engine.report.history_tokens
            view.evaluated = engine.report.evaluated_tokens
            view.pending = engine.report.pending_tokens
            view.reason = engine.report.reason
            view.stopToken = failure == nil && !closing && queued == nil && engine.report.reason == 2
                ? engine.report.stop_token : .max
            view.admittedBytes = engine.admissionUpperBound
            view.score = failure == nil && !closing && queued == nil ? engine.score : nil
            view.status = UInt32(clamping: failure?.code ?? 0)
            view.message = String((failure?.localizedDescription ?? "").prefix(512))
            view.busy = closing || queued != nil
            view.phase = view.busy ? "Cleanup queued" : engine.phase
            let publishedScore = view.score != nil
            condition.broadcast()
            condition.unlock()
            if case .score = command { afterScorePublication?(publishedScore) }
        }
        var releaseError: NSError?
        do { try autoreleasepool { try engine.unload() } } catch { releaseError = error as NSError }
        condition.lock()
        // All native calls have returned; setters cannot overlap flag release.
        if let flag = cancelFlag {
            var error = TokletError()
            let code = toklet_cancel_release(flag, &error)
            if code != TOKLET_OK { releaseError = nativeError(code, &error) }
            cancelFlag = nil
        }
        view.hasModel = false; view.hasSession = false
        view.savedHistory = 0; view.busy = false
        view.closed = true; view.workerFinished = true
        view.score = nil
        view.stopToken = .max
        view.phase = releaseError == nil ? "Closed" : "Release failed; restart required"
        if let error = releaseError {
            view.status = UInt32(clamping: error.code); view.message = error.localizedDescription
        }
        thread = nil
        condition.broadcast(); condition.unlock()
    }
}

private final class Sink {
    let worker: NativeWorker
    let autoCancelAfter: Int
    init(_ worker: NativeWorker, _ count: Int) { self.worker = worker; autoCancelAfter = count }
}
private let nativeSink: @convention(c) (UnsafeMutableRawPointer?, TokletTokenEvent) -> UInt32 = { context, event in
    guard let context else { return 1 }
    let sink = Unmanaged<Sink>.fromOpaque(context).takeUnretainedValue()
    return sink.worker.record(event, autoCancelAfter: sink.autoCancelAfter)
}

// Private thread-local native owners. All FFI callbacks are synchronous and the
// Sink remains strongly owned until generate returns. No actor/queue is used as
// a substitute for the C boundary's OS-thread affinity.
private final class Engine {
    var model: OpaquePointer?
    var session: OpaquePointer?
    var settings: NativeSettings?
    var saved: [UInt32] = []
    var savedCount = 0
    var admissionUpperBound: UInt64 = 0
    var report = TokletReport()
    var phase = "Idle"
    var score: NativeDistributionScore?

    func releaseSession() throws {
        score = nil
        if let value = session {
            session = nil
            var error = TokletError()
            try requireOK(toklet_session_release(value, &error), &error)
        }
        report = TokletReport()
    }
    func unload() throws {
        try releaseSession()
        if let value = model {
            model = nil
            var error = TokletError()
            try requireOK(toklet_model_release(value, &error), &error)
        }
        saved = []; savedCount = 0; admissionUpperBound = 0; settings = nil; phase = "Unloaded; fresh prompt required"
    }
    func createSession() throws {
        guard let model, let settings else { throw invalid("Load a model first") }
        var options = settings.options(); var error = TokletError()
        try requireOK(toklet_session_create(model, &options, &session, &error), &error)
        try requireOK(toklet_session_state(session, &report, &error), &error)
        admissionUpperBound = max(admissionUpperBound,report.admitted_bytes)
    }
    func refreshHistory() throws {
        guard let session else { return }
        var count = 0; var error = TokletError()
        let code = saved.withUnsafeMutableBufferPointer {
            toklet_session_history(session, $0.baseAddress, $0.count, &count, &error)
        }
        try requireOK(code, &error)
        savedCount = count
    }
    func generate(_ operation: UInt32, _ tokens: [UInt32], _ maximum: UInt32,
                  _ stops: [UInt32], _ autoCancel: Int, _ flag: OpaquePointer,
                  _ worker: NativeWorker) throws {
        guard let session else { throw invalid("Load or replay a session first") }
        guard tokens.count <= 32768, stops.count <= 32768, maximum <= 128,
              autoCancel >= 0 && autoCancel <= 128 else { throw invalid("Request exceeds harness bounds") }
        let sink = Sink(worker, autoCancel)
        var error = TokletError(); report = TokletReport()
        let code = withExtendedLifetime(sink) {
            tokens.withUnsafeBufferPointer { input in
                stops.withUnsafeBufferPointer { stop in
                toklet_session_generate(session, operation, input.baseAddress, input.count,
                    maximum, stop.baseAddress, stop.count, flag, nativeSink,
                    Unmanaged.passUnretained(sink).toOpaque(), &report, &error)
                }
            }
        }
        // Query exact state even on recoverable generation failure. Poisoned or
        // otherwise unreadable state cannot be offered for replay.
        do { try refreshHistory() }
        catch {
            savedCount = 0
            try? releaseSession()
            phase = "Session unavailable; reset or reload required"
            throw error
        }
        if report.abi_version != TOKLET_ABI_VERSION {
            var queryError = TokletError()
            try requireOK(toklet_session_state(session, &report, &queryError), &queryError)
        }
        phase = "Ready"
        try requireOK(code, &error)
    }
    func perform(_ command: NativeCommand, flag: OpaquePointer, worker: NativeWorker) throws {
        score = nil
        switch command {
        case .score(let target):
            guard let session else { throw invalid("Load or replay a session first") }
            var error = TokletError(); var value = TokletScore()
            try requireOK(toklet_session_score(session,target,&value,&error),&error)
            guard value.abi_version == TOKLET_ABI_VERSION,
                  value.struct_size == UInt32(MemoryLayout<TokletScore>.size),
                  value.target_token == target, value.vocab_size > 0,
                  target < value.vocab_size, value.greedy_token < value.vocab_size,
                  value.evaluated_tokens > 0, value.evaluated_tokens == report.evaluated_tokens,
                  report.history_tokens == report.evaluated_tokens, report.pending_tokens == 0,
                  value.target_logit.isFinite, value.greedy_logit.isFinite,
                  value.greedy_logit >= value.target_logit,
                  value.negative_log_likelihood.isFinite, value.negative_log_likelihood >= 0 else {
                throw invalid("Invalid committed-distribution score record")
            }
            score = NativeDistributionScore(target:target,greedy:value.greedy_token,
                vocabulary:value.vocab_size,evaluated:value.evaluated_tokens,
                targetLogit:value.target_logit,greedyLogit:value.greedy_logit,
                negativeLogLikelihood:value.negative_log_likelihood)
            phase = "Ready"

        case .load(let requested):
            guard model == nil else { throw invalid("Unload the current model first") }
            guard (1...32768).contains(requested.capacity), requested.chunk > 0 else { throw invalid("Invalid capacity/chunk") }
            guard requested.directory.utf8.count <= 16384, requested.packedDirectory.utf8.count <= 16384 else {
                throw invalid("Artifact path exceeds 16 KiB")
            }
            let path = Array(requested.directory.utf8); let packed = Array(requested.packedDirectory.utf8)
            guard !path.isEmpty, path.count <= 16384, packed.count <= 16384,
                  !path.contains(0), !packed.contains(0) else { throw invalid("Invalid artifact path") }
            var options = requested.options(); var error = TokletError()
            let code = path.withUnsafeBufferPointer { path in
                packed.withUnsafeBufferPointer { packed in
                    requested.metal ? toklet_model_load_metal(path.baseAddress,path.count,packed.baseAddress,packed.count,&options,&model,&error)
                                    : toklet_model_load(path.baseAddress,path.count,&options,&model,&error)
                }
            }
            try requireOK(code, &error)
            settings = requested
            saved = Array(repeating: 0, count: Int(requested.capacity)); savedCount = 0
            do { try createSession() } catch { try? unload(); throw error }
            try requireOK(toklet_session_state(session,&report,&error),&error)
            phase = "Ready"
        case .generate(let operation,let tokens,let maximum,let stops,let autoCancel):
            try generate(operation,tokens,maximum,stops,autoCancel,flag,worker)
        case .reset:
            guard let session else { throw invalid("Load or replay a session first") }
            var error = TokletError()
            try requireOK(toklet_session_reset(session,&error),&error)
            savedCount = 0
            try requireOK(toklet_session_state(session,&report,&error),&error)
            phase = "Ready"
        case .suspend:
            try releaseSession()
            phase = model == nil ? "Unloaded" : "Session released; replay required"
        case .replay:
            guard session == nil else { throw invalid("Suspend the session before replay") }
            let prefix = Array(saved.prefix(savedCount))
            func restorePrefix() {
                saved.replaceSubrange(0..<prefix.count,with:prefix)
                savedCount = prefix.count
            }
            do {
                try createSession()
                if !prefix.isEmpty {
                    // Replay is transactional in the application: cancellation
                    // may leave native history empty or only partially evaluated.
                    try generate(0,prefix,0,[],0,flag,worker)
                    if report.reason == 3 {
                        try releaseSession(); restorePrefix(); report.reason = 3
                        phase = "Replay cancelled; prefix retained"
                        return
                    }
                    guard savedCount == prefix.count, Array(saved.prefix(savedCount)) == prefix,
                          report.evaluated_tokens == UInt32(prefix.count), report.pending_tokens == 0 else {
                        throw invalid("Incomplete replay; retained prefix required")
                    }
                }
                phase = "Ready"
            } catch {
                try? releaseSession(); restorePrefix()
                phase = "Replay failed; prefix retained"
                throw error
            }
        case .unload:
            try unload()
        }
    }
}

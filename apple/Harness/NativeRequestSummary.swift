import Foundation
import CToklet

// O1: fixed records, collected only at request/phase boundaries. No text, IDs
// of generated tokens, file paths, native pointers or numerical arrays are kept.
enum NativeRequestSummaryMode: Sendable {
    case disabled, summary
    static var configured: Self {
        #if os(macOS)
        ProcessInfo.processInfo.environment["TOKLET_REQUEST_SUMMARIES"] == "1" ? .summary : .disabled
        #else
        .disabled
        #endif
    }
}

// Validated copy of the native engine's already-completed Report. The native
// clock is independent of callback uptime. Unknown reports never become zeros.
struct NativeGenerationMetrics: Codable, Sendable {
    let sessionInstance, reusedTokens, prefillTokens, sampledTokens, emittedTokens: UInt64
    let prefillSeconds, decodeSeconds, samplingSeconds, totalSeconds: Double
    let firstTokenSeconds: Double?
    let prefillTokensPerSecond: Double?

    init?(_ value: TokletGenerationMetrics, expectedSession: UInt64) {
        let report = UInt32(TOKLET_GENERATION_REPORT_KNOWN), first = UInt32(TOKLET_GENERATION_FIRST_TOKEN_KNOWN)
        let times = [value.prefill_seconds,value.decode_seconds,value.sampling_seconds,value.total_seconds,value.first_token_seconds]
        guard value.abi_version == TOKLET_ABI_VERSION,
              value.struct_size == UInt32(MemoryLayout<TokletGenerationMetrics>.size), value.reserved == 0,
              value.known_fields & report != 0, value.known_fields & ~(report | first) == 0,
              value.session_instance == expectedSession, expectedSession > 0,
              value.reused_tokens <= 32768, value.prefill_tokens <= 32768,
              value.sampled_tokens <= 32768, value.emitted_tokens <= value.sampled_tokens,
              times.allSatisfy({ $0.isFinite && $0 >= 0 }),
              value.prefill_seconds + value.decode_seconds + value.sampling_seconds <= value.total_seconds + 1e-9,
              value.first_token_seconds <= value.total_seconds,
              (value.known_fields & first != 0) == (value.emitted_tokens > 0),
              value.known_fields & first != 0 || value.first_token_seconds == 0 else { return nil }
        sessionInstance = value.session_instance; reusedTokens = value.reused_tokens
        prefillTokens = value.prefill_tokens; sampledTokens = value.sampled_tokens; emittedTokens = value.emitted_tokens
        prefillSeconds = value.prefill_seconds; decodeSeconds = value.decode_seconds
        samplingSeconds = value.sampling_seconds; totalSeconds = value.total_seconds
        firstTokenSeconds = value.known_fields & first == 0 ? nil : value.first_token_seconds
        let rate = value.prefill_tokens > 0 && value.prefill_seconds > 0
            ? Double(value.prefill_tokens) / value.prefill_seconds : nil
        guard rate?.isFinite ?? true else { return nil }
        prefillTokensPerSecond = rate
    }
}

enum NativeGenerationMetricStatus: String, Codable, Sendable {
    case notStarted, completed, generationFailed, invalidRecord
}

struct NativeRequestObservation: Sendable {
    enum Phase: String, Codable { case queued, encoding, evaluation, decoding, completion }
    var phase: Phase = .queued
    var phaseStart: UInt64 = 0
    var encodingNanoseconds: UInt64?
    var evaluationNanoseconds: UInt64?
    var decodingNanoseconds: UInt64?
    var promptTokens: UInt64?
    var evaluationStarted = false
    var failurePhase: Phase?
    var nativeMetrics: NativeGenerationMetrics?
    var nativeMetricStatus: NativeGenerationMetricStatus = .notStarted
    mutating func enter(_ next: Phase) {
        finishPhase()
        phase = next; phaseStart = DispatchTime.now().uptimeNanoseconds
        if next == .evaluation { evaluationStarted = true }
    }
    mutating func finishPhase() {
        guard phaseStart > 0 else { return }
        let duration = DispatchTime.now().uptimeNanoseconds - phaseStart
        switch phase {
        case .encoding: encodingNanoseconds = duration
        case .evaluation: evaluationNanoseconds = duration
        case .decoding: decodingNanoseconds = duration
        default: break
        }
        phaseStart = 0
    }
}

struct NativeRequestSummary: Codable, Sendable {
    enum Outcome: String, Codable { case outputLimit, stopToken, cancelled, rejected, failed }
    struct State: Codable, Equatable, Sendable {
        let modelInstance, sessionInstance, history, committed, pending: UInt64
        init(_ context: NativeContextFacts) {
            modelInstance = context.modelInstance; sessionInstance = context.sessionInstance
            history = context.history; committed = context.committed; pending = context.pending
        }
    }
    var schemaVersion: Int = 2
    let sequence: UInt64
    let requestID: UUID
    let stateGeneration: UInt64
    let submittedUptimeNanoseconds: UInt64
    var startedUptimeNanoseconds: UInt64?
    var completedUptimeNanoseconds: UInt64?
    var firstCallbackNanoseconds: UInt64?
    var encodingNanoseconds: UInt64?
    // Includes synchronous callbacks; GPU time is not added to this duration.
    var evaluationNanoseconds: UInt64?
    var decodingNanoseconds: UInt64?
    var promptTokens: UInt64?
    var nativeMetrics: NativeGenerationMetrics?
    var nativeMetricStatus: NativeGenerationMetricStatus = .notStarted
    var nativeFirstTokenSeconds: Double?
    var nativeLastTokenSeconds: Double?
    var callbackTTFTMilliseconds: Double?
    var callbackDecodeTokensPerSecond: Double?
    var nativeDecodeTokensPerSecond: Double?
    var requestMilliseconds: Double?
    var queueMilliseconds: Double?
    var cancelToTerminalMilliseconds: Double?
    var cancellationUptimeNanoseconds: UInt64?
    var outputTextBytes: UInt64?
    var replayRequired: Bool?
    let requestedOutputTokens: UInt32
    var callbackTokens: UInt32 = 0
    var nativeEmittedTokens: UInt32?
    var before: State?
    var after: State?
    var capacity: UInt32?
    var chunk: UInt32?
    var sessionAdmittedBytes: UInt64?
    var kvOwnedBytes: UInt64?
    var codecAllowanceBytes: UInt64?
    var processPhysicalBytes: UInt64?
    var processLifetimePeakBytes: UInt64?
    var outcome: Outcome = .rejected
    var failedPhase: NativeRequestObservation.Phase?
    var status: UInt32 = 0
    var evaluationStarted = false

    mutating func observeCallback(_ elapsed: Double, uptime: UInt64) {
        if callbackTokens == 0 { firstCallbackNanoseconds = uptime; nativeFirstTokenSeconds = elapsed }
        callbackTokens += 1
        nativeLastTokenSeconds = elapsed
        if let first = firstCallbackNanoseconds, uptime > first, callbackTokens > 1 {
            callbackDecodeTokensPerSecond = Double(callbackTokens - 1) * 1e9 / Double(uptime - first)
        }
        if let first = nativeFirstTokenSeconds, elapsed > first, callbackTokens > 1 {
            nativeDecodeTokensPerSecond = Double(callbackTokens - 1) / (elapsed - first)
        }
    }

    mutating func finish(observation: NativeRequestObservation?, failure: Bool,
                         status: UInt32, reason: UInt32, now: UInt64) {
        completedUptimeNanoseconds = now; self.status = status
        requestMilliseconds = Double(now - submittedUptimeNanoseconds) / 1e6
        queueMilliseconds = startedUptimeNanoseconds.map { Double($0 - submittedUptimeNanoseconds) / 1e6 }
        callbackTTFTMilliseconds = firstCallbackNanoseconds.map { Double($0 - submittedUptimeNanoseconds) / 1e6 }
        cancelToTerminalMilliseconds = cancellationUptimeNanoseconds.map { Double(now - $0) / 1e6 }
        if let observation {
            encodingNanoseconds = observation.encodingNanoseconds
            evaluationNanoseconds = observation.evaluationNanoseconds
            decodingNanoseconds = observation.decodingNanoseconds
            promptTokens = observation.promptTokens
            nativeMetrics = observation.nativeMetrics; nativeMetricStatus = observation.nativeMetricStatus
            evaluationStarted = observation.evaluationStarted
            if failure { failedPhase = observation.failurePhase ?? observation.phase }
        }
        if failure { outcome = evaluationStarted ? .failed : .rejected }
        else if reason == 3 { outcome = .cancelled }
        else if reason == 2 { outcome = .stopToken }
        else { outcome = .outputLimit }
    }
}

// All access is serialized by NativeWorker.condition. Export copies fixed records
// while idle, then encodes outside the lock. Caller owns the returned bounded Data.
final class NativeRequestSummaryStore {
    static let capacity = 32
    static let reservedBytes: UInt64 = 1 << 20
    static let maximumRecordBytes = 4096
    private var records = [NativeRequestSummary?](repeating:nil,count:capacity)
    private var next = 0
    private(set) var dropped: UInt64 = 0
    private(set) var sequence: UInt64 = 0
    private(set) var exportErrors: UInt64 = 0
    private(set) var budgetSkippedRecords: UInt64 = 0
    static func fitsBudget(admitted: UInt64, budget: UInt64) -> Bool {
        admitted <= budget && reservedBytes <= budget - admitted
    }
    func noteBudgetSkip() { budgetSkippedRecords = min(budgetSkippedRecords, UInt64.max - 1) + 1 }
    func noteExportError() { exportErrors = min(exportErrors, UInt64.max - 1) + 1 }
    init() {
        precondition(MemoryLayout<NativeRequestSummary>.stride < 16 * 1024)
        precondition(MemoryLayout<NativeRequestSummary?>.stride * Self.capacity < Int(Self.reservedBytes) / 2)
    }
    func begin(id: UUID, maximum: UInt32, generation: UInt64, now: UInt64,
               context: NativeContextFacts?, configuration: NativeSessionConfiguration?) -> NativeRequestSummary {
        sequence = min(sequence, UInt64.max - 1) + 1
        return NativeRequestSummary(sequence:sequence,requestID:id,stateGeneration:generation,
            submittedUptimeNanoseconds:now,requestedOutputTokens:maximum,
            before:context.map(NativeRequestSummary.State.init),capacity:configuration?.capacity,chunk:configuration?.chunk)
    }
    func append(_ record: NativeRequestSummary) {
        if records[next] != nil { dropped = min(dropped, UInt64.max - 1) + 1 }
        records[next] = record; next = (next + 1) % Self.capacity
    }
    func snapshot() -> ([NativeRequestSummary], UInt64) {
        ((0..<Self.capacity).compactMap { records[(next + $0) % Self.capacity] }, dropped)
    }
    static func export(_ records: [NativeRequestSummary], dropped: UInt64, exportErrors: UInt64 = 0, budgetSkipped: UInt64 = 0) throws -> Data {
        struct Manifest: Encodable {
            let schemaVersion = 2
            let kind = "request_summary_manifest"
            let profile = "google/gemma-4-E2B-it BF16 managed Mac G2"
            let revision = NativeAcceptedIdentity.Profile.gemmaBF16.revision
            let modelAndCodecManifestSHA256 = NativeAcceptedIdentity.Profile.gemmaBF16.manifestSHA256
            let clock = "DispatchTime uptime nanoseconds; durations overlap"
            let cacheAction = "Chat resets the numerical prefix (operation 3)"
            let processMemoryInstrument = "task_info TASK_VM_INFO; lifetime peak is not request peak"
            let reservedBytes = NativeRequestSummaryStore.reservedBytes
            let unavailable = ["engine/package build identity: bind export to runner build receipt",
                "GPU dispatch/wait/readback counters: not exposed by this ABI",
                "nativeMetrics absent: nativeMetricStatus explains notStarted, generationFailed or invalidRecord; partial native reports are unavailable",
                "ttft_event_ms and first-visible/final-answer/UI-update/request-UI timings: no calibrated event-to-host or render acknowledgment",
                "load/conversion/compile phases and memory pressure: separate existing load receipt, not per-request measurements",
                "token interval distribution: only first/last bounded aggregates in O1",
                "decode evaluation count: not exposed by the native Report; before/after counts are session totals",
                "replay-required: true when evaluation failed and state is unreadable; null otherwise (ABI has no poison/replay flag)",
                "weights/scratch breakdown: Gemma ABI exposes aggregate admission, KV and codec allowance",
                "missing state: native context unavailable or request had not started",
                "missing phase/footprint: phase not reached or instrument unavailable"]
            let buildIdentity: String
            let recordScope = "accepted Gemma Chat requests, plus idle/busy submit rejections while identity is visible; other models unsupported"
            let memoryAdmission = "1MiB headroom check for collection/export; fixed ring allocated at worker initialization; native admission unchanged"
            let metricScope = "request terminal is worker publication, not terminal UI; token rates use emitted callbacks, hidden stops excluded"
            let stateGenerationScope = "worker snapshot revision at submission; capacity/chunk are submission-time fields; owner IDs in before/after authoritative"
            let budgetSkippedRecords: UInt64
            let exportErrors: UInt64
            let droppedRecords: UInt64
        }
        guard records.count <= capacity else { throw NativeContextFailure("Summary export exceeds capacity") }
        guard ProcessInfo.processInfo.environment["TOKLET_BUILD_ID", default:""].utf8.count <= 256 else {
            throw NativeContextFailure("Build identity exceeds export bound")
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(Manifest(buildIdentity:ProcessInfo.processInfo.environment["TOKLET_BUILD_ID"] ?? "unavailable: bind to build receipt",budgetSkippedRecords:budgetSkipped,exportErrors:exportErrors,droppedRecords:dropped)); data.append(10)
        for record in records {
            let row = try encoder.encode(record)
            guard row.count <= maximumRecordBytes else { throw NativeContextFailure("Summary record exceeds wire limit") }
            data.append(row); data.append(10)
        }
        guard data.count <= (capacity + 1) * (maximumRecordBytes + 1) else {
            throw NativeContextFailure("Summary export exceeds wire limit")
        }
        return data
    }
}

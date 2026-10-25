import Foundation

enum GenerationStopReason: String, Equatable, Sendable {
    case outputLimit = "output_limit", stopToken = "stop_token", cancelled, failed
}

struct GenerationResponseSummary: Equatable, Sendable {
    let nextSequence, emittedTokens: UInt64
    let decodedUTF8Bytes: Int
    let terminal: GenerationStopReason?
    let sampledTotal, committedTotal, pendingTotal: UInt64?
    let sessionCommittedTokens, sessionPendingTokens: UInt64?
    let replayRequired: Bool?
    let stopTokenID: UInt32?
}

// Single owner; not Sendable. No snapshots expose its mutable byte storage.
// Synthetic consumer foundation only, awaiting the real native cursor owner.
final class GenerationResponse {
    static let maximumStoredUTF8Bytes = 8 * 1024 * 1024
    static let eventEnvelopeBytes = 2048
    private let generationID: UUID
    private let limits: GenerationModeLimits
    private let requestedOutput: UInt64
    private let promptTokens: UInt64
    private let admittedStopTokens: Set<UInt32>
    private let maximumEventJSONBytes: Int
    private var storage: [UInt8]
    private(set) var summary = GenerationResponseSummary(nextSequence: 0, emittedTokens: 0,
        decodedUTF8Bytes: 0, terminal: nil, sampledTotal: nil, committedTotal: nil, pendingTotal: nil,
        sessionCommittedTokens: nil, sessionPendingTokens: nil, replayRequired: nil, stopTokenID: nil)

    // Explicit caller text/event ceilings are separate from numerical admission.
    init(generationID: UUID, limits: GenerationModeLimits, promptTokens: UInt64, requestedOutput: UInt64,
         admittedStopTokens: Set<UInt32>,
         storedUTF8Bytes: Int, eventJSONBytes: Int) throws {
        let (positions, contextOverflow) = promptTokens.addingReportingOverflow(requestedOutput)
        guard requestedOutput > 0, requestedOutput <= limits.maximumOutputTokens,
              promptTokens <= limits.maximumPromptTokens, !contextOverflow, positions <= limits.totalContextTokens,
              admittedStopTokens.count <= 8,
              (1...Self.maximumStoredUTF8Bytes).contains(storedUTF8Bytes),
              UInt64(storedUTF8Bytes) <= limits.maximumDecodedBytes else {
            throw GenerationContractError.invalid("Invalid response storage admission")
        }
        let batchBytes = min(limits.events.maximumBatchUTF8Bytes, UInt64(storedUTF8Bytes))
        let (escaped, multiplyOverflow) = batchBytes.multipliedReportingOverflow(by: 6)
        let (encoded, addOverflow) = escaped.addingReportingOverflow(UInt64(Self.eventEnvelopeBytes))
        guard !multiplyOverflow, !addOverflow, encoded <= UInt64(Int.max), eventJSONBytes == Int(encoded) else {
            throw GenerationContractError.invalid("Invalid event JSON admission")
        }
        self.generationID = generationID; self.limits = limits; self.requestedOutput = requestedOutput
        self.promptTokens = promptTokens; self.admittedStopTokens = admittedStopTokens
        maximumEventJSONBytes = eventJSONBytes
        storage = [UInt8](repeating: 0, count: storedUTF8Bytes)
    }

    // Explicit cold materialization; callers must admit this additional String.
    func text() -> String { String(decoding: storage.prefix(summary.decodedUTF8Bytes), as: UTF8.self) }
    var allocatedUTF8Capacity: Int { storage.capacity }

    func consume(_ data: Data) throws {
        guard summary.terminal == nil else { throw GenerationContractError.invalid("Response already terminal") }
        let maximumString = Int(min(limits.events.maximumBatchUTF8Bytes, UInt64(storage.count)))
        let object = try GenerationJSON.object(data, maximumBytes: maximumEventJSONBytes, maximumStringBytes: maximumString,
            maximumTotalStringBytes: maximumString + Self.eventEnvelopeBytes)
        guard let id = UUID(uuidString: try GenerationJSON.string(object["generation_id"], maximumBytes: 36)), id == generationID,
              try GenerationJSON.integer(object["sequence"]) == summary.nextSequence else {
            throw GenerationContractError.invalid("Stale or out-of-order response event")
        }
        let kind = try GenerationJSON.string(object["kind"], maximumBytes: 8)
        let tokens = try GenerationJSON.integer(object["token_count"])
        guard let delta = object["utf8_delta"] as? String else { throw GenerationContractError.invalid("Missing UTF-8 delta") }
        let bytes = delta.utf8.count
        let (emitted, tokenOverflow) = summary.emittedTokens.addingReportingOverflow(tokens)
        let (next, sequenceOverflow) = summary.nextSequence.addingReportingOverflow(1)
        guard !tokenOverflow, !sequenceOverflow, tokens <= limits.events.maximumBatchTokens, emitted <= requestedOutput,
              UInt64(bytes) <= limits.events.maximumBatchUTF8Bytes, bytes <= storage.count - summary.decodedUTF8Bytes else {
            throw GenerationContractError.invalid("Response batch exceeds admitted bounds")
        }
        let common: Set<String> = ["generation_id", "sequence", "kind", "token_count", "utf8_delta"]
        var reason: GenerationStopReason?
        var sampled: UInt64?, committed: UInt64?, pending: UInt64?
        var sessionCommitted: UInt64?, sessionPending: UInt64?, replay: Bool?, stopID: UInt32?
        switch kind {
        case "text":
            try GenerationJSON.keys(object, common)
            guard tokens > 0 || !delta.isEmpty else { throw GenerationContractError.invalid("Empty text batch") }
        case "terminal":
            guard try GenerationJSON.boolean(object["terminal_flush"]),
                  let stop = GenerationStopReason(rawValue: try GenerationJSON.string(object["stop_reason"], maximumBytes: 16)) else {
                throw GenerationContractError.invalid("Missing terminal flush or stop reason")
            }
            var terminalKeys = common.union(["stop_reason", "sampled_total", "emitted_total", "committed_total", "pending_total",
                "session_committed_tokens", "session_pending_tokens", "replay_required", "terminal_flush"])
            if stop == .stopToken { terminalKeys.insert("stop_token_id") }
            try GenerationJSON.keys(object, terminalKeys)
            let actualSampled = try GenerationJSON.integer(object["sampled_total"])
            let actualEmitted = try GenerationJSON.integer(object["emitted_total"])
            let actualCommitted = try GenerationJSON.integer(object["committed_total"])
            let actualPending = try GenerationJSON.integer(object["pending_total"])
            let fullCommitted = try GenerationJSON.integer(object["session_committed_tokens"])
            let fullPending = try GenerationJSON.integer(object["session_pending_tokens"])
            let requiresReplay = try GenerationJSON.boolean(object["replay_required"])
            let (history, historyOverflow) = actualCommitted.addingReportingOverflow(actualPending)
            let (sessionHistory, sessionOverflow) = fullCommitted.addingReportingOverflow(fullPending)
            let (expectedHistory, promptOverflow) = promptTokens.addingReportingOverflow(actualSampled)
            let (expectedCommitted, committedOverflow) = promptTokens.addingReportingOverflow(actualCommitted)
            let (stoppedSampled, stopOverflow) = actualEmitted.addingReportingOverflow(1)
            guard !historyOverflow, actualPending <= 1, history == actualSampled,
                  !sessionOverflow, !promptOverflow, !committedOverflow,
                  sessionHistory <= limits.totalContextTokens, sessionHistory <= expectedHistory,
                  requiresReplay || (sessionHistory == expectedHistory
                    && (actualSampled == 0 || (fullCommitted == expectedCommitted && fullPending == actualPending))),
                  actualSampled <= requestedOutput, actualEmitted == emitted, actualEmitted <= actualSampled,
                  stop != .outputLimit || (actualEmitted == requestedOutput && actualSampled == actualEmitted),
                  stop != .stopToken || (!stopOverflow && actualSampled == stoppedSampled) else {
                throw GenerationContractError.invalid("Inconsistent terminal counters")
            }
            if stop == .stopToken {
                let token = try GenerationJSON.integer(object["stop_token_id"])
                guard let value = UInt32(exactly: token), admittedStopTokens.contains(value) else {
                    throw GenerationContractError.invalid("Unadmitted hidden stop token")
                }
                stopID = value
            }
            reason = stop; sampled = actualSampled; committed = actualCommitted; pending = actualPending
            sessionCommitted = fullCommitted; sessionPending = fullPending; replay = requiresReplay
        default: throw GenerationContractError.invalid("Unknown response event kind")
        }
        // Everything that can fail precedes publication. A rejected event leaves
        // the valid prefix, cursor and counters untouched for retry/replay.
        var offset = summary.decodedUTF8Bytes
        for byte in delta.utf8 { storage[offset] = byte; offset += 1 }
        summary = GenerationResponseSummary(nextSequence: next, emittedTokens: emitted,
            decodedUTF8Bytes: offset, terminal: reason, sampledTotal: sampled, committedTotal: committed, pendingTotal: pending,
            sessionCommittedTokens: sessionCommitted, sessionPendingTokens: sessionPending, replayRequired: replay, stopTokenID: stopID)
    }
}

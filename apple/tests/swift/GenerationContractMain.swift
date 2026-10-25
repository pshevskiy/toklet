import Foundation

private func check(_ value: @autoclosure () -> Bool, _ message: String) throws {
    guard value() else { throw GenerationContractError.invalid("Consumer check failed: " + message) }
}
private func rejects(_ label: String, _ body: () throws -> Void) throws {
    do { try body() } catch { return }
    throw GenerationContractError.invalid("Expected rejection: " + label)
}
private func json(_ value: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
}

@main
struct GenerationContractMain {
    static let id = UUID(uuidString: "8E93D1A2-FE57-4E7B-BDC0-5C66C35A3F80")!
    static func event(_ sequence: UInt64 = 0, tokens: UInt64 = 1, text: String = "a") -> [String: Any] {
        ["generation_id": id.uuidString, "sequence": sequence, "kind": "text", "token_count": tokens, "utf8_delta": text]
    }
    static func terminal(_ sequence: UInt64 = 0, reason: String = "cancelled", sampled: UInt64 = 0,
                         emitted: UInt64 = 0, committed: UInt64 = 0, pending: UInt64 = 0) -> [String: Any] {
        var value = event(sequence, tokens: 0, text: "")
        value["kind"] = "terminal"; value["stop_reason"] = reason
        value["sampled_total"] = sampled; value["emitted_total"] = emitted
        value["committed_total"] = committed; value["pending_total"] = pending
        value["session_committed_tokens"] = 9 + committed; value["session_pending_tokens"] = pending
        value["replay_required"] = false; value["terminal_flush"] = true
        return value
    }
    static func consumer(_ limits: GenerationModeLimits, output: UInt64 = 3, bytes: Int = 64) throws -> GenerationResponse {
        try GenerationResponse(generationID: id, limits: limits, promptTokens: 9, requestedOutput: output,
            admittedStopTokens: [1, 50, 106], storedUTF8Bytes: bytes,
            eventJSONBytes: 6 * min(Int(limits.events.maximumBatchUTF8Bytes), bytes) + GenerationResponse.eventEnvelopeBytes)
    }
    static func capabilityChecks(_ source: Data) throws -> GenerationModeLimits {
        let contract = try GenerationCapabilities.decode(source)
        try check(contract.identity.repository == "synthetic/consumer-fixture", "pinned fixture identity")
        let limits = try contract.validateRequest(mode: .chat, promptTokens: 131072, outputTokens: 131072, inputBytes: 8388608)
        try check(limits.maximumPromptTokens + limits.maximumOutputTokens > limits.totalContextTokens,
            "independent maxima can exceed total together")
        try rejects("actual prompt plus output") { _ = try contract.validateRequest(mode: .chat, promptTokens: 131073, outputTokens: 131072, inputBytes: 0) }
        try rejects("unavailable raw") { _ = try contract.validateRequest(mode: .raw, promptTokens: 1, outputTokens: 1, inputBytes: 1) }
        try rejects("zero output") { _ = try contract.validateRequest(mode: .chat, promptTokens: 0, outputTokens: 0, inputBytes: 0) }
        try rejects("prompt arithmetic overflow") { _ = try contract.validateRequest(mode: .chat, promptTokens: .max, outputTokens: 1, inputBytes: 0) }
        try rejects("input bytes") { _ = try contract.validateRequest(mode: .chat, promptTokens: 1, outputTokens: 1, inputBytes: 8388609) }
        var root = try GenerationJSON.object(source, maximumBytes: 65536)
        func changed(_ label: String, _ change: (inout [String: Any]) -> Void) throws {
            var copy = root; change(&copy)
            try rejects(label) { _ = try GenerationCapabilities.decode(json(copy)) }
        }
        try changed("unknown version") { $0["contract_version"] = 2 }
        try changed("Boolean version") { $0["contract_version"] = true }
        try changed("fraction version") { $0["contract_version"] = 1.5 }
        try changed("unknown top field") { $0["extension"] = true }
        try changed("missing top field") { $0.removeValue(forKey: "model_identity") }
        let modes = root["modes"] as! [[String: Any]]
        try changed("duplicate mode") { $0["modes"] = [modes[1], modes[1]] }
        try changed("implicit missing mode") { $0["modes"] = [modes[1]] }
        for bad: Any in [true, 0, -1, 1.5, "131072", NSNull()] {
            try changed("invalid positive count") { value in
                var entries = modes; entries[1]["max_output_tokens"] = bad; value["modes"] = entries
            }
        }
        try changed("unavailable with limits") { value in
            var entries = modes; entries[0]["max_output_tokens"] = 1; value["modes"] = entries
        }
        try changed("unavailable without meaningful reason") { value in
            var entries = modes; entries[0]["reason"] = " \n\t"; value["modes"] = entries
        }
        try changed("runnable with reason") { value in
            var entries = modes; entries[1]["reason"] = "ignored"; value["modes"] = entries
        }
        for field in ["revision", "config_sha256", "tokenizer_sha256"] {
            try changed("unpinned identity") { value in
                var identity = value["model_identity"] as! [String: Any]; identity[field] = "main"; value["model_identity"] = identity
            }
        }
        let text = String(decoding: source, as: UTF8.self)
        for duplicate in ["\"contract_version\":1,\"contract_version\":1,", "\"contract_version\":1,\"\\u0063ontract_version\":1,"] {
            let altered = "{" + duplicate + text.dropFirst()
            try rejects("duplicate or escaped duplicate key") { _ = try GenerationCapabilities.decode(Data(altered.utf8)) }
        }
        let overflow = text.replacingOccurrences(of: "131072", with: "18446744073709551616")
        try rejects("UInt64 overflow") { _ = try GenerationCapabilities.decode(Data(overflow.utf8)) }
        for fractional in ["1.0", "1e0", "1.00000000000000000000001"] {
            let invalid = text.replacingOccurrences(of: "\"contract_version\": 1", with: "\"contract_version\": " + fractional)
            try rejects("integer lexeme cannot lose fractional precision") { _ = try GenerationCapabilities.decode(Data(invalid.utf8)) }
        }
        for scalar in ["1.00000000000000000000001", "9007199254740992.1", "1e0", "-1"] {
            try rejects("pre-Foundation unsigned numeric grammar") {
                _ = try GenerationJSON.object(Data(("{\"count\":" + scalar + "}").utf8), maximumBytes: 256)
            }
        }
        let largest = try GenerationJSON.object(Data("{\"count\":18446744073709551615}".utf8), maximumBytes: 256)
        let largestValue = try GenerationJSON.integer(largest["count"])
        try check(largestValue == UInt64.max, "UInt64 maximum remains exact on the actual Apple parser")
        try rejects("metadata size") { _ = try GenerationCapabilities.decode(Data(repeating: 32, count: 65537)) }
        try rejects("UTF16 is not the UTF8 wire contract") {
            _ = try GenerationCapabilities.decode(String(decoding: source, as: UTF8.self).data(using: .utf16)!)
        }
        try rejects("nesting limit before Foundation parse") {
            _ = try GenerationJSON.object(Data((String(repeating: "[", count: 100) + "0" + String(repeating: "]", count: 100)).utf8), maximumBytes: 1024)
        }
        try rejects("object cardinality before Foundation parse") {
            _ = try GenerationJSON.object(json(Dictionary(uniqueKeysWithValues: (0...64).map { ("key\($0)", 0) })), maximumBytes: 4096)
        }
        let unicode = Data("{\"value\":\"\\uD83C\\uDF0D\\n\\u0000\\u00E9\"}".utf8)
        let unicodeObject = try GenerationJSON.object(unicode, maximumBytes: 256)
        try check(unicodeObject["value"] as? String == "🌍\n\u{0000}é", "escaped scalar byte-count preflight")
        try rejects("decoded string before Foundation parse") {
            _ = try GenerationJSON.object(json(["value": String(repeating: "é", count: 33)]), maximumBytes: 1024, maximumStringBytes: 64)
        }
        try rejects("aggregate strings before Foundation parse") {
            _ = try GenerationJSON.object(json(["first": String(repeating: "x", count: 64), "second": String(repeating: "y", count: 64)]),
                maximumBytes: 1024, maximumStringBytes: 64, maximumTotalStringBytes: 100)
        }
        root.removeAll()
        return limits
    }
    static func responseChecks(_ limits: GenerationModeLimits, fixture: Data) throws {
        let events = try JSONSerialization.jsonObject(with: fixture) as! [[String: Any]]
        let value = try consumer(limits)
        for entry in events { try value.consume(json(entry)) }
        try check(value.text() == "Hi 🌍\n" && value.summary.emittedTokens == 3 && value.summary.decodedUTF8Bytes == 8,
            "frozen Unicode, zero-token drain and terminal flush")
        try check(value.summary.terminal == .outputLimit && value.summary.sessionCommittedTokens == 11,
            "terminal and numerical session state kept separately")
        try rejects("terminal seals owner") { try value.consume(json(event(3))) }
        let stream = try consumer(limits)
        func atomicRejection(_ label: String, _ data: Data) throws {
            let before = stream.summary, text = stream.text()
            try rejects(label) { try stream.consume(data) }
            try check(stream.summary == before && stream.text() == text, "atomic rejection " + label)
        }
        var wrong = event(); wrong["generation_id"] = UUID().uuidString
        try atomicRejection("stale generation", json(wrong))
        try atomicRejection("future sequence", json(event(1)))
        wrong = event(); wrong["extra"] = true
        try atomicRejection("unknown event field", json(wrong))
        wrong = event(); wrong["token_count"] = true
        try atomicRejection("Boolean event count", json(wrong))
        try atomicRejection("empty text batch", json(event(tokens: 0, text: "")))
        try atomicRejection("batch token bound", json(event(tokens: 129)))
        try atomicRejection("stored text bound", json(event(text: String(repeating: "x", count: 65))))
        try atomicRejection("invalid JSON UTF-8", Data([123, 255, 125]))
        try atomicRejection("invalid JSON surrogate", Data("{\"utf8_delta\":\"\\ud800\"}".utf8))
        let duplicate = String(decoding: try json(event()), as: UTF8.self)
            .replacingOccurrences(of: "\"sequence\":0", with: "\"sequence\":0,\"\\u0073equence\":0")
        try atomicRejection("escaped duplicate event field", Data(duplicate.utf8))
        try stream.consume(json(event()))
        try atomicRejection("duplicate sequence", json(event()))
        var end = terminal(1, sampled: 1, emitted: 1, pending: 1)
        end["terminal_flush"] = false
        try atomicRejection("missing terminal flush", json(end))
        end["terminal_flush"] = true; end["sampled_total"] = UInt64.max; end["committed_total"] = UInt64.max
        try atomicRejection("terminal sum overflow", json(end))
        end = terminal(1, sampled: 1, emitted: 1, pending: 1); end["session_pending_tokens"] = UInt64.max
        try atomicRejection("session sum overflow", json(end))
        end = terminal(1, reason: "output_limit", sampled: 1, emitted: 1, pending: 1)
        try atomicRejection("early stop is not requested limit", json(end))
        end = terminal(1, sampled: 1, emitted: 1, pending: 1); end["committed_total"] = 9
        try atomicRejection("full-session count is not request-relative", json(end))
        try stream.consume(json(terminal(1, sampled: 1, emitted: 1, pending: 1)))
        for reason in ["failed", "cancelled"] {
            let interrupted = try consumer(limits)
            try interrupted.consume(json(terminal(reason: reason, sampled: 3, emitted: 0, committed: 2, pending: 1)))
            try check(interrupted.summary.sampledTotal == 3 && interrupted.summary.emittedTokens == 0,
                "unemitted sampled tokens on failure/cancel")
        }
        let prefill = try consumer(limits)
        end = terminal(); end["session_committed_tokens"] = 3; end["replay_required"] = true
        try prefill.consume(json(end))
        try check(prefill.summary.replayRequired == true && prefill.summary.sessionCommittedTokens == 3, "explicit partial-prefix replay")
        for sampleBeforePrompt in [false, true] {
            let rejected = try consumer(limits)
            let before = rejected.summary, originalText = rejected.text()
            var invalid = terminal(sampled: sampleBeforePrompt ? 1 : 0, pending: sampleBeforePrompt ? 1 : 0)
            invalid["replay_required"] = !sampleBeforePrompt
            invalid["session_committed_tokens"] = sampleBeforePrompt ? 0 : 10
            invalid["session_pending_tokens"] = sampleBeforePrompt ? 10 : 0
            try rejects(sampleBeforePrompt ? "sampling before prompt commitment" : "replay prefix exceeds request") {
                try rejected.consume(json(invalid))
            }
            try check(rejected.summary == before && rejected.text() == originalText, "counter rejection preserves text/cursor/terminal")
        }
        let pendingPrompt = try consumer(limits)
        end = terminal(); end["session_committed_tokens"] = 3; end["session_pending_tokens"] = 6
        try pendingPrompt.consume(json(end))
        try check(pendingPrompt.summary.sampledTotal == 0 && pendingPrompt.summary.sessionPendingTokens == 6,
            "unsampled partial prefill may retain pending prompt tokens")
        let hidden = try consumer(limits)
        try hidden.consume(json(event()))
        end = terminal(1, reason: "stop_token", sampled: 2, emitted: 1, committed: 1, pending: 1)
        end["stop_token_id"] = 51
        try rejects("unadmitted stop") { try hidden.consume(json(end)) }
        end["stop_token_id"] = 106
        try hidden.consume(json(end))
        try check(hidden.summary.stopTokenID == 106 && hidden.text() == "a", "stop is hidden and bound to admitted IDs")
        try rejects("decoded host ceiling") { _ = try consumer(limits, bytes: GenerationResponse.maximumStoredUTF8Bytes + 1) }
        try rejects("event envelope admission") {
            _ = try GenerationResponse(generationID: id, limits: limits, promptTokens: 9, requestedOutput: 3,
                admittedStopTokens: [], storedUTF8Bytes: 64, eventJSONBytes: 1)
        }
    }
    static func longSynthetic(_ limits: GenerationModeLimits) throws -> [String: Any] {
        let stream = try consumer(limits, output: 131072, bytes: GenerationResponse.maximumStoredUTF8Bytes)
        let capacity = stream.allocatedUTF8Capacity
        let delta = String(repeating: "\u{0000}", count: 4096)
        for sequence in 0..<1024 {
            let encoded = try json(event(UInt64(sequence), tokens: 128, text: delta))
            try stream.consume(encoded)
            try check(stream.allocatedUTF8Capacity == capacity, "no response-storage growth")
        }
        let end = terminal(1024, reason: "output_limit", sampled: 131072, emitted: 131072, committed: 131071, pending: 1)
        try stream.consume(json(end))
        let text = stream.text()
        try check(text.utf8.count == 4194304 && text.utf8.allSatisfy { $0 == 0 }, "exact escaped long synthetic text")
        try check(stream.summary.emittedTokens == 131072 && stream.summary.nextSequence == 1025, "synthetic ordered counters")
        return ["synthetic_emitted_count": stream.summary.emittedTokens, "synthetic_decoded_bytes": stream.summary.decodedUTF8Bytes,
                "owned_utf8_capacity": capacity, "storage_capacity_unchanged": true]
    }
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw GenerationContractError.invalid("Fixture directory required") }
        let folder = URL(fileURLWithPath: CommandLine.arguments[1])
        let limits = try capabilityChecks(Data(contentsOf: folder.appendingPathComponent("capabilities.json")))
        try responseChecks(limits, fixture: Data(contentsOf: folder.appendingPathComponent("events.json")))
        var result = try longSynthetic(limits)
        result["passed"] = true; result["synthetic_only"] = true
        result["real_generation_tested"] = false; result["physical_device_qualified"] = false
        print(String(decoding: try json(result), as: UTF8.self))
    }
}

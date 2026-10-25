import Foundation
import CoreFoundation

// Proposed v1 wire consumer only. No existing profile, ABI or UI uses this yet.
enum GenerationContractError: Error, Equatable {
    case invalid(String)
}

enum GenerationMode: String, CaseIterable, Sendable { case raw, chat }

struct GenerationIdentity: Equatable, Sendable {
    let repository, revision, configSHA256, tokenizerSHA256: String
}

struct GenerationEventLimits: Equatable, Sendable {
    let maximumBatchTokens, maximumBatchUTF8Bytes, maximumFallbackTokens: UInt64
}

struct GenerationModeLimits: Equatable, Sendable {
    let totalContextTokens, maximumPromptTokens, maximumOutputTokens: UInt64
    let maximumInputBytes, maximumDecodedBytes, tokenizerAllowanceBytes: UInt64
    let events: GenerationEventLimits
}

enum GenerationModeCapability: Equatable, Sendable {
    case unavailable(reason: String)
    case runnable(GenerationModeLimits)
}

struct GenerationCapabilities: Equatable, Sendable {
    static let maximumJSONBytes = 64 * 1024
    let identity: GenerationIdentity
    let modes: [GenerationMode: GenerationModeCapability]

    static func decode(_ data: Data) throws -> Self {
        let object = try GenerationJSON.object(data, maximumBytes: maximumJSONBytes)
        try GenerationJSON.keys(object, ["contract_version", "model_identity", "modes"])
        guard try GenerationJSON.integer(object["contract_version"]) == 1 else {
            throw GenerationContractError.invalid("Unsupported capability version")
        }
        let model = try GenerationJSON.dictionary(object["model_identity"])
        try GenerationJSON.keys(model, ["repository", "revision", "config_sha256", "tokenizer_sha256"])
        let repository = try GenerationJSON.string(model["repository"], maximumBytes: 512)
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ part in
            !part.isEmpty && part != "." && part != ".." && part.utf8.allSatisfy {
                (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 95].contains($0)
            }
        }) else { throw GenerationContractError.invalid("Invalid repository identity") }
        let identity = GenerationIdentity(repository: repository,
            revision: try GenerationJSON.hex(model["revision"], count: 40),
            configSHA256: try GenerationJSON.hex(model["config_sha256"], count: 64),
            tokenizerSHA256: try GenerationJSON.hex(model["tokenizer_sha256"], count: 64))
        guard let entries = object["modes"] as? [Any], entries.count == 2 else {
            throw GenerationContractError.invalid("Both modes must be explicit")
        }
        var modes: [GenerationMode: GenerationModeCapability] = [:]
        for entry in entries {
            let value = try GenerationJSON.dictionary(entry)
            guard let mode = GenerationMode(rawValue: try GenerationJSON.string(value["mode"], maximumBytes: 8)), modes[mode] == nil else {
                throw GenerationContractError.invalid("Unknown or duplicate generation mode")
            }
            if try GenerationJSON.boolean(value["runnable"]) {
                try GenerationJSON.keys(value, ["mode", "runnable", "total_context_tokens", "max_prompt_tokens", "max_output_tokens",
                    "max_input_bytes", "max_decoded_bytes", "tokenizer_allowance_bytes", "event_limits"])
                let events = try GenerationJSON.dictionary(value["event_limits"])
                try GenerationJSON.keys(events, ["max_batch_tokens", "max_batch_utf8_bytes", "max_fallback_tokens"])
                let limits = GenerationModeLimits(
                    totalContextTokens: try GenerationJSON.positive(value["total_context_tokens"]),
                    maximumPromptTokens: try GenerationJSON.positive(value["max_prompt_tokens"]),
                    maximumOutputTokens: try GenerationJSON.positive(value["max_output_tokens"]),
                    maximumInputBytes: try GenerationJSON.positive(value["max_input_bytes"]),
                    maximumDecodedBytes: try GenerationJSON.positive(value["max_decoded_bytes"]),
                    tokenizerAllowanceBytes: try GenerationJSON.positive(value["tokenizer_allowance_bytes"]),
                    events: GenerationEventLimits(maximumBatchTokens: try GenerationJSON.positive(events["max_batch_tokens"]),
                        maximumBatchUTF8Bytes: try GenerationJSON.positive(events["max_batch_utf8_bytes"]),
                        maximumFallbackTokens: try GenerationJSON.positive(events["max_fallback_tokens"])))
                // The two maxima need not fit together; each actual request must.
                guard limits.maximumPromptTokens <= limits.totalContextTokens,
                      limits.maximumOutputTokens <= limits.totalContextTokens else {
                    throw GenerationContractError.invalid("Mode maxima exceed context")
                }
                modes[mode] = .runnable(limits)
            } else {
                try GenerationJSON.keys(value, ["mode", "runnable", "reason"])
                let reason = try GenerationJSON.string(value["reason"], maximumBytes: 2048)
                guard !reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw GenerationContractError.invalid("Unavailable mode needs a reason")
                }
                modes[mode] = .unavailable(reason: reason)
            }
        }
        return Self(identity: identity, modes: modes)
    }

    // Metadata validation, not native graph/resource admission or a reservation.
    func validateRequest(mode: GenerationMode, promptTokens: UInt64, outputTokens: UInt64, inputBytes: UInt64) throws -> GenerationModeLimits {
        guard case .runnable(let limits) = modes[mode] else {
            throw GenerationContractError.invalid("Generation mode is unavailable")
        }
        let (positions, overflow) = promptTokens.addingReportingOverflow(outputTokens)
        guard outputTokens > 0, !overflow, promptTokens <= limits.maximumPromptTokens,
              outputTokens <= limits.maximumOutputTokens, positions <= limits.totalContextTokens,
              inputBytes <= limits.maximumInputBytes else {
            throw GenerationContractError.invalid("Request exceeds mode limits")
        }
        return limits
    }
}

// A bounded Foundation parser with explicit duplicate-key detection. Foundation
// alone accepts repeated keys. Only this cold metadata/event boundary uses Any.
enum GenerationJSON {
    static func object(_ data: Data, maximumBytes: Int, maximumStringBytes: Int = 2048,
                       maximumTotalStringBytes: Int? = nil) throws -> [String: Any] {
        guard maximumBytes > 0, !data.isEmpty, data.count <= maximumBytes else {
            throw GenerationContractError.invalid("JSON byte limit")
        }
        try uniqueKeys(data, maximumStringBytes: max(64, maximumStringBytes), maximumTotalStringBytes: maximumTotalStringBytes ?? maximumBytes)
        return try dictionary(JSONSerialization.jsonObject(with: data))
    }
    static func dictionary(_ value: Any?) throws -> [String: Any] {
        guard let object = value as? [String: Any] else { throw GenerationContractError.invalid("Expected object") }
        return object
    }
    static func keys(_ value: [String: Any], _ expected: Set<String>) throws {
        guard Set(value.keys) == expected else { throw GenerationContractError.invalid("Missing or unknown fields") }
    }
    static func boolean(_ value: Any?) throws -> Bool {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw GenerationContractError.invalid("Expected Boolean")
        }
        return number.boolValue
    }
    static func integer(_ value: Any?) throws -> UInt64 {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              let result = UInt64(number.stringValue), number.compare(NSNumber(value: result)) == .orderedSame else {
            throw GenerationContractError.invalid("Expected exact unsigned integer")
        }
        return result
    }
    static func positive(_ value: Any?) throws -> UInt64 {
        let number = try integer(value)
        guard number > 0 else { throw GenerationContractError.invalid("Expected positive limit") }
        return number
    }
    static func string(_ value: Any?, maximumBytes: Int) throws -> String {
        guard let text = value as? String, !text.isEmpty, text.utf8.count <= maximumBytes else {
            throw GenerationContractError.invalid("Invalid bounded string")
        }
        return text
    }
    static func hex(_ value: Any?, count: Int) throws -> String {
        let text = try string(value, maximumBytes: count)
        guard text.utf8.count == count, text.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw GenerationContractError.invalid("Identity is not a pinned hexadecimal digest")
        }
        return text
    }
    private enum Frame { case object(Set<String>), array }
    private static func uniqueKeys(_ data: Data, maximumStringBytes: Int, maximumTotalStringBytes: Int) throws {
        var stack: [Frame] = []
        var containers = 0, separators = 0, keys = 0, stringBytes = 0
        var index = data.startIndex
        while index < data.endIndex {
            switch data[index] {
            case 123, 91:
                containers += 1
                guard stack.count < 8, containers <= 16 else { throw GenerationContractError.invalid("JSON structure limit") }
                stack.append(data[index] == 123 ? .object([]) : .array)
            case 125, 93:
                guard !stack.isEmpty else { throw GenerationContractError.invalid("Unbalanced JSON") }
                stack.removeLast()
            case 34:
                let start = index
                let (end, bytes) = try stringEnd(data, start: index, maximumBytes: maximumStringBytes)
                stringBytes += bytes
                guard stringBytes <= maximumTotalStringBytes else { throw GenerationContractError.invalid("JSON total string limit") }
                index = end
                var next = index + 1
                while next < data.endIndex && [9, 10, 13, 32].contains(data[next]) { next += 1 }
                if next < data.endIndex && data[next] == 58, case .object(var names)? = stack.last {
                    keys += 1
                    guard keys <= 64, bytes <= 64 else { throw GenerationContractError.invalid("JSON key limit") }
                    let name = try JSONDecoder().decode(String.self, from: data.subdata(in: start..<(index + 1)))
                    guard names.insert(name).inserted else { throw GenerationContractError.invalid("Duplicate JSON key") }
                    stack[stack.count - 1] = .object(names)
                }
            case 44:
                separators += 1
                guard separators <= 64 else { throw GenerationContractError.invalid("JSON item limit") }
            case 9, 10, 13, 32, 58: break
            default:
                guard [45, 102, 110, 116].contains(data[index]) || (48...57).contains(data[index]) else {
                    throw GenerationContractError.invalid("Invalid JSON scalar or encoding")
                }
                let start = index
                while index < data.endIndex && ![9, 10, 13, 32, 44, 93, 125].contains(data[index]) { index += 1 }
                guard index - start <= 32 else { throw GenerationContractError.invalid("JSON scalar limit") }
                if data[start] == 45 || (48...57).contains(data[start]) {
                    let number = data[start..<index]
                    guard number.allSatisfy({ (48...57).contains($0) }),
                          UInt64(String(decoding: number, as: UTF8.self)) != nil else {
                        throw GenerationContractError.invalid("Expected unsigned integer lexeme")
                    }
                }
                index -= 1
            }
            index += 1
        }
    }
    // Count decoded UTF-8 before Foundation creates any large String/object.
    // JSONSerialization still validates complete grammar and raw UTF-8.
    private static func stringEnd(_ data: Data, start: Int, maximumBytes: Int) throws -> (Int, Int) {
        func hex(_ start: Int) throws -> UInt32 {
            guard start + 4 <= data.endIndex else { throw GenerationContractError.invalid("Incomplete Unicode escape") }
            var value: UInt32 = 0
            for byte in data[start..<(start + 4)] {
                let digit: UInt8
                switch byte {
                case 48...57: digit = byte - 48
                case 65...70: digit = byte - 55
                case 97...102: digit = byte - 87
                default: throw GenerationContractError.invalid("Invalid Unicode escape")
                }
                value = value * 16 + UInt32(digit)
            }
            return value
        }
        var index = start + 1, bytes = 0
        while index < data.endIndex && data[index] != 34 {
            if data[index] == 92 {
                guard index + 1 < data.endIndex else { throw GenerationContractError.invalid("Incomplete escape") }
                switch data[index + 1] {
                case 34, 47, 92, 98, 102, 110, 114, 116: bytes += 1; index += 2
                case 117:
                    let scalar = try hex(index + 2)
                    index += 6
                    if (0xd800...0xdbff).contains(scalar) {
                        guard index + 6 <= data.endIndex, data[index] == 92, data[index + 1] == 117,
                              (0xdc00...0xdfff).contains(try hex(index + 2)) else {
                            throw GenerationContractError.invalid("Unpaired Unicode surrogate")
                        }
                        bytes += 4; index += 6
                    } else {
                        guard !(0xdc00...0xdfff).contains(scalar) else { throw GenerationContractError.invalid("Unpaired Unicode surrogate") }
                        bytes += scalar <= 0x7f ? 1 : (scalar <= 0x7ff ? 2 : 3)
                    }
                default: throw GenerationContractError.invalid("Invalid JSON escape")
                }
            } else {
                guard data[index] >= 32 else { throw GenerationContractError.invalid("Unescaped JSON control") }
                bytes += 1; index += 1
            }
            guard bytes <= maximumBytes else { throw GenerationContractError.invalid("JSON decoded string limit") }
        }
        guard index < data.endIndex else { throw GenerationContractError.invalid("Incomplete JSON string") }
        return (index, bytes)
    }
}

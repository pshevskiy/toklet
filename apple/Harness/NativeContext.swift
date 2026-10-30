import Foundation
import CToklet

struct NativeContextFailure: Error, LocalizedError, Equatable {
    let message: String
    var errorDescription: String? { message }
    init(_ message: String) { self.message = message }
}

// Fixed-width execution facts, separate from the unused JSON wire proposal.
// A declaration is loaded metadata; neither it nor allocation grants qualification.
struct NativeContextFacts: Equatable, Sendable {
    let modelInstance, sessionInstance: UInt64
    let backend: UInt32
    let declaredTotal, declaredInput, declaredOutput: UInt64?
    let engineGlobalCeiling, loadedLimit, capacity, offset: UInt64
    let history, committed, pending, numericalAdmission: UInt64

    init(_ value: TokletContextInfo) throws {
        let known = UInt32(TOKLET_CONTEXT_DECLARED_TOTAL_KNOWN | TOKLET_CONTEXT_DECLARED_INPUT_KNOWN | TOKLET_CONTEXT_DECLARED_OUTPUT_KNOWN)
        guard value.abi_version == TOKLET_ABI_VERSION,
              value.struct_size == UInt32(MemoryLayout<TokletContextInfo>.size),
              value.known_fields & ~known == 0, value.reserved.0 == 0, value.reserved.1 == 0,
              value.backend == TOKLET_CONTEXT_CPU || value.backend == TOKLET_CONTEXT_METAL,
              value.context_qualification == TOKLET_CONTEXT_NOT_EVALUATED,
              value.whole_app_device_fit == TOKLET_CONTEXT_NOT_EVALUATED,
              value.model_instance > 0, value.session_instance > 0,
              value.model_instance != value.session_instance,
              value.engine_global_position_ceiling > 0, value.loaded_position_limit > 0,
              value.loaded_position_limit <= value.engine_global_position_ceiling,
              value.session_capacity > 0, value.session_capacity <= value.loaded_position_limit,
              value.numerical_and_bridge_admitted_bytes > 0 else {
            throw NativeContextFailure("Invalid native context record")
        }
        func declaration(_ bit: UInt32, _ number: UInt64) throws -> UInt64? {
            if value.known_fields & bit == 0 {
                guard number == 0 else { throw NativeContextFailure("Unknown model limit contains a value") }
                return nil
            }
            guard number > 0 else { throw NativeContextFailure("Known model limit is empty") }
            return number
        }
        let total = try declaration(UInt32(TOKLET_CONTEXT_DECLARED_TOTAL_KNOWN),value.declared_total_positions)
        let input = try declaration(UInt32(TOKLET_CONTEXT_DECLARED_INPUT_KNOWN),value.declared_input_tokens)
        let output = try declaration(UInt32(TOKLET_CONTEXT_DECLARED_OUTPUT_KNOWN),value.declared_output_tokens)
        let end = value.absolute_position_offset.addingReportingOverflow(value.session_capacity)
        let sampled = value.committed_tokens.addingReportingOverflow(value.pending_tokens)
        guard !end.overflow, end.partialValue <= value.loaded_position_limit,
              !sampled.overflow, sampled.partialValue == value.history_tokens,
              value.history_tokens <= value.session_capacity,
              total.map({ value.loaded_position_limit <= $0 }) ?? true else {
            throw NativeContextFailure("Inconsistent native context counts")
        }
        modelInstance = value.model_instance; sessionInstance = value.session_instance; backend = value.backend
        declaredTotal = total; declaredInput = input; declaredOutput = output
        engineGlobalCeiling = value.engine_global_position_ceiling; loadedLimit = value.loaded_position_limit
        capacity = value.session_capacity; offset = value.absolute_position_offset
        history = value.history_tokens; committed = value.committed_tokens; pending = value.pending_tokens
        numericalAdmission = value.numerical_and_bridge_admitted_bytes
    }
}

// Accepted configuration of one live session, not token-history authority or
// artifact identity. Retained separately from transient command results. A
// caller still needs idle/current NativeContextFacts for numerical admission.
struct NativeSessionConfiguration: Equatable, Sendable {
    let modelInstance, sessionInstance: UInt64
    let capacity, chunk: UInt32
    let wholeSessionBudget: UInt64
    let acceptedRequestID: UUID?

    init(context: NativeContextFacts, capacity: UInt32, chunk: UInt32,
         wholeSessionBudget: UInt64, acceptedRequestID: UUID? = nil) throws {
        guard context.modelInstance > 0, context.sessionInstance > 0,
              context.modelInstance != context.sessionInstance,
              capacity > 0, UInt64(capacity) == context.capacity,
              context.capacity <= context.loadedLimit, context.offset == 0,
              chunk > 0, chunk <= capacity,
              wholeSessionBudget > UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES),
              wholeSessionBudget <= 6 * 1024 * 1024 * 1024,
              context.numericalAdmission <= wholeSessionBudget else {
            throw NativeContextFailure("Invalid accepted session configuration")
        }
        // Ordinary CPU constructors accept chunks above the preview/transition
        // request limit 1024; reflect accepted settings without narrowing them.
        modelInstance = context.modelInstance; sessionInstance = context.sessionInstance
        self.capacity = capacity; self.chunk = chunk
        self.wholeSessionBudget = wholeSessionBudget; self.acceptedRequestID = acceptedRequestID
    }
}

// Retain only model/configuration facts across suspend. There is deliberately
// no session identity, active history or available-capacity claim in this value.
struct NativeModelContext: Equatable, Sendable {
    let modelInstance: UInt64
    let backend: UInt32
    let declaredTotal, declaredInput, declaredOutput: UInt64?
    let engineGlobalCeiling, loadedLimit, configuredCapacity, offset: UInt64
    init(_ facts: NativeContextFacts) {
        modelInstance = facts.modelInstance; backend = facts.backend
        declaredTotal = facts.declaredTotal; declaredInput = facts.declaredInput; declaredOutput = facts.declaredOutput
        engineGlobalCeiling = facts.engineGlobalCeiling; loadedLimit = facts.loadedLimit
        configuredCapacity = facts.capacity; offset = facts.offset
    }
    var sharedWindow: UInt64 { configuredCapacity }
    var windowSummary: String {
        if let total = declaredTotal { return "Model window \(total.formatted()) · selected \(sharedWindow.formatted()) tokens" }
        return "Current window \(sharedWindow.formatted()) tokens · model declaration unavailable"
    }
    var partialReason: String {
        if let total = declaredTotal, sharedWindow < total {
            return "Partial context: the selected window uses \(sharedWindow.formatted()) of the model’s \(total.formatted()) positions. History, prompt and reply share that window."
        }
        return "History, prompt and reply share the current window. Full model-window qualification is not recorded."
    }
}

enum NativeGenerationMode: Equatable, Sendable { case raw, chat }

struct NativeModeLimits: Equatable, Sendable {
    let sharedWindow, maximumPrompt, maximumOutput: UInt64
    let maximumInputTextBytes: Int
    let outputReason: String
    func validate(input: Int, output: UInt32) throws {
        guard let tokens = UInt64(exactly:input) else { throw NativeContextFailure("Invalid input token count") }
        try validateHistory(tokens:tokens,output:output)
    }
    func validateHistory(tokens: UInt64, output: UInt32) throws {
        guard tokens <= maximumPrompt, UInt64(output) <= maximumOutput else {
            throw NativeContextFailure("Request exceeds current mode limits: input up to \(maximumPrompt.formatted()), reply up to \(maximumOutput.formatted()) tokens. \(outputReason)")
        }
    }
}

// Existing app/codec/event constraints are explicit policy, never model caps.
// Changing these needs separate stream/codec/device qualification.
enum NativeGenerationPolicy {
    static let maximumRawOutput: UInt32 = 128
    static let maximumChatOutput = UInt32(TOKLET_MAX_CHAT_OUTPUT_TOKENS)
    static let maximumChatPrompt: UInt64 = 512
    static let maximumRawPrompt: UInt64 = 32768
    static let maximumConversationBytes = 64 * 1024
    static let maximumDraftBytes = 16 * 1024
    static let maximumRawTokenTextBytes = 256 * 1024
    static let maximumDecodedBytes = 64 * 1024

    static func limits(_ mode: NativeGenerationMode, model: NativeModelContext) -> NativeModeLimits {
        let chat = mode == .chat
        let prompt = min(model.sharedWindow,model.declaredInput ?? .max,chat ? maximumChatPrompt : maximumRawPrompt)
        // The engine requires a nonempty prompt/history even in Continue mode.
        let output = min(model.sharedWindow - 1,model.declaredOutput ?? .max,
                         UInt64(chat ? maximumChatOutput : maximumRawOutput))
        return NativeModeLimits(sharedWindow:model.sharedWindow,maximumPrompt:prompt,maximumOutput:output,
            maximumInputTextBytes:chat ? maximumConversationBytes : maximumRawTokenTextBytes,
            outputReason:chat ? "The current chat decoder/event limit is \(maximumChatOutput) tokens." : "Expert diagnostics currently retain at most \(maximumRawOutput) output tokens.")
    }
}

struct NativeRequestBudget: Equatable, Sendable {
    let context: NativeContextFacts
    let operation: UInt32
    let baseHistory, input, historyAfterInput, output, required, remaining: UInt64
    init(_ value: TokletRequestAdmission, expected: NativeContextFacts, operation: UInt32, input: Int, output: UInt32) throws {
        let facts = try NativeContextFacts(value.context)
        guard facts == expected, value.operation == operation, value.reserved == 0,
              operation <= 3, let count = UInt64(exactly:input), operation != 2 || count == 0 else {
            throw NativeContextFailure("Stale or invalid native request admission")
        }
        let base = operation == 0 || operation == 3 ? 0 : expected.history
        let after = base.addingReportingOverflow(count)
        let end = after.partialValue.addingReportingOverflow(UInt64(output))
        guard !after.overflow, !end.overflow, after.partialValue > 0,
              end.partialValue <= expected.capacity,
              value.base_history_tokens == base, value.input_tokens == count,
              value.history_after_input_tokens == after.partialValue,
              value.requested_output_tokens == UInt64(output), value.required_capacity == end.partialValue,
              value.remaining_tokens == expected.capacity - end.partialValue else {
            throw NativeContextFailure("Inconsistent shared-window admission")
        }
        context = facts; self.operation = operation
        baseHistory = base; self.input = count; historyAfterInput = after.partialValue
        self.output = UInt64(output); required = end.partialValue; remaining = value.remaining_tokens
    }
}

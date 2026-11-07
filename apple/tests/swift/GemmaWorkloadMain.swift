import Foundation
import CToklet

private func requireOK(_ status: UInt32, _ error: inout TokletError) throws {
    if status != TOKLET_OK {
        let text = withUnsafeBytes(of:&error.message) { bytes in
            String(decoding:bytes.prefix(while:{ $0 != 0 }),as:UTF8.self)
        }
        throw NativeContextFailure("Native status \(status): \(text)")
    }
}

// Explicit raw-token diagnostic. All handles and callbacks stay on this thread.
// Captures are bounded to 128 emitted IDs/request, unlike routine summaries.
private final class WorkloadSink {
    var ids = [UInt32]()
    var callbacks = [UInt64]()
    let cancel: OpaquePointer
    let cancelAfter: Int
    init(_ cancel: OpaquePointer, after: Int) {
        self.cancel = cancel; cancelAfter = after
        ids.reserveCapacity(128); callbacks.reserveCapacity(128)
    }
    func emit(_ event: TokletTokenEvent) -> UInt32 {
        guard ids.count < 128, event.index == ids.count else { return 1 }
        ids.append(event.token_id); callbacks.append(DispatchTime.now().uptimeNanoseconds)
        if cancelAfter > 0 && ids.count == cancelAfter { return toklet_cancel_set(cancel,1,nil) }
        return 0
    }
}
private let workloadEmit: @convention(c) (UnsafeMutableRawPointer?, TokletTokenEvent) -> UInt32 = { context, event in
    guard let context else { return 1 }
    return Unmanaged<WorkloadSink>.fromOpaque(context).takeUnretainedValue().emit(event)
}

private struct WorkloadState: Codable, Equatable {
    let history, committed, pending: UInt32
    init(_ report: TokletReport) {
        history = report.history_tokens; committed = report.evaluated_tokens; pending = report.pending_tokens
    }
}
private struct WorkloadSample: Encodable {
    let kind = "request"
    let workload, mode: String
    let trial: Int
    let warmup: Bool
    let submittedUptimeNanoseconds, completedUptimeNanoseconds: UInt64
    let callbackUptimeNanoseconds: [UInt64]
    let operation: UInt32
    let input: [UInt32]
    let maximum: UInt32
    let ids: [UInt32]
    let reason, status: UInt32
    let before, after: WorkloadState
    let native: NativeGenerationMetrics?
    let footprint: NativeFootprint
    let thermalState: Int
}

private final class GemmaWorkload {
    var model, session, chat, cancel: OpaquePointer?
    var prompt = [UInt32]()
    var options = TokletOptions(abi_version:TOKLET_ABI_VERSION,
        struct_size:UInt32(MemoryLayout<TokletOptions>.size),capacity:512,chunk:32,memory_limit_bytes:6 << 30)
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw NativeContextFailure(message) }
    }
    static func write<T: Encodable>(_ value: T) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(value); data.append(10); FileHandle.standardOutput.write(data)
    }
    init(directory: String, entry: UInt64) throws {
        // Explicit cleanup also covers a throwing initializer.
        do {
        var error = TokletError()
        let before = try NativeFootprint.read(), loadStart = Self.now()
        let path = Array(directory.utf8)
        try requireOK(path.withUnsafeBufferPointer {
            toklet_model_load_gemma_bf16_audited($0.baseAddress,$0.count,&options,&model,&error)
        },&error)
        let loadEnd = Self.now()
        try requireOK(toklet_session_create(model,&options,&session,&error),&error)
        let sessionEnd = Self.now()
        try requireOK(toklet_chat_create_complete_audited(model,0,&chat,&error),&error)
        try requireOK(toklet_cancel_create(&cancel,&error),&error)
        let ready = Self.now()
        let conversation = Array(#"[{"role":"user","content":"Name the capital of France."}]"#.utf8)
        var count = 0, tokens = [UInt32](repeating:0,count:512)
        try requireOK(conversation.withUnsafeBufferPointer { bytes in
            tokens.withUnsafeMutableBufferPointer {
                toklet_chat_encode(chat,bytes.baseAddress,bytes.count,$0.baseAddress,$0.count,&count,&error)
            }
        },&error)
        prompt = Array(tokens.prefix(count))
        try Self.require(prompt.count == 15,"Pinned prompt changed")
        var memory = TokletGemmaBf16MemoryInfo()
        try requireOK(toklet_session_gemma_bf16_memory(session,&memory,&error),&error)
        let after = try NativeFootprint.read()
        try Self.require(memory.session_admitted_bytes <= 6 << 30 && after.lifetimePeakPhysicalBytes <= 6 << 30,"Memory ceiling exceeded")
        struct Ready: Encodable {
            let kind = "ready"
            let entry, loadStart, loadEnd, sessionEnd, ready: UInt64
            let prepared: UInt64
            let before, after: NativeFootprint
            let admittedBytes, kvBytes: UInt64
            let thermalState: Int
            let prompt: [UInt32]
        }
        try Self.write(Ready(entry:entry,loadStart:loadStart,loadEnd:loadEnd,sessionEnd:sessionEnd,ready:ready,
            prepared:Self.now(),before:before,after:after,admittedBytes:memory.session_admitted_bytes,
            kvBytes:memory.kv_owned_bytes,thermalState:ProcessInfo.processInfo.thermalState.rawValue,prompt:prompt))
        } catch { close(); throw error }
    }
    func close() {
        _ = toklet_cancel_release(cancel,nil); _ = toklet_chat_release(chat,nil)
        _ = toklet_session_release(session,nil); _ = toklet_model_release(model,nil)
        cancel = nil; chat = nil; session = nil; model = nil
    }
    deinit { close() }
    func state() throws -> WorkloadState {
        var report = TokletReport(), error = TokletError()
        try requireOK(toklet_session_state(session,&report,&error),&error)
        return WorkloadState(report)
    }
    @discardableResult
    func generate(_ name: String, _ operation: UInt32, _ input: [UInt32], _ maximum: UInt32,
                  observed: Bool = true, trial: Int = 0, warmup: Bool = false,
                  preCancelled: Bool = false, cancelAfter: Int = 0,
                  stops: [UInt32] = [], publish: Bool = true, expectedStatus: UInt32 = TOKLET_OK) throws -> WorkloadSample {
        guard let cancel else { throw NativeContextFailure("Missing cancel handle") }
        var error = TokletError(), report = TokletReport(), metrics = TokletGenerationMetrics(), context = TokletContextInfo()
        try requireOK(toklet_cancel_set(cancel,preCancelled ? 1 : 0,&error),&error)
        let before = try state()
        try requireOK(toklet_session_context(session,&context,&error),&error)
        let sink = WorkloadSink(cancel,after:cancelAfter)
        let start = Self.now()
        let status = autoreleasepool { withExtendedLifetime(sink) {
            input.withUnsafeBufferPointer { input in
                stops.withUnsafeBufferPointer { stops in
                    if observed {
                        return toklet_session_generate_observed(session,operation,input.baseAddress,input.count,maximum,
                            stops.baseAddress,stops.count,cancel,workloadEmit,Unmanaged.passUnretained(sink).toOpaque(),&report,&metrics,&error)
                    }
                    return toklet_session_generate(session,operation,input.baseAddress,input.count,maximum,
                        stops.baseAddress,stops.count,cancel,workloadEmit,Unmanaged.passUnretained(sink).toOpaque(),&report,&error)
                }
            }
        } }
        let end = Self.now()
        try Self.require(status == expectedStatus,"Unexpected status in \(name): \(status)")
        let native = observed && status == TOKLET_OK ? NativeGenerationMetrics(metrics,expectedSession:context.session_instance) : nil
        try Self.require(!observed || status != TOKLET_OK || native != nil,"Invalid completed metrics")
        try Self.require(status == TOKLET_OK || metrics.known_fields == 0,"Failed request retained metrics")
        let after = try state(), footprint = try NativeFootprint.read()
        try Self.require(footprint.lifetimePeakPhysicalBytes <= 6 << 30,"Process memory ceiling exceeded")
        try Self.require(status != TOKLET_OK || (report.emitted_tokens == sink.ids.count && after.history == after.committed + after.pending),"Callback/state mismatch")
        let sample = WorkloadSample(workload:name,mode:observed ? "observed":"legacy",trial:trial,warmup:warmup,
            submittedUptimeNanoseconds:start,completedUptimeNanoseconds:end,callbackUptimeNanoseconds:sink.callbacks,
            operation:operation,input:input,maximum:maximum,ids:sink.ids,reason:report.reason,status:status,
            before:before,after:after,native:native,footprint:footprint,thermalState:ProcessInfo.processInfo.thermalState.rawValue)
        if publish { try Self.write(sample) }
        return sample
    }
    func prepare(_ name: String) throws -> (UInt32, [UInt32], UInt32) {
        switch name {
        case "reset": return (3,prompt,4)
        case "append":
            try generate("setup",3,Array(prompt.dropLast()),0,publish:false)
            return (1,[prompt.last!],4)
        case "fully_cached":
            try generate("setup",3,prompt,0,publish:false)
            return (0,prompt,4)
        case "pending":
            let seed = try generate("setup",3,prompt,1,publish:false)
            try Self.require(seed.ids == [818] && seed.after.pending == 1,"Pending seed changed")
            return (2,[],3)
        case "edited_prefix":
            try generate("setup",3,prompt,4,publish:false)
            var edited = prompt; edited[edited.count-1] = prompt[2]
            return (0,edited,4)
        default: throw NativeContextFailure("Unknown workload")
        }
    }
    func paired() throws {
        let reference = try generate("reference",3,prompt,4)
        try Self.require(reference.ids == [818,5279,529,7001],"Frozen greedy reference differs")
        var edited = prompt; edited[edited.count-1] = prompt[2]
        let editedReference = try generate("edited_reference",3,edited,4)
        for trial in 0..<13 {
            for name in ["reset","append","fully_cached","pending","edited_prefix"] {
                var pair = [WorkloadSample]()
                for observed in trial % 2 == 0 ? [false,true] : [true,false] {
                    let (operation,input,maximum) = try prepare(name)
                    let row = try generate(name,operation,input,maximum,observed:observed,trial:trial,warmup:trial<3)
                    let expected = name == "edited_prefix" ? editedReference.ids : name == "pending" ? Array(reference.ids.dropFirst()) : reference.ids
                    try Self.require(row.ids == expected,"Replay equivalence failed: \(name)")
                    pair.append(row)
                }
                try Self.require(pair[0].after == pair[1].after && pair[0].ids == pair[1].ids,"Paired work changed")
            }
        }
    }
    func controls() throws {
        let seed = try generate("reference",3,prompt,4)
        let rejected = try generate("capacity_rejected",3,Array(repeating:prompt[0],count:385),128,expectedStatus:TOKLET_INVALID)
        try Self.require(seed.after == rejected.after,"Capacity rejection mutated state")
        let preCancel = try generate("pre_cancelled",3,prompt,4,preCancelled:true)
        try Self.require(preCancel.after == seed.after && preCancel.reason == 3 && preCancel.ids.isEmpty,"Pre-cancel mutated state")
        let cancelled = try generate("cancelled",3,prompt,4,cancelAfter:1)
        try Self.require(cancelled.ids == [818] && cancelled.after.pending == 1 && cancelled.reason == 3,"Cancel prefix changed")
        let hidden = try generate("hidden_stop",3,prompt,4,stops:[818])
        try Self.require(hidden.ids.isEmpty && hidden.reason == 2 && hidden.native?.sampledTokens == 1 && hidden.after.pending == 1,"Hidden stop semantics")
        let boundary = Array((0..<384).map { prompt[$0 % prompt.count] })
        let legacy = try generate("boundary_384_128",3,boundary,128,observed:false)
        let observed = try generate("boundary_384_128",3,boundary,128)
        try Self.require(legacy.ids == observed.ids && legacy.after == observed.after && observed.after.history == 512 && observed.after.committed == 511,"Boundary parity failed")
    }
    func sustained() throws {
        for trial in 0..<3 { try generate("sustained",3,prompt,4,trial:trial,warmup:true) }
        let start = Self.now()
        var trial = 3
        repeat {
            let row = try generate("sustained",3,prompt,4,trial:trial)
            try Self.require(row.ids == [818,5279,529,7001] && row.after.history == 19 && row.after.committed == 18,"Sustained work changed")
            trial += 1
            try Self.require(trial <= 2048,"Sustained capture limit exceeded")
        } while Self.now() - start < 600_000_000_000
    }
}

@main
struct GemmaWorkloadMain {
    static func main() throws {
        let entry = GemmaWorkload.now()
        guard CommandLine.arguments.count == 3,
              ["cold","paired","controls","sustained"].contains(CommandLine.arguments[2]) else {
            throw NativeContextFailure("Expected model folder and cold/paired/controls/sustained")
        }
        let runner = try autoreleasepool { try GemmaWorkload(directory:CommandLine.arguments[1],entry:entry) }
        switch CommandLine.arguments[2] {
        case "cold":
            let row = try runner.generate("new_process_first_request",3,runner.prompt,4)
            try GemmaWorkload.require(row.ids == [818,5279,529,7001],"Cold output differs")
        case "paired": try runner.paired()
        case "controls": try runner.controls()
        default: try runner.sustained()
        }
    }
}

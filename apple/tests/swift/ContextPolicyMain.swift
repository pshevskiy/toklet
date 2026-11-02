import Foundation
import CToklet

@main
struct ContextPolicyMain {
    static func require(_ value: @autoclosure () -> Bool,_ label: String) throws {
        if !value() { throw NativeContextFailure("Context policy check: " + label) }
    }
    static func rejected(_ work: () throws -> Void,_ label: String) throws {
        do { try work() } catch { return }
        throw NativeContextFailure("Accepted invalid context policy: " + label)
    }
    static func record() -> TokletContextInfo {
        var r = TokletContextInfo()
        r.abi_version = TOKLET_ABI_VERSION; r.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        r.known_fields = UInt32(TOKLET_CONTEXT_DECLARED_TOTAL_KNOWN); r.backend = TOKLET_CONTEXT_METAL
        r.model_instance = 11; r.session_instance = 12; r.declared_total_positions = 40960
        r.engine_global_position_ceiling = 32768; r.loaded_position_limit = 512; r.session_capacity = 512
        r.history_tokens = 5; r.committed_tokens = 4; r.pending_tokens = 1
        r.numerical_and_bridge_admitted_bytes = 1_000_000
        return r
    }
    static func main() throws {
        let value = record(), facts = try NativeContextFacts(value), model = NativeModelContext(facts)
        try require(facts.declaredTotal == 40960 && facts.loadedLimit == 512 && facts.capacity == 512,"separate declared/global/loaded/selected facts")
        let bad: [(String,(inout TokletContextInfo)->Void)] = [
            ("version",{ $0.abi_version = 2 }),("size",{ $0.struct_size += 1 }),
            ("reserved0",{ $0.reserved.0 = 1 }),("reserved1",{ $0.reserved.1 = 1 }),
            ("known-mask",{ $0.known_fields |= 8 }),("unknown-value",{ $0.known_fields = 0 }),
            ("empty-known-total",{ $0.declared_total_positions = 0 }),
            ("unknown-input",{ $0.declared_input_tokens = 1 }),
            ("empty-known-input",{ $0.known_fields |= UInt32(TOKLET_CONTEXT_DECLARED_INPUT_KNOWN) }),
            ("empty-known-output",{ $0.known_fields |= UInt32(TOKLET_CONTEXT_DECLARED_OUTPUT_KNOWN) }),
            ("backend",{ $0.backend = 2 }),("qualification",{ $0.context_qualification = 1 }),
            ("device-fit",{ $0.whole_app_device_fit = 1 }),("model-id",{ $0.model_instance = 0 }),
            ("session-id",{ $0.session_instance = 0 }),("aliased-id",{ $0.session_instance = 11 }),
            ("global",{ $0.engine_global_position_ceiling = 256 }),
            ("loaded",{ $0.loaded_position_limit = 0 }),("capacity",{ $0.session_capacity = 0 }),
            ("capacity+1",{ $0.session_capacity = 513 }),("offset+capacity",{ $0.absolute_position_offset = 1 }),
            ("offset-overflow",{ $0.absolute_position_offset = .max }),
            ("history",{ $0.history_tokens = 513 }),("pending-partition",{ $0.pending_tokens = 2 }),
            ("count-overflow",{ $0.committed_tokens = .max }),
            ("declaration-below-load",{ $0.declared_total_positions = 511 }),
            ("admission",{ $0.numerical_and_bridge_admitted_bytes = 0 })]
        for (name,mutation) in bad {
            var copy = value; mutation(&copy)
            try rejected({ _ = try NativeContextFacts(copy) },name)
        }
        var later = value; later.session_instance = 13; later.history_tokens = 500; later.committed_tokens = 500; later.pending_tokens = 0
        let next = try NativeContextFacts(later)
        try require(NativeModelContext(next) == model && next != facts,"retained model excludes live session/history")
        let chat = NativeGenerationPolicy.limits(.chat,model:model), raw = NativeGenerationPolicy.limits(.raw,model:model)
        try require(chat.maximumPrompt == 512 && chat.maximumOutput == 256 && raw.maximumOutput == 128,"existing mode limits remain explicit")
        var wide = value; wide.loaded_position_limit = 1024; wide.session_capacity = 1024
        let wideLimits = NativeGenerationPolicy.limits(.chat,model:NativeModelContext(try NativeContextFacts(wide)))
        try require(wideLimits.maximumPrompt == 1024 && wideLimits.maximumOutput == 256,"Chat prompt follows active KV window")
        wide.loaded_position_limit = 32768; wide.session_capacity = 32768
        let ceiling = NativeGenerationPolicy.limits(.chat,model:NativeModelContext(try NativeContextFacts(wide)))
        try require(ceiling.maximumPrompt == 32768 && ceiling.maximumOutput == 256,"Codec ceiling and reply bound remain separate")
        try require(chat.maximumInputTextBytes == 65536 && raw.maximumInputTextBytes == 262144 && NativeGenerationPolicy.maximumDraftBytes == 16384,"mode-specific byte limits")
        try require(NativeGenerationPolicy.limits(.chat,model:NativeModelContext(next)) == chat,"reply choices do not consume old native history twice")
        var bounded = value; bounded.known_fields |= UInt32(TOKLET_CONTEXT_DECLARED_INPUT_KNOWN | TOKLET_CONTEXT_DECLARED_OUTPUT_KNOWN)
        bounded.declared_input_tokens = 100; bounded.declared_output_tokens = 32
        let caps = NativeGenerationPolicy.limits(.chat,model:NativeModelContext(try NativeContextFacts(bounded)))
        try require(caps.maximumPrompt == 100 && caps.maximumOutput == 32,"documented independent caps apply when known")
        try rejected({ try caps.validateHistory(tokens:101,output:0) },"full input-history model cap")
        var legacy = value; legacy.known_fields = 0; legacy.declared_total_positions = 0; legacy.backend = TOKLET_CONTEXT_CPU
        let unknown = try NativeContextFacts(legacy)
        try require(unknown.declaredTotal == nil && unknown.declaredOutput == nil,"legacy maxima remain unknown")
        var small = legacy; small.loaded_position_limit = 32; small.session_capacity = 32
        let current = try NativeContextFacts(small)
        // Frozen integer admission witnesses from the accepted native L3.1 run.
        let rows: [(UInt32,UInt64,UInt64,UInt64,UInt32,UInt64,UInt64)] = [
            (0,0,2,2,30,32,0),(3,0,2,2,30,32,0),(1,5,2,7,25,32,0),(2,5,0,5,27,32,0),(2,5,0,5,0,5,27)]
        var accepted = [TokletRequestAdmission]()
        for (op,base,input,after,output,required,remaining) in rows {
            var r = TokletRequestAdmission(); r.context = small; r.operation = op
            r.base_history_tokens = base; r.input_tokens = input; r.history_after_input_tokens = after
            r.requested_output_tokens = UInt64(output); r.required_capacity = required; r.remaining_tokens = remaining
            let budget = try NativeRequestBudget(r,expected:current,operation:op,input:Int(input),output:output)
            try require(budget.required == required && budget.remaining == remaining,"frozen shared arithmetic")
            accepted.append(r)
        }
        let budgetBad: [(String,(inout TokletRequestAdmission)->Void)] = [
            ("operation",{ $0.operation = 3 }),("reserved",{ $0.reserved = 1 }),
            ("model-instance",{ $0.context.model_instance = 13 }),("session-instance",{ $0.context.session_instance = 13 }),
            ("base",{ $0.base_history_tokens = 4 }),("new-input",{ $0.input_tokens = 3 }),
            ("input-history",{ $0.history_after_input_tokens = 6 }),("requested-output",{ $0.requested_output_tokens = 26 }),
            ("required",{ $0.required_capacity = 31 }),("remaining",{ $0.remaining_tokens = 1 })]
        for (name,mutation) in budgetBad {
            var copy = accepted[2]; mutation(&copy)
            try rejected({ _ = try NativeRequestBudget(copy,expected:current,operation:1,input:2,output:25) },name)
        }
        try rejected({ _ = try NativeRequestBudget(accepted[2],expected:current,operation:1,input:2,output:26) },"exact+1 shared budget")
        var crowded = small; crowded.history_tokens = 31; crowded.committed_tokens = 30; crowded.pending_tokens = 1
        let crowdedFacts = try NativeContextFacts(crowded)
        var reset = accepted[1]; reset.context = crowded
        _ = try NativeRequestBudget(reset,expected:crowdedFacts,operation:3,input:2,output:30)
        var huge = small; huge.engine_global_position_ceiling = .max; huge.loaded_position_limit = .max; huge.session_capacity = .max
        huge.history_tokens = .max - 1; huge.committed_tokens = .max - 2
        let hugeFacts = try NativeContextFacts(huge)
        var overflow = accepted[2]; overflow.context = huge
        try rejected({ _ = try NativeRequestBudget(overflow,expected:hugeFacts,operation:1,input:4,output:1) },"history+input overflow")
        let summary: [String:Any] = ["passed":true,"invalid_context_records":bad.count,"frozen_shared_cases":rows.count,"invalid_budget_records":budgetBad.count+2,"mode_byte_limits_preserved":true,"reply_choices_ignore_old_history":true,"model_caps_optional":true]
        print("TOKLET_CONTEXT_POLICY_JSON=" + String(decoding:try JSONSerialization.data(withJSONObject:summary,options:.sortedKeys),as:UTF8.self))
    }
}

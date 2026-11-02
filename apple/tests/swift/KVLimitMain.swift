import Foundation
import CToklet

private final class KVPreparationCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func prepared() { lock.lock(); value += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return value }
}

@main
@MainActor
struct KVLimitMain {
    static func check(_ ok: Bool, _ message: String) throws {
        if !ok { throw NativeContextFailure(message) }
    }
    static func rejects(_ body: () throws -> Void) throws {
        var rejected = false
        do { try body() } catch { rejected = true }
        try check(rejected,"Invalid KV constraint accepted")
    }
    static func wire() -> [String:Any] {
        let memory: [String:Any] = ["weights_bytes":100,"kv_bytes":80,"scratch_bytes":4,"history_bytes":4,
            "logits_bytes":4,"metadata_bytes":4,"metadata_allowance_bytes":4,"admitted_bytes":200]
        let resources: [String:Any] = ["schema_version":1,"device_name":"synthetic","maximum_buffer_bytes":1000,
            "requested_load_context":4,"requested_capacity":4,"requested_chunk_size":1,"requested_position_offset":0,
            "requested_load_budget_bytes":1000,"requested_session_budget_bytes":1000,"projection":"MlxAffine",
            "normalization":"Compensated","readback":"FullLogits","attention":"Reference",
            "source_payload_bytes":100,"source_admitted_load_bytes":100,"resident_weights_bytes":100,
            "resident_admitted_load_bytes":100,"session_memory":memory,"fits_source_load_budget":true,
            "fits_resident_load_budget":true,"fits_session_budget":true,"fits_requested_budgets":true,
            "numerical_buffers_allocated":false,"payload_validation_required":true,"qualification_evaluated":false]
        func point(_ n: Int, _ kv: Int, _ fits: Bool) -> [String:Any] {
            ["capacity":n,"footprint":["admitted_bytes":120+kv,"largest_buffer_bytes":100,"kv_bytes":kv],
             "fits_budget":true,"fits_buffer":true,"fits_kv_budget":fits]
        }
        return ["resources":resources,"capacity":["minimum":1,"maximum":4,
            "limits":["budget_bytes":1000,"maximum_buffer_bytes":1000,"kv_budget_bytes":50],
            "selected":point(2,40,true),"first_excluded":point(3,60,false)]]
    }
    static func set(_ object: inout [String:Any], _ path: [String], _ value: Any?) {
        let key = path[0]
        if path.count == 1 { if let value { object[key] = value } else { object.removeValue(forKey:key) }; return }
        var child = object[key] as! [String:Any]
        set(&child,Array(path.dropFirst()),value); object[key] = child
    }
    static func controls() throws {
        let selection = NativePreviewSelection(id:UUID(),directory:"/never-opened",profile:.ordinary)
        var draft = FriendlyPlanningDraft()
        try check(try draft.request(selection:selection).kvBudgetBytes == nil,"Blank KV ceiling is not unlimited")
        draft.kvBudgetMiB = "0"
        try check(try draft.request(selection:selection).kvBudgetBytes == 0,"Explicit zero became unlimited")
        draft.kvBudgetMiB = "8"
        let request = try draft.request(selection:selection)
        let derived = try request.capacityRequest(planningUpper:1024)
        try check(request.kvBudgetBytes == 8388608 && derived.kvBudgetBytes == 8388608,"KV ceiling lost during request derivation")
        for bad in ["-1","+1"," 8","8 ","1.5","6145","18446744073709551616","١"] {
            draft.kvBudgetMiB = bad; try rejects { _ = try draft.request(selection:selection) }
        }
        var changed = request; changed.kvBudgetBytes = nil
        try check(changed != request,"KV edit retained stale request equality")
        var capRequest = NativeCapacityPlan.Request(context:4,upper:4,chunk:1,offset:0,loadBudget:1000,sessionBudget:1000)
        capRequest.kvBudgetBytes = 50
        let base = wire()
        let decoded = try NativeCapacityPlan.decode(JSONSerialization.data(withJSONObject:base),request:capRequest)
        try check(decoded.capacity.selected?.capacity == 2 && decoded.capacity.firstExcluded?.fitsKvBudget == false,"KV-only adjacent exclusion rejected")
        for (path,value) in [(["capacity","limits","kv_budget_bytes"],51 as Any?),
                             (["capacity","selected","fits_kv_budget"],false as Any?),
                             (["capacity","selected","footprint","kv_bytes"],nil),
                             (["capacity","first_excluded","fits_kv_budget"],true as Any?),
                             (["capacity","selected","footprint","kv_bytes"],81 as Any?)] {
            var bad = base; set(&bad,path,value)
            try rejects { _ = try NativeCapacityPlan.decode(JSONSerialization.data(withJSONObject:bad),request:capRequest) }
        }
        var zero = base; capRequest.kvBudgetBytes = 0
        set(&zero,["capacity","limits","kv_budget_bytes"],0)
        set(&zero,["capacity","selected"],NSNull())
        set(&zero,["capacity","first_excluded"],["capacity":1,"footprint":["admitted_bytes":140,"largest_buffer_bytes":100,"kv_bytes":20],"fits_budget":true,"fits_buffer":true,"fits_kv_budget":false])
        try check(try NativeCapacityPlan.decode(JSONSerialization.data(withJSONObject:zero),request:capRequest).capacity.selected == nil,"Zero-byte no-fit report rejected")
        var raw = TokletContextInfo()
        raw.abi_version = TOKLET_ABI_VERSION; raw.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        raw.backend = TOKLET_CONTEXT_CPU; raw.model_instance = 1; raw.session_instance = 2
        raw.engine_global_position_ceiling = 32768; raw.loaded_position_limit = 4; raw.session_capacity = 4
        raw.numerical_and_bridge_admitted_bytes = 1024 * 1024
        let config = try NativeSessionConfiguration(context:NativeContextFacts(raw),capacity:4,chunk:1,wholeSessionBudget:128*1024*1024)
        let unavailable = NativeSessionMemorySnapshot.capture(configuration:config) { throw NativeContextFailure("synthetic unavailable") }
        var transition = NativeTransitionRequest(id:UUID(),selection:.init(id:UUID(),modelInstance:1,sessionInstance:2),
            capacity:4,chunk:1,wholeSessionBudget:128*1024*1024,coexistenceBudget:256*1024*1024,
            replayReserve:0,textReserve:0,applicationReserve:2*1024*1024)
        try transition.validateCandidateKV(unavailable) // Unlimited preserves optional reporting.
        transition.kvBudgetBytes = 50
        try transition.validateKV(bytes:50)
        try rejects { try transition.validateKV(bytes:51) }
        try rejects { try transition.validateCandidateKV(unavailable) }
        print("KV_LIMIT_CONTROLS passed")
    }
    static func main() throws {
        try controls()
        guard CommandLine.arguments.count > 1 else { return }
        try check(CommandLine.arguments.count == 3,"Expected cached model and fresh output directory")
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let harness = HarnessStore(enabled:true)
        guard let original = harness.worker else { throw NativeContextFailure("Missing worker") }
        original.close(); try check(original.waitUntilClosed(seconds:30),"Initial close timeout")
        let counter = KVPreparationCounter(), worker = try NativeWorker(beforeTransitionAcceptance:{ _,_ in counter.prepared() })
        harness.worker = worker; harness.state = worker.snapshot()
        let preferences = FriendlyPreferenceStore(directory:output.appendingPathComponent("preferences"))
        let chat = FriendlyChatStore(harness:harness,fileURL:output.appendingPathComponent("chat.json"),preferences:preferences)
        var failure: Error?
        func wait() throws -> NativeSnapshot {
            guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("KV worker timeout") }
            harness.poll(); chat.observe(harness.state); return state
        }
        do {
            chat.setQwen2LoadUpper(1024); chat.openManaged(model,profile:.qwen,ifSelected:chat.workspace.selected)
            let loaded = try wait(); try check(loaded.status == 0,"Load failed: \(loaded.message)")
            guard let identity = loaded.acceptedIdentity else { throw NativeContextFailure("Missing identity") }
            chat.showPlanning(); chat.editPlanning(.upper,"1024"); chat.editPlanning(.kvBudgetMiB,"8"); chat.previewPlanning()
            _ = try wait()
            guard let preview = chat.planning.result, let plan = preview.capacity else { throw NativeContextFailure("Missing constrained preview") }
            try check(plan.capacity.selected?.capacity == 682 && plan.capacity.selected?.footprint.kvBytes == 8381088
                && plan.capacity.firstExcluded?.capacity == 683 && plan.capacity.firstExcluded?.fitsKvBudget == false
                && preview.request.kvBudgetBytes == 8388608 && chat.planningApplyUnavailableReason == nil,"Physical KV ceiling did not constrain preview")
            chat.applyPlanning(); let applied = try wait()
            guard let configuration = applied.sessionConfiguration, let memory = applied.sessionMemory,
                  case .available(_,let owned,_,_,_) = memory.reading else { throw NativeContextFailure("Actual candidate KV unavailable") }
            try check(applied.transition?.committed == true && applied.transition?.request.kvBudgetBytes == 8388608
                && owned.kvBytes == 8381088 && configuration.capacity == 682 && applied.acceptedIdentity == identity
                && counter.count == 1,"Actual accepted KV does not obey ceiling")
            let record = try preferences.url(identity.namespace), saved = try Data(contentsOf:record)
            let json = try JSONSerialization.jsonObject(with:saved) as! [String:Any]
            try check(json["schema"] as? Int == 1 && json["kv_budget_bytes"] == nil,"Ephemeral KV ceiling changed saved schema")
            let selection = NativeTransitionSelection(id:UUID(),modelInstance:configuration.modelInstance,sessionInstance:configuration.sessionInstance)
            var rejected = NativeTransitionRequest(id:UUID(),selection:selection,capacity:683,chunk:32,
                wholeSessionBudget:configuration.wholeSessionBudget,coexistenceBudget:HubProfile.qwen.processAllowance,
                replayReserve:0,textReserve:HubProfile.qwen.textBudget,applicationReserve:NativeTransitionHeadroomCheck.applicationReserve)
            rejected.kvBudgetBytes = 8388608; rejected.memoryPolicy = .friendly
            worker.selectTransition(selection); try check(worker.submit(.replaceWithEmptySession(rejected)),"KV rejection not submitted")
            let denied = try wait()
            try check(denied.transition?.committed == false && denied.transition?.message.contains("KV exceeds") == true
                && denied.sessionConfiguration == configuration && denied.acceptedIdentity == identity && counter.count == 1
                && (try Data(contentsOf:record)) == saved,"Resident over-ceiling request allocated/committed or changed settings")
            chat.editPlanning(.kvBudgetMiB,"0"); chat.previewPlanning(); _ = try wait()
            try check(chat.planning.result?.capacity?.capacity.selected == nil && chat.planningApplyUnavailableReason != nil
                && worker.snapshot().sessionConfiguration == configuration,"Zero KV preview changed current owner")
            chat.editPlanning(.upper,"1024"); chat.editPlanning(.kvBudgetMiB,""); chat.previewPlanning(); _ = try wait()
            try check(chat.planning.result?.capacity?.capacity.selected?.capacity == 1024
                && chat.planning.result?.request.kvBudgetBytes == nil,"Unlimited preview did not preserve legacy capacity")
            print("KV_LIMIT_MAC passed ceiling=8388608 selected=682 actual_kv=8381088 excluded=683 preparation_callback_unchanged_on_rejection=true zero_and_unlimited=true")
        } catch { failure = error }
        worker.close(); try check(worker.waitUntilClosed(seconds:60),"KV worker close timeout")
        if let failure { throw failure }
    }
}

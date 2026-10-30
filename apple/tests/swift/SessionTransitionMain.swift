import Foundation
import CToklet

private func check(_ ok: @autoclosure () -> Bool, _ message: String) throws {
    if !ok() { throw NativeContextFailure(message) }
}
private func rejects(_ body: () throws -> Void) throws {
    do { try body() } catch { return }
    throw NativeContextFailure("Malformed transition accepted")
}
private func idle(_ worker: NativeWorker) throws -> NativeSnapshot {
    guard let value = worker.waitUntilIdle(seconds:60) else { throw NativeContextFailure("Worker deadline") }
    return value
}
private func unchanged(_ lhs: NativeSnapshot, _ rhs: NativeSnapshot) throws {
    var left = lhs, right = rhs
    left.revision = 0; right.revision = 0; left.transition = nil; right.transition = nil
    try check(String(reflecting:left) == String(reflecting:right),"Rejected transition changed ordinary snapshot")
}
private func capture(_ label: String, _ snapshot: NativeSnapshot) throws {
    var fields: [String:Any] = ["label":label,"revision":snapshot.revision,"busy":snapshot.busy,
        "has_model":snapshot.hasModel,"has_session":snapshot.hasSession,"restart_required":snapshot.restartRequired,
        "history":snapshot.history,"committed":snapshot.evaluated,"pending":snapshot.pending,
        "ids":snapshot.ids,"saved_prefix":snapshot.savedPrefix as Any? ?? NSNull(),"status":snapshot.status]
    if let context = snapshot.liveContext {
        fields["context"] = ["model":context.modelInstance,"session":context.sessionInstance,
            "capacity":context.capacity,"loaded_limit":context.loadedLimit,"admission":context.numericalAdmission]
    }
    if let result = snapshot.transition {
        fields["transition"] = ["outcome":result.outcome.rawValue,"committed":result.committed,
            "status":result.status,"requested_capacity":result.request.capacity,"requested_chunk":result.request.chunk]
        if let plan = result.plan {
            fields["plan"] = ["current_chunk":plan.current.chunk,"candidate_chunk":plan.candidate.chunk,
                "shared_bytes":plan.sharedModel.totalBytes,"current_owned_bytes":plan.currentAllocated.totalBytes,
                "candidate_planned_bytes":plan.candidatePlanned.totalBytes,"reserves_bytes":plan.reserves.totalBytes,
                "aggregate_bytes":plan.aggregateRequiredBytes]
        }
    }
    let data = try JSONSerialization.data(withJSONObject:fields,options:[.sortedKeys])
    guard data.count <= 65536 else { throw NativeContextFailure("Oversized transition diagnostic") }
    print("SESSION_TRANSITION_RECORD " + String(decoding:data,as:UTF8.self))
}
private final class Gate: @unchecked Sendable {
    let reached = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
    private let lock = NSLock()
    private var armed = false
    private var rejection: UInt32?
    func arm() { lock.lock(); rejection = nil; armed = true; lock.unlock() }
    func pause() {
        lock.lock(); let wait = armed; armed = false; lock.unlock()
        if wait { reached.signal(); precondition(proceed.wait(timeout:.now()+30) == .success) }
    }
    func reject(_ status: UInt32) {
        lock.lock(); let active = armed; armed = false
        if active { rejection = status }
        lock.unlock()
        if active { reached.signal() }
    }
    func wait(_ worker: NativeWorker) throws {
        let reachedInTime = reached.wait(timeout:.now()+30) == .success
        lock.lock(); let status = rejection; lock.unlock()
        if !reachedInTime {
            let raw = "SESSION_TRANSITION_BARRIER_DEADLINE " + String(reflecting:worker.snapshot()) + "\n"
            FileHandle.standardError.write(Data(raw.utf8.prefix(65536)))
            throw NativeContextFailure("Transition barrier deadline; actual snapshot retained on stderr")
        }
        if let status {
            let terminal = worker.waitUntilIdle(seconds:30)
            let observed = terminal ?? worker.snapshot()
            let raw = "SESSION_TRANSITION_UNEXPECTED_REJECTION " + String(reflecting:observed) + "\n"
            FileHandle.standardError.write(Data(raw.utf8.prefix(65536)))
            throw NativeContextFailure("Transition rejected before acceptance, status \(status): \(observed.transition?.message ?? observed.message)")
        }
    }
}
private func selection(_ snapshot: NativeSnapshot) throws -> NativeTransitionSelection {
    guard let facts = snapshot.liveContext else { throw NativeContextFailure("Missing live context") }
    return NativeTransitionSelection(id:UUID(),modelInstance:facts.modelInstance,sessionInstance:facts.sessionInstance)
}
private func request(_ selection: NativeTransitionSelection, capacity: UInt32 = 16,
                     chunk: UInt32 = 3, budget: UInt64 = 512 * 1024 * 1024) -> NativeTransitionRequest {
    NativeTransitionRequest(id:UUID(),selection:selection,capacity:capacity,chunk:chunk,
        wholeSessionBudget:128*1024*1024,coexistenceBudget:budget,replayReserve:0,
        textReserve:0,applicationReserve:2*1024*1024)
}
private func loaded(_ worker: NativeWorker, root: URL, metal: Bool) throws -> NativeSnapshot {
    var settings = NativeSettings(directory:root.appendingPathComponent("artifacts/m7-native-tiny-v2").path)
    if metal {
        settings.metal = true; settings.mlx = true
        settings.directory = root.appendingPathComponent("tests/fixtures/mlx-qwen3-v1/tied").path
    }
    try check(worker.submit(.load(settings)),"Load enqueue")
    let initial = try idle(worker)
    try check(initial.status == 0 && initial.hasSession,"Load failed: \(initial.status) \(initial.message)")
    try check(worker.submit(.generate(operation:0,tokens:[1,7,13,3],maximum:2,stops:[])),"Generate enqueue")
    let value = try idle(worker)
    try check(value.status == 0 && value.ids.count == 2 && value.history == 6 && value.pending == 1,
              "Generation witness failed: \(value.status) \(value.message)")
    return value
}

private func pureControls() throws -> Int {
    var raw = TokletContextInfo()
    raw.abi_version = TOKLET_ABI_VERSION; raw.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
    raw.model_instance = 1; raw.session_instance = 2; raw.backend = TOKLET_CONTEXT_CPU
    raw.engine_global_position_ceiling = 32768; raw.loaded_position_limit = 32; raw.session_capacity = 32
    raw.history_tokens = 6; raw.committed_tokens = 5; raw.pending_tokens = 1
    let bridge = UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES), allowance: UInt64 = 64*1024*1024
    raw.numerical_and_bridge_admitted_bytes = 150+300+allowance+bridge
    let facts = try NativeContextFacts(raw)
    let selected = NativeTransitionSelection(id:UUID(),modelInstance:1,sessionInstance:2)
    let wanted = request(selected)
    let reserve = 2*allowance+2*bridge+131328+2*1024*1024
    let valid: [String:Any] = [
        "schema_version":1,"report_kind":"same_model_session_transition","model_instance":1,"session_instance":2,
        "backend":"cpu","mode_selection":"existing_native_factory","loaded_position_limit":32,
        "execution_modes":["projection":"Scalar","normalization":"Scalar","attention":"Scalar","readback":"FullLogits"],
        "current":["capacity":32,"chunk":3,"position_offset":0,"individual_numerical_budget_bytes":128*1024*1024-bridge],
        "candidate":["capacity":16,"chunk":3,"position_offset":0,"individual_numerical_budget_bytes":128*1024*1024-bridge],
        "history_tokens":6,"committed_tokens":5,"pending_tokens":1,
        "shared_model":["learned_bytes":100,"positional_bytes":20,"model_table_bytes":30,"total_bytes":150],
        "current_allocated":["kv_bytes":40,"scratch_bytes":50,"history_bytes":60,"logits_bytes":70,"session_table_bytes":80,"total_bytes":300],
        "candidate_planned":["kv_bytes":10,"scratch_bytes":20,"history_bytes":30,"logits_bytes":40,"session_table_bytes":50,"total_bytes":150],
        "reserves":["current_metadata_driver_bytes":allowance,"candidate_metadata_driver_bytes":allowance,
            "current_bridge_bytes":bridge,"candidate_bridge_bytes":bridge,"json_snapshot_bytes":131328,
            "replay_bytes":0,"text_bytes":0,"application_bytes":2*1024*1024,"total_bytes":reserve],
        "aggregate_required_bytes":150+300+150+reserve,"transition_budget_bytes":wanted.coexistenceBudget,
        "fits_transition_budget":true,"candidate_constructor_required_bytes":150+150+allowance+bridge,
        "candidate_whole_budget_bytes":wanted.wholeSessionBudget,"fits_candidate_individual_budget":true,
        "candidate_largest_device_buffer_bytes":NSNull(),"device_maximum_buffer_bytes":NSNull(),
        "candidate_allocated":false,"reserves_verified":false,"creation_revalidation_required":true,
        "context_qualification_evaluated":false,"whole_app_device_fit_evaluated":false]
    func decode(_ object: [String:Any], _ expected: NativeTransitionRequest? = nil) throws {
        _ = try NativeTransitionPlan.decode(JSONSerialization.data(withJSONObject:object),request:expected ?? wanted,
            current:facts,chunk:3,wholeBudget:128*1024*1024)
    }
    try decode(valid)
    var historical = raw
    historical.numerical_and_bridge_admitted_bytes += 4096
    _ = try NativeTransitionPlan.decode(JSONSerialization.data(withJSONObject:valid),request:wanted,
        current:NativeContextFacts(historical),chunk:3,wholeBudget:128*1024*1024)
    let bad: [(String,Any)] = [("schema_version",2),("report_kind","other"),("model_instance",3),
        ("session_instance",3),("backend","metal"),("loaded_position_limit",64),("history_tokens",7),
        ("committed_tokens",4),("pending_tokens",0),("aggregate_required_bytes",0),
        ("fits_transition_budget",false),("candidate_constructor_required_bytes",0),
        ("candidate_allocated",true),("reserves_verified",true),("creation_revalidation_required",false),
        ("context_qualification_evaluated",true),("whole_app_device_fit_evaluated",true),
        ("candidate_largest_device_buffer_bytes",1),("transition_budget_bytes",UInt64.max)]
    for (key,value) in bad { var changed = valid; changed[key] = value; try rejects { try decode(changed) } }
    for section in ["shared_model","current_allocated","candidate_planned","reserves"] {
        var changed = valid, table = changed[section] as! [String:Any]
        table["total_bytes"] = UInt64.max; changed[section] = table
        try rejects { try decode(changed) }
    }
    var missing = valid; missing.removeValue(forKey:"reserves_verified")
    try rejects { try decode(missing) }
    try rejects { try decode(valid,request(selected,capacity:8)) }
    try rejects { _ = try NativeTransitionPlan.sum([UInt64.max,1]) }
    try rejects { try request(selected,capacity:0).validate() }
    try rejects { try request(selected,capacity:16,chunk:17).validate() }
    var candidate = raw
    candidate.session_instance = 3; candidate.session_capacity = 16
    candidate.history_tokens = 0; candidate.committed_tokens = 0; candidate.pending_tokens = 0
    try NativeTransitionPlan.validateCandidate(NativeContextFacts(candidate),old:facts,request:wanted)
    candidate.loaded_position_limit = 64
    try rejects { try NativeTransitionPlan.validateCandidate(NativeContextFacts(candidate),old:facts,request:wanted) }
    candidate.loaded_position_limit = 32; candidate.pending_tokens = 1; candidate.history_tokens = 1
    try rejects { try NativeTransitionPlan.validateCandidate(NativeContextFacts(candidate),old:facts,request:wanted) }
    // Metal aggregate owns the layer table once, but its constructor Memory
    // excludes it. Use a nonzero real-sized table to distinguish the equations.
    var metalRaw = raw; metalRaw.backend = TOKLET_CONTEXT_METAL
    let metalFacts = try NativeContextFacts(metalRaw)
    var metal = valid
    metal["backend"] = "metal"
    metal["execution_modes"] = ["projection":"MlxAffine","normalization":"Original","attention":"Reference","readback":"FullLogits"]
    metal["shared_model"] = ["learned_bytes":100,"positional_bytes":20,"model_table_bytes":4480,"total_bytes":4600]
    metal["aggregate_required_bytes"] = 4600+300+150+reserve
    metal["candidate_constructor_required_bytes"] = 120+150+allowance+bridge
    metal["candidate_largest_device_buffer_bytes"] = 64
    metal["device_maximum_buffer_bytes"] = 1024
    func decodeMetal(_ value: [String:Any]) throws {
        _ = try NativeTransitionPlan.decode(JSONSerialization.data(withJSONObject:value),request:wanted,
            current:metalFacts,chunk:3,wholeBudget:128*1024*1024)
    }
    try decodeMetal(metal)
    var wrongMetalConstructor = metal
    wrongMetalConstructor["candidate_constructor_required_bytes"] = 4600+150+allowance+bridge
    try rejects { try decodeMetal(wrongMetalConstructor) }
    var missingMetalTable = metal
    missingMetalTable["aggregate_required_bytes"] = 120+300+150+reserve
    try rejects { try decodeMetal(missingMetalTable) }
    var missingCPUConstructorTable = valid
    missingCPUConstructorTable["candidate_constructor_required_bytes"] = 120+150+allowance+bridge
    try rejects { try decode(missingCPUConstructorTable) }
    return 37
}

@main
struct SessionTransitionMain {
    static func main() throws {
        guard CommandLine.arguments.count == 3, ["cpu","metal"].contains(CommandLine.arguments[2]) else {
            throw NativeContextFailure("Expected root and cpu/metal")
        }
        let root = URL(fileURLWithPath:CommandLine.arguments[1]), metal = CommandLine.arguments[2] == "metal"
        let pure = try pureControls()
        let gate = Gate()
        let worker = try NativeWorker(captureSavedPrefix:true,beforeTransitionAcceptance:{ _,_ in gate.pause() },
            beforeTransitionRejection:{ _,status in gate.reject(status) })
        defer { gate.proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let generated = try loaded(worker,root:root,metal:metal), selected = try selection(generated)
        try capture("used-current",generated)
        try check(worker.selectTransition(selected),"Selection")
        try check(!worker.submit(.replaceWithEmptySession(request(selected,capacity:0))),"Malformed request enqueued")
        try check(worker.submit(.replaceWithEmptySession(request(selected,budget:0))),"Underbudget enqueue")
        var state = try idle(worker)
        try capture("underbudget-rejection",state)
        try check(state.transition?.outcome == .rejected,"Missing budget rejection")
        try unchanged(generated,state)
        try check(worker.submit(.replaceWithEmptySession(request(selected)),initiallyCancelled:true),"Initial cancel enqueue")
        state = try idle(worker); try unchanged(generated,state)
        try check(state.transition == nil,"Initially cancelled result published")
        gate.arm()
        try check(worker.submit(.replaceWithEmptySession(request(selected))),"Candidate enqueue")
        try gate.wait(worker); worker.cancel(); gate.proceed.signal()
        state = try idle(worker); try unchanged(generated,state)
        try check(state.transition == nil,"Cancelled candidate published")
        try check(worker.selectTransition(selected),"Reselect")
        let smaller = request(selected,capacity:3,chunk:2)
        try check(worker.submit(.replaceWithEmptySession(smaller)),"Shrink enqueue")
        state = try idle(worker)
        try capture("empty-adoption",state)
        try check(state.transition?.outcome == .committed && state.transition?.committed == true,"Empty adoption failed: \(String(describing:state.transition))")
        try check(state.liveContext?.modelInstance == generated.liveContext?.modelInstance
            && state.liveContext?.sessionInstance != generated.liveContext?.sessionInstance
            && state.liveContext?.capacity == 3 && state.modelContext?.configuredCapacity == 3
            && state.liveContext?.loadedLimit == generated.liveContext?.loadedLimit
            && state.history == 0 && state.evaluated == 0 && state.pending == 0 && state.savedPrefix == [],"Committed identity/cache facts")
        try check(state.runtimePlan?.requestedCapacity == generated.runtimePlan?.requestedCapacity
            && state.compatibilityDirectory == generated.compatibilityDirectory && state.chatEnabled == generated.chatEnabled,"Model/load metadata changed")
        try check(!worker.submit(.replaceWithEmptySession(smaller)),"Old selection reused")
        try check(worker.submit(.generate(operation:3,tokens:[1],maximum:1,stops:[])),"New session unusable")
        state = try idle(worker); try check(state.status == 0 && state.ids.count == 1,"Candidate generation failed")

        var barriers = 0
        for stage in ["dequeue","acceptance","rejection"] {
            for action in ["selection","cancel","suspend","unload","close"] {
                try stale(root:root,metal:metal,stage:stage,action:action); barriers += 1
            }
        }
        try afterAcceptance(root:root,metal:metal)
        for fault in [NativeTransitionRetirementFault.busy,.consumedPanic] {
            try retirement(root:root,metal:metal,fault:fault)
        }
        worker.close(); try check(worker.waitUntilClosed(seconds:30),"Final close")
        print("SESSION_TRANSITION_CHECKS " + String(data:try JSONSerialization.data(withJSONObject:[
            "passed":true,"pure_controls":pure,"stale_barriers":barriers,"retirement_controls":2,
            "backend":metal ? "metal":"cpu","empty_commit":true,"old_pending_witness":true,
            "postcommit_cancel":true,"closed":true],options:[.sortedKeys]),encoding:.utf8)!)
    }

    private static func stale(root: URL, metal: Bool, stage: String, action: String) throws {
        let gate = Gate()
        let worker = try NativeWorker(captureSavedPrefix:true,
            beforeCommandDequeue:{ command in if case .replaceWithEmptySession = command, stage == "dequeue" { gate.pause() } },
            beforeTransitionAcceptance:{ _,_ in if stage == "acceptance" { gate.pause() } },
            beforeTransitionRejection:{ _,status in
                if stage == "rejection" { gate.pause() } else { gate.reject(status) } })
        defer { gate.proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let before = try loaded(worker,root:root,metal:metal), selected = try selection(before)
        worker.selectTransition(selected); gate.arm()
        try check(worker.submit(.replaceWithEmptySession(request(selected,budget:stage == "rejection" ? 0 : 512*1024*1024))),"Stale enqueue")
        try gate.wait(worker)
        switch action {
        case "selection": worker.selectTransition(NativeTransitionSelection(id:UUID(),modelInstance:selected.modelInstance,sessionInstance:selected.sessionInstance))
        case "cancel": worker.cancel()
        case "suspend": worker.interrupt(unload:false)
        case "unload": worker.interrupt(unload:true)
        default: worker.close()
        }
        gate.proceed.signal()
        if action == "close" { try check(worker.waitUntilClosed(seconds:30),"Stale close") }
        let after = try idle(worker)
        try check(after.transition == nil,"Stale transition published")
        if action == "selection" || action == "cancel" {
            try unchanged(before,after)
            try check(worker.submit(.generate(operation:2,tokens:[],maximum:1,stops:[])),"Old continuation enqueue")
            let continued = try idle(worker); try check(continued.status == 0,"Old session lost usability")
        } else { try check(!after.hasSession,"Lifecycle retained session") }
    }

    private static func afterAcceptance(root: URL, metal: Bool) throws {
        let gate = Gate()
        let worker = try NativeWorker(beforeTransitionRejection:{ _,status in gate.reject(status) },
            afterTransitionAcceptance:{ _ in gate.pause() })
        defer { gate.proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let before = try loaded(worker,root:root,metal:metal), selected = try selection(before)
        worker.selectTransition(selected); gate.arm()
        try check(worker.submit(.replaceWithEmptySession(request(selected))),"Postaccept enqueue")
        try gate.wait(worker)
        let committed = worker.snapshot()
        try capture("accepted-before-late-cancel",committed)
        try check(committed.transition?.committed == true && committed.liveContext?.sessionInstance != selected.sessionInstance,"Acceptance not visible")
        worker.cancel(); gate.proceed.signal()
        let after = try idle(worker)
        try capture("accepted-after-late-cancel",after)
        try check(after.liveContext == committed.liveContext && after.hasSession,"Late cancellation rolled back adoption")
    }

    private static func retirement(root: URL, metal: Bool, fault: NativeTransitionRetirementFault) throws {
        let worker = try NativeWorker(transitionRetirementFault:{ fault })
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let before = try loaded(worker,root:root,metal:metal), selected = try selection(before)
        worker.selectTransition(selected)
        try check(worker.submit(.replaceWithEmptySession(request(selected))),"Retirement enqueue")
        let after = try idle(worker)
        try capture(fault == .busy ? "retirement-busy" : "retirement-panic",after)
        try check(after.restartRequired && after.transition?.outcome == .cleanupFailed
            && after.transition?.committed == true && after.liveContext?.sessionInstance != selected.sessionInstance,
            "Cleanup failure claimed rollback")
        try check(after.transition?.status == (fault == .busy ? TOKLET_BUSY:TOKLET_PANIC),"Cleanup status mismatch")
        try check(!worker.submit(.reset),"Quarantined worker accepted command")
        worker.close(); try check(worker.waitUntilClosed(seconds:30),"Retirement shutdown")
    }
}

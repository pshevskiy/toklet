import Foundation
import CryptoKit

private final class AdvisoryGate: @unchecked Sendable {
    private let lock = NSLock()
    private var armed: String?, loads = 0, commits = 0, rejections = 0, previews = 0
    let reached = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
    func arm(_ stage: String) { lock.lock(); armed = stage; lock.unlock() }
    func pause(_ stage: String) {
        lock.lock(); let wait = armed == stage; if wait { armed = nil }; lock.unlock()
        if wait { reached.signal(); precondition(proceed.wait(timeout:.now()+30) == .success,"Advisory barrier deadline") }
    }
    func command(_ command: NativeCommand) {
        if case .load = command { lock.lock(); loads += 1; let n = loads; lock.unlock(); precondition(n == 1) }
        if case .preview = command { lock.lock(); previews += 1; lock.unlock() }
    }
    func accepted() { lock.lock(); commits += 1; lock.unlock() }
    func rejected() { lock.lock(); rejections += 1; lock.unlock() }
    var counts: (Int,Int,Int,Int) { lock.lock(); defer { lock.unlock() }; return (loads,commits,rejections,previews) }
}

@main
@MainActor
struct SmolAdvisoryLoadedMain {
    static func check(_ value: Bool, _ message: String) throws { if !value { throw NativeContextFailure(message) } }
    static func hash(_ bytes: Data) -> String { SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined() }
    static func idle(_ harness: HarnessStore, _ chat: FriendlyChatStore, _ worker: NativeWorker) throws {
        guard worker.waitUntilIdle(seconds:180) != nil else { throw NativeContextFailure("Advisory idle deadline") }
        harness.poll(); chat.observe(harness.state)
    }
    static func preserved(_ a: NativeSnapshot, _ b: NativeSnapshot) throws {
        var a = a, b = b
        a.revision = 0; b.revision = 0; a.preview = nil; b.preview = nil; a.transition = nil; b.transition = nil
        try check(String(reflecting:a) == String(reflecting:b),"Preview/rejection changed active state")
    }
    // An explicit synthetic underestimate, not a native estimator report. Its
    // only role is to prove that actual model-bound admission rejects the attempt.
    static func underestimate(_ request: NativePreviewRequest) throws -> NativePreviewResult {
        let derived = try request.capacityRequest(planningUpper:512), bytes: UInt64 = 256*1024
        let memory: [String:Any] = ["weights_bytes":bytes-7,"kv_bytes":1,"scratch_bytes":1,"history_bytes":1,
            "logits_bytes":1,"metadata_bytes":1,"metadata_allowance_bytes":2,"admitted_bytes":bytes]
        let resources: [String:Any] = ["schema_version":1,"device_name":"synthetic underestimate","maximum_buffer_bytes":256*1024*1024,
            "requested_load_context":derived.context,"requested_capacity":derived.upper,"requested_chunk_size":derived.chunk,
            "requested_position_offset":0,"requested_load_budget_bytes":request.loadBudget,"requested_session_budget_bytes":request.sessionBudget,
            "projection":"MlxAffine","normalization":"Compensated","readback":"FullLogits","attention":"Reference",
            "source_payload_bytes":bytes,"source_admitted_load_bytes":request.loadBudget+1,"resident_weights_bytes":bytes-7,
            "resident_admitted_load_bytes":request.loadBudget+1,"session_memory":memory,
            "fits_source_load_budget":false,"fits_resident_load_budget":false,"fits_session_budget":true,
            "fits_requested_budgets":false,"numerical_buffers_allocated":false,"payload_validation_required":true,"qualification_evaluated":false]
        let point: [String:Any] = ["capacity":derived.upper,"footprint":["admitted_bytes":bytes,"largest_buffer_bytes":bytes-7],"fits_budget":true,"fits_buffer":true]
        let wire: [String:Any] = ["resources":resources,"capacity":["minimum":derived.chunk,"maximum":derived.upper,
            "limits":["budget_bytes":request.sessionBudget,"maximum_buffer_bytes":256*1024*1024],"selected":point,"first_excluded":NSNull()]]
        let plan = try NativeCapacityPlan.decode(JSONSerialization.data(withJSONObject:wire),request:derived)
        return NativePreviewResult(request:request,outcome:.planned,limits:nil,capacityRequest:derived,capacity:plan,limitation:nil,status:0,message:"")
    }
    static func requestFields(_ request: NativeTransitionRequest) -> [String:Any] {
        ["id":request.id.uuidString,"selection":request.selection.id.uuidString,
         "model":request.selection.modelInstance,"old_session":request.selection.sessionInstance,
         "capacity":request.capacity,"chunk":request.chunk,"whole_budget":request.wholeSessionBudget,
         "coexistence_budget":request.coexistenceBudget,"replay_reserve":request.replayReserve,
         "text_reserve":request.textReserve,"application_reserve":request.applicationReserve]
    }
    static func frame(_ event: String, _ worker: NativeWorker, _ chat: FriendlyChatStore,
                      _ workspace: URL, _ qwen: URL, _ writes: Int, preview: NativePreviewResult? = nil,
                      captured: NativeSnapshot? = nil, submitted: NativeTransitionRequest? = nil) throws {
        let state = captured ?? worker.snapshot(), encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var row: [String:Any] = ["event":event,"snapshot":String(reflecting:state),"preference_writes":writes,
            "workspace_sha256":hash(try encoder.encode(chat.workspace)),"file_sha256":hash(try Data(contentsOf:workspace)),
            "qwen_sha256":hash(try Data(contentsOf:qwen)),"draft":chat.workspace.current.draft,"reply":chat.workspace.outputLimit,
            "messages":chat.workspace.current.messages.count,"history":state.history,"committed":state.evaluated,"pending":state.pending,"generated":state.ids.count]
        if let i = state.acceptedIdentity { row["model"] = i.modelInstance; row["manifest"] = i.namespace.manifestSHA256; row["loaded_limit"] = i.loadedLimit }
        if let c = state.sessionConfiguration { row["configuration"] = ["model":c.modelInstance,"session":c.sessionInstance,"capacity":c.capacity,"chunk":c.chunk,"whole_budget":c.wholeSessionBudget,"accepted_request":c.acceptedRequestID?.uuidString as Any? ?? NSNull()] }
        if let value = preview ?? state.preview {
            row["estimator"] = value.request.estimator.rawValue; row["profile"] = value.request.selection.profile.rawValue
            row["requested_upper"] = value.request.upper; row["effective_upper"] = value.capacityRequest?.upper as Any? ?? NSNull()
            row["selected"] = value.capacity?.capacity.selected?.capacity as Any? ?? NSNull()
            row["preview_allocated"] = value.capacity?.resources.numericalBuffersAllocated as Any? ?? NSNull()
            row["preview_identity_authority"] = false
        }
        if let submitted { row["submitted_request"] = requestFields(submitted) }
        if let t = state.transition {
            row["transition_request"] = requestFields(t.request)
            row["transition"] = ["committed":t.committed,"status":t.status,"message":t.message,"outcome":t.outcome.rawValue]
            if let p = t.plan {
                row["actual_plan"] = ["model":p.modelInstance,"session":p.sessionInstance,"loaded_limit":p.loadedPositionLimit,
                    "shared":p.sharedModel.totalBytes,"learned":p.sharedModel.learnedBytes,"positional":p.sharedModel.positionalBytes,"model_table":p.sharedModel.modelTableBytes,
                    "current_owned":p.currentAllocated.totalBytes,"candidate_planned":p.candidatePlanned.totalBytes,
                    "reserves":p.reserves.totalBytes,"current_allowance":p.reserves.currentMetadataDriverBytes,
                    "candidate_allowance":p.reserves.candidateMetadataDriverBytes,"current_bridge":p.reserves.currentBridgeBytes,
                    "candidate_bridge":p.reserves.candidateBridgeBytes,"json":p.reserves.jsonSnapshotBytes,
                    "replay":p.reserves.replayBytes,"text":p.reserves.textBytes,"application":p.reserves.applicationBytes,
                    "aggregate":p.aggregateRequiredBytes,"transition_budget":p.transitionBudgetBytes,
                    "candidate_constructor":p.candidateConstructorRequiredBytes,"candidate_budget":p.candidateWholeBudgetBytes,
                    "fits_aggregate":p.fitsTransitionBudget,"fits_individual":p.fitsCandidateIndividualBudget,
                    "candidate_allocated_in_plan":p.candidateAllocated] as [String:Any]
            }
            if let c = t.candidate {
                row["actual_candidate_admission"] = c.numericalAdmission
                row["actual_candidate"] = ["model":c.modelInstance,"session":c.sessionInstance,
                    "capacity":c.capacity,"loaded_limit":c.loadedLimit,"offset":c.offset]
            }
            if let h = t.headroom { row["headroom"] = ["required":h.requiredBytes,"available":h.availableBytes,"instrument":h.instrument,"fits":h.fits] }
        }
        let raw = try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]);try check(raw.count<=131072,"Frame cap")
        FileHandle.standardOutput.write(raw+Data([10]))
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected cached Smol model and exclusive output directory") }
        let folder = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let file = output.appendingPathComponent("chat.json"), preferenceDirectory = output.appendingPathComponent("preferences-v1")
        let seed = FriendlyPreferenceStore(directory:preferenceDirectory)
        let qwenNamespace = NativeAcceptedIdentity.Profile.qwen3.namespace
        try seed.save(.init(namespace:qwenNamespace,capacity:128,chunk:8,reply:32,wholeBudget:HubProfile.qwen3.numericalBudget))
        let qwen = try seed.url(qwenNamespace), qwenBytes = try Data(contentsOf:qwen)
        var workspace = ChatWorkspace(); workspace.conversations[0].draft = "Keep this Smol advisory draft."
        let conversation = workspace.selected; try JSONEncoder().encode(workspace).write(to:file,options:.atomic)
        let harness = HarnessStore(enabled:true)
        guard let original = harness.worker else { throw NativeContextFailure("No initial worker") }
        original.close(); try check(original.waitUntilClosed(seconds:30),"Initial close")
        let gate = AdvisoryGate()
        let worker = try NativeWorker(beforeCommandDequeue:{ gate.command($0) },afterPreviewLimits:{ _,_ in gate.pause("limits") },
            beforePreviewPublication:{ _ in gate.pause("publication") },beforeTransitionRejection:{ _,_ in gate.rejected() },
            afterTransitionAcceptance:{ _ in gate.accepted() })
        harness.worker = worker; harness.state = worker.snapshot()
        var writes = 0
        let preferences = FriendlyPreferenceStore(directory:preferenceDirectory,persist:{ data,url in writes += 1; try data.write(to:url,options:.atomic) })
        let chat = FriendlyChatStore(harness:harness,fileURL:file,preferences:preferences)
        var failure: Error?
        do {
            chat.openManaged(folder,profile:.smol,ifSelected:conversation); try idle(harness,chat,worker)
            let loaded = worker.snapshot()
            try check(chat.ready && loaded.acceptedIdentity?.profile == .smol && loaded.sessionConfiguration?.capacity == 512 && writes == 0,"Audited load failed")
            let identity = loaded.acceptedIdentity, baseline = try Data(contentsOf:file)
            try frame("loaded",worker,chat,file,qwen,writes)
            for stage in ["limits","publication"] {
                chat.showPlanning(); chat.editPlanning(.upper,"1024"); chat.editPlanning(.chunk,"16")
                gate.arm(stage); chat.previewPlanning()
                guard gate.reached.wait(timeout:.now()+30) == .success else {
                    FileHandle.standardError.write(Data((String(reflecting:worker.snapshot())+"\n").utf8.prefix(131072)))
                    throw NativeContextFailure("Advisory gate not reached")
                }
                chat.hidePlanning(); gate.proceed.signal(); try idle(harness,chat,worker)
                try preserved(loaded,worker.snapshot())
                try check(worker.snapshot().preview == nil && chat.planning.result == nil && writes == 0,"Stale advisory result published")
                try frame(stage == "limits" ? "stale-after-limits":"stale-before-publication",worker,chat,file,qwen,writes)
            }
            chat.showPlanning(); chat.editPlanning(.upper,"1024"); chat.editPlanning(.chunk,"16"); chat.previewPlanning(); try idle(harness,chat,worker)
            guard let wide = chat.planning.result else { throw NativeContextFailure("No advisory result") }
            try check(wide.isResidentAdvisoryEstimate && wide.request.selection.profile == .smolAudited
                && wide.request.upper == 1024 && wide.capacityRequest?.upper == 512
                && wide.capacity?.capacity.selected?.capacity == 512 && wide.capacity?.resources.numericalBuffersAllocated == false,"Resident bound/estimate provenance")
            try preserved(loaded,worker.snapshot());try frame("bounded-advisory",worker,chat,file,qwen,writes)
            chat.editPlanning(.upper,"256"); chat.previewPlanning(); try idle(harness,chat,worker)
            guard let target = chat.planning.target, let actualEstimate = chat.planning.result else { throw NativeContextFailure("No selected attempt") }
            try check(actualEstimate.capacity?.capacity.selected?.capacity == 256 && chat.planningApplyUnavailableReason == nil,"Selected estimate unavailable")
            var injected = FriendlyPlanningState(); injected.select(target); injected.edit(.upper,"256"); injected.edit(.chunk,"16"); injected.edit(.budgetMiB,"1")
            let syntheticRequest = try injected.begin(); injected.observe(try underestimate(syntheticRequest),busy:false)
            let rejectedAttempt = try FriendlyApplyState.prepare(planning:injected,current:worker.snapshot(),target:target,reply:chat.workspace.outputLimit)
            try check(worker.selectTransition(rejectedAttempt.transition.selection) && worker.submit(.replaceWithEmptySession(rejectedAttempt.transition)),"Underestimate enqueue")
            try idle(harness,chat,worker)
            try check(worker.snapshot().transition?.committed == false && worker.snapshot().transition?.outcome == .rejected
                && worker.snapshot().transition?.status != 0
                && worker.snapshot().transition?.message == "Transition memory budget does not fit" && writes == 0,"Actual D1 did not reject the synthetic underestimate")
            try preserved(loaded,worker.snapshot());try frame("underestimate-rejected",worker,chat,file,qwen,writes,preview:try underestimate(syntheticRequest))
            try check(try Data(contentsOf:file) == baseline && Data(contentsOf:qwen) == qwenBytes,"Preview/rejection wrote user state")
            chat.applyPlanning()
            guard let submitted = chat.applying.request,
                  let accepted = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Apply terminal capture unavailable") }
            // Capture coherent terminal worker evidence before UI reconciliation
            // legitimately retires the transient result when the target changes.
            try frame("accepted-before-reconcile",worker,chat,file,qwen,writes,preview:actualEstimate,
                captured:accepted,submitted:submitted.transition)
            guard let transition = accepted.transition, let configuration = accepted.sessionConfiguration,
                  let candidate = transition.candidate, let plan = transition.plan else { throw NativeContextFailure("Missing captured Apply evidence") }
            try check(!accepted.busy && !accepted.restartRequired && accepted.acceptedIdentity == identity
                && submitted.previewID == actualEstimate.request.id && transition.request == submitted.transition
                && transition.committed && transition.outcome == .committed && transition.status == 0
                && transition.headroom?.fits == true && writes == 0
                && plan.modelInstance == submitted.transition.selection.modelInstance
                && plan.sessionInstance == loaded.sessionConfiguration?.sessionInstance
                && plan.sessionInstance == submitted.transition.selection.sessionInstance
                && candidate.modelInstance == configuration.modelInstance && candidate.sessionInstance == configuration.sessionInstance
                && candidate.capacity == 256 && candidate.loadedLimit == 512 && candidate.offset == 0
                && configuration.acceptedRequestID == submitted.transition.id && configuration.capacity == 256
                && configuration.chunk == 16 && configuration.wholeSessionBudget == submitted.transition.wholeSessionBudget
                && configuration.sessionInstance != loaded.sessionConfiguration?.sessionInstance,"Captured D1/headroom/D2 Apply evidence mismatch")
            harness.poll();chat.observe(harness.state)
            let final = worker.snapshot()
            try check(final.acceptedIdentity == identity && final.sessionConfiguration == configuration
                && chat.currentSessionConfiguration == configuration && chat.applying.committed && !chat.applying.pending
                && writes == 1 && final.history == 0 && final.evaluated == 0 && final.pending == 0 && final.ids.isEmpty,
                "Durable Apply reconciliation or save failed")
            try check(try Data(contentsOf:file) == baseline && Data(contentsOf:qwen) == qwenBytes
                && chat.workspace.current.draft == "Keep this Smol advisory draft." && chat.workspace.current.messages.isEmpty,"Apply changed conversation/draft/Qwen record")
            try check((try preferences.load(NativeAcceptedIdentity.Profile.smol.namespace))?.capacity == 256,"Accepted Smol settings not saved")
            try frame("applied",worker,chat,file,qwen,writes,preview:actualEstimate,submitted:submitted.transition)
        } catch { failure = error }
        gate.proceed.signal(); worker.close(); try check(worker.waitUntilClosed(seconds:60),"Worker close")
        harness.poll();chat.observe(harness.state);try frame("closed",worker,chat,file,qwen,writes)
        if let failure { throw failure }
        let counts = gate.counts;try check(counts.0 == 1 && counts.1 == 1 && counts.2 == 1 && counts.3 == 4,"Unexpected load/transition count")
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:["passed":true,"closed":true,"model_loads":counts.0,
            "advisory_previews":counts.3,"stale_previews":2,"accepted_replacements":counts.1,"synthetic_underestimate_rejections":counts.2,
            "preference_writes":writes,"real_generation_tokens":0,"rendered_UI":false,"phone":false],options:[.sortedKeys])+Data([10]))
    }
}

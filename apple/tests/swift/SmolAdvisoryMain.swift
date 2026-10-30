import Foundation
import CToklet

// Synthetic request/owner/Apply controls. No model, files, device or UI execution.
@main
@MainActor
struct SmolAdvisoryMain {
    static var names: [String] = []
    static func check(_ name: String, _ value: Bool) throws {
        names.append(name); if !value { throw NativeContextFailure(name) }
    }
    static func rejects(_ name: String, _ body: () throws -> Void) throws {
        var rejected = false; do { try body() } catch { rejected = true }; try check(name,rejected)
    }
    static func facts(model: UInt64 = 1, session: UInt64 = 2, upper: UInt64 = 128) throws -> NativeContextFacts {
        var raw = TokletContextInfo()
        raw.abi_version = TOKLET_ABI_VERSION; raw.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        raw.backend = TOKLET_CONTEXT_METAL; raw.model_instance = model; raw.session_instance = session
        raw.engine_global_position_ceiling = 32768; raw.loaded_position_limit = upper; raw.session_capacity = upper
        raw.numerical_and_bridge_admitted_bytes = 80*1024*1024
        return try NativeContextFacts(raw)
    }
    static func result(_ request: NativePreviewRequest, metadataUpper: UInt32 = 512) throws -> NativePreviewResult {
        let derived = try request.capacityRequest(planningUpper:metadataUpper)
        let bytes: UInt64 = 100*1024*1024, budget = request.sessionBudget, fits = bytes <= budget
        let memory: [String:Any] = ["weights_bytes":bytes-7,"kv_bytes":1,"scratch_bytes":1,"history_bytes":1,
            "logits_bytes":1,"metadata_bytes":1,"metadata_allowance_bytes":2,"admitted_bytes":bytes]
        let resources: [String:Any] = ["schema_version":1,"device_name":"synthetic reopened metadata","maximum_buffer_bytes":256*1024*1024,
            "requested_load_context":derived.context,"requested_capacity":derived.upper,"requested_chunk_size":derived.chunk,
            "requested_position_offset":0,"requested_load_budget_bytes":request.loadBudget,"requested_session_budget_bytes":budget,
            "projection":"MlxAffine","normalization":"Compensated","readback":"FullLogits","attention":"Reference",
            "source_payload_bytes":bytes,"source_admitted_load_bytes":request.loadBudget+1,"resident_weights_bytes":bytes-7,
            "resident_admitted_load_bytes":request.loadBudget+1,"session_memory":memory,
            "fits_source_load_budget":false,"fits_resident_load_budget":false,"fits_session_budget":fits,
            "fits_requested_budgets":false,"numerical_buffers_allocated":false,"payload_validation_required":true,"qualification_evaluated":false]
        let point: [String:Any] = ["capacity":fits ? derived.upper:derived.chunk,
            "footprint":["admitted_bytes":bytes,"largest_buffer_bytes":bytes-7],"fits_budget":fits,"fits_buffer":true]
        let wire: [String:Any] = ["resources":resources,"capacity":["minimum":derived.chunk,"maximum":derived.upper,
            "limits":["budget_bytes":budget,"maximum_buffer_bytes":256*1024*1024],
            "selected":fits ? (point as Any):(NSNull() as Any),"first_excluded":fits ? (NSNull() as Any):(point as Any)]]
        let plan = try NativeCapacityPlan.decode(JSONSerialization.data(withJSONObject:wire),request:derived)
        return NativePreviewResult(request:request,outcome:.planned,limits:nil,capacityRequest:derived,capacity:plan,limitation:nil,status:0,message:"")
    }
    static func main() throws {
        let f = try facts(), owner = NSObject(), conversation = UUID()
        let cfg = try NativeSessionConfiguration(context:f,capacity:128,chunk:16,wholeSessionBudget:512*1024*1024)
        let identity = NativeAcceptedIdentity(namespace:NativeAcceptedIdentity.Profile.smol.namespace,assets:[],modelInstance:1,
            loadedLimit:128,sourceRanges:694,sourceReads:694)
        var snapshot = NativeSnapshot(); snapshot.hasModel = true; snapshot.hasSession = true; snapshot.chatEnabled = true
        snapshot.liveContext = f; snapshot.modelContext = NativeModelContext(f); snapshot.sessionConfiguration = cfg; snapshot.acceptedIdentity = identity
        let target = FriendlyPlanningTarget(worker:ObjectIdentifier(owner),modelInstance:1,conversation:conversation,
            modelKey:"smol",directory:"/synthetic/reopened",profile:.smolAudited,wholeBudget:cfg.wholeSessionBudget,
            textBudget:256*1024*1024,configuration:cfg,coexistenceBudget:2304*1024*1024,residentUpper:128)
        var planning = FriendlyPlanningState(); planning.select(target); planning.edit(.upper,"1024")
        let request = try planning.begin()
        try check("explicit-mode-keeps-audited-profile",request.estimator == .residentAdvisory && request.selection.profile == .smolAudited)
        try check("exact-resident-correlation",request.matchesResident(snapshot))
        let derived = try request.capacityRequest(planningUpper:32768)
        try check("loaded-upper-not-reopened-declaration",derived.upper == 128 && derived.context == 128 && request.upper == 1024)
        try check("smaller-file-upper-only-narrows",try request.capacityRequest(planningUpper:64).upper == 64)
        var defaultRequest = request; defaultRequest.estimator = .sourceProfile; defaultRequest.resident = nil
        try defaultRequest.validate()
        try check("default-audited-profile-unchanged",defaultRequest.estimator == .sourceProfile && defaultRequest.selection.profile == .smolAudited)
        var ordinary = defaultRequest
        ordinary = NativePreviewRequest(id:UUID(),selection:.init(id:UUID(),directory:"/synthetic",profile:.ordinary),upper:1024,chunk:16,offset:0,contextCeiling:nil,loadBudget:512*1024*1024,sessionBudget:512*1024*1024)
        try check("ordinary-factory-upper-unchanged",try ordinary.capacityRequest(planningUpper:2048).upper == 1024)
        for variant in 0..<8 {
            var bad = request
            switch variant {
            case 0: bad.resident = nil
            case 1: bad.resident = .init(modelInstance:1,sessionInstance:1,loadedUpper:128)
            case 2: bad.resident = .init(modelInstance:1,sessionInstance:2,loadedUpper:0)
            case 3: bad.resident = .init(modelInstance:1,sessionInstance:2,loadedUpper:513)
            case 4: bad.normalization = .original
            case 5: bad.readback = .greedyToken
            case 6: bad.attention = .online
            default: bad.estimator = .sourceProfile
            }
            try rejects("invalid-mode-\(variant)") { try bad.validate() }
        }
        var foreignMode = ordinary; foreignMode.estimator = .residentAdvisory; foreignMode.resident = request.resident
        try rejects("ordinary-cannot-claim-smol-estimator") { try foreignMode.validate() }
        for variant in 0..<8 {
            var wrong = snapshot
            switch variant {
            case 0: wrong.acceptedIdentity = nil
            case 1: wrong.acceptedIdentity = .init(namespace:NativeAcceptedIdentity.Profile.qwen3.namespace,assets:[],modelInstance:1,loadedLimit:128,sourceRanges:694,sourceReads:694)
            case 2: wrong.liveContext = try facts(session:3)
            case 3: wrong.modelContext = NativeModelContext(try facts(model:3,session:4))
            case 4: wrong.modelContext = NativeModelContext(try facts(upper:512))
            case 5: wrong.sessionConfiguration = nil
            case 6: wrong.restartRequired = true
            default: wrong.hasSession = false
            }
            try check("stale-owner-\(variant)",!request.matchesResident(wrong))
        }
        let plan = try result(request); planning.observe(plan,busy:false)
        let accepted = try FriendlyApplyState.prepare(planning:planning,current:snapshot,target:target,reply:64)
        try check("apply-exact-selected-point",accepted.transition.capacity == 128 && accepted.transition.chunk == 16)
        try check("bridge-added-once",accepted.transition.wholeSessionBudget == target.wholeBudget)
        try check("actual-admission-policy-retained",accepted.transition.memoryPolicy == .friendly
            && accepted.transition.applicationReserve == NativeTransitionHeadroomCheck.applicationReserve
            && accepted.transition.coexistenceBudget == target.coexistenceBudget && accepted.transition.textReserve == target.textBudget)
        try check("loading-fit-is-not-replacement-gate",plan.capacity?.resources.fitsSourceLoadBudget == false && plan.capacity?.resources.fitsResidentLoadBudget == false)
        try check("estimate-not-owner-or-allocation-proof",plan.isResidentAdvisoryEstimate && plan.capacity?.resources.numericalBuffersAllocated == false && snapshot.acceptedIdentity == identity)
        try rejects("reply-equal-capacity") { _ = try FriendlyApplyState.prepare(planning:planning,current:snapshot,target:target,reply:128) }
        var differentSession = snapshot; differentSession.liveContext = try facts(session:3)
        try rejects("apply-stale-session") { _ = try FriendlyApplyState.prepare(planning:planning,current:differentSession,target:target,reply:64) }
        // A later file read may narrow the estimate, but cannot change resident
        // identity or bypass its final model-bound transaction admission.
        var changedFiles = planning; changedFiles.observe(try result(request,metadataUpper:64),busy:false)
        let changedAttempt = try FriendlyApplyState.prepare(planning:changedFiles,current:snapshot,target:target,reply:32)
        try check("file-change-is-advisory-only",changedAttempt.transition.capacity == 64 && snapshot.acceptedIdentity == identity && snapshot.sessionConfiguration == cfg)
        var edited = planning; edited.edit(.chunk,"8")
        try rejects("edited-request-loses-authority") { _ = try FriendlyApplyState.prepare(planning:edited,current:snapshot,target:target,reply:64) }
        var zero = request
        // Budgets are immutable in a request; construct the explicit empty-fit witness.
        zero = NativePreviewRequest(id:UUID(),selection:request.selection,upper:128,chunk:16,offset:0,contextCeiling:nil,loadBudget:512*1024*1024,sessionBudget:0)
        zero.estimator = .residentAdvisory; zero.resident = request.resident
        try check("zero-budget-no-selection",try result(zero).capacity?.capacity.selected == nil)
        guard names.count == 33 else { throw NativeContextFailure("Unexpected advisory control count") }
        let raw = try JSONSerialization.data(withJSONObject:["passed":true,"controls":names,"count":names.count,"model_loaded":false,"actual_D1_D2_executed":false],options:[.sortedKeys])
        FileHandle.standardOutput.write(raw+Data([10]))
    }
}

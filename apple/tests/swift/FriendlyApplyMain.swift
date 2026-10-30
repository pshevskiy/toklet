import Foundation
import CToklet

// Synthetic UI authority/state controls. No model, UI rendering or generation.
@main
@MainActor
struct FriendlyApplyMain {
    static var checks = 0
    static func require(_ value: Bool, _ message: String) throws {
        checks += 1
        if !value { throw NativeContextFailure(message) }
    }
    static func rejects(_ body: () throws -> Void) throws {
        var failed = false
        do { try body() } catch { failed = true }
        try require(failed,"Stale or invalid Apply authority accepted")
    }
    static func facts(session: UInt64 = 2, capacity: UInt64 = 512) throws -> NativeContextFacts {
        var raw = TokletContextInfo()
        raw.abi_version = TOKLET_ABI_VERSION; raw.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        raw.model_instance = 1; raw.session_instance = session; raw.backend = TOKLET_CONTEXT_METAL
        raw.engine_global_position_ceiling = 32768; raw.loaded_position_limit = 512; raw.session_capacity = capacity
        raw.numerical_and_bridge_admitted_bytes = 80*1024*1024
        return try NativeContextFacts(raw)
    }
    // Decode a genuine-shaped wire report, deliberately with BOTH loading
    // flags false. Existing resident replacement only uses selected session fit.
    static func plan(_ request: NativePreviewRequest, selected: UInt32 = 256) throws -> NativePreviewResult {
        let capacityRequest = try request.capacityRequest(planningUpper:1024)
        let bytes: UInt64 = 100*1024*1024, budget = request.sessionBudget
        let memory: [String:Any] = ["weights_bytes":bytes-7,"kv_bytes":1,"scratch_bytes":1,"history_bytes":1,
            "logits_bytes":1,"metadata_bytes":1,"metadata_allowance_bytes":2,"admitted_bytes":bytes]
        let resources: [String:Any] = ["schema_version":1,"device_name":"synthetic","maximum_buffer_bytes":budget,
            "requested_load_context":1024,"requested_capacity":request.upper,"requested_chunk_size":request.chunk,
            "requested_position_offset":0,"requested_load_budget_bytes":budget,"requested_session_budget_bytes":budget,
            "projection":"MlxAffine","normalization":"Compensated","readback":"FullLogits","attention":"Reference",
            "source_payload_bytes":bytes,"source_admitted_load_bytes":budget+1,"resident_weights_bytes":bytes-7,
            "resident_admitted_load_bytes":budget+1,"session_memory":memory,
            "fits_source_load_budget":false,"fits_resident_load_budget":false,"fits_session_budget":true,
            "fits_requested_budgets":false,"numerical_buffers_allocated":false,"payload_validation_required":true,
            "qualification_evaluated":false]
        let point: [String:Any] = ["capacity":selected,"footprint":["admitted_bytes":bytes,"largest_buffer_bytes":bytes-7],
            "fits_budget":true,"fits_buffer":true]
        let wire: [String:Any] = ["resources":resources,"capacity":["minimum":request.chunk,"maximum":request.upper,
            "limits":["budget_bytes":budget,"maximum_buffer_bytes":budget],"selected":point,"first_excluded":NSNull()]]
        let value = try NativeCapacityPlan.decode(JSONSerialization.data(withJSONObject:wire),request:capacityRequest)
        return NativePreviewResult(request:request,outcome:.planned,limits:nil,capacityRequest:capacityRequest,
            capacity:value,limitation:nil,status:0,message:"")
    }
    static func main() throws {
        let owner = NSObject(), other = NSObject(), chat = UUID(), currentFacts = try facts()
        let configuration = try NativeSessionConfiguration(context:currentFacts,capacity:512,chunk:32,wholeSessionBudget:512*1024*1024)
        let target = FriendlyPlanningTarget(worker:ObjectIdentifier(owner),modelInstance:1,conversation:chat,
            modelKey:"resident",directory:"/unchanged/model",profile:.ordinary,wholeBudget:configuration.wholeSessionBudget,
            textBudget:128*1024*1024,configuration:configuration,coexistenceBudget:2304*1024*1024)
        var current = NativeSnapshot(); current.hasModel = true; current.hasSession = true; current.chatEnabled = true
        current.liveContext = currentFacts; current.modelContext = NativeModelContext(currentFacts); current.sessionConfiguration = configuration
        var planning = FriendlyPlanningState(); planning.select(target)
        planning.edit(.upper,"256"); planning.edit(.chunk,"16"); planning.edit(.budgetMiB,"768")
        let request = try planning.begin(); let result = try plan(request)
        planning.observe(result,busy:false)
        let candidate = try FriendlyApplyState.prepare(planning:planning,current:current,target:target,reply:128)
        try require(candidate.previewID == request.id && candidate.transition.capacity == 256 && candidate.transition.chunk == 16,
                    "Displayed estimate was not used exactly")
        try require(candidate.transition.wholeSessionBudget == 768*1024*1024
                    && candidate.transition.coexistenceBudget == target.coexistenceBudget
                    && candidate.transition.textReserve == target.textBudget && candidate.transition.replayReserve == 0
                    && candidate.transition.applicationReserve == NativeTransitionHeadroomCheck.applicationReserve
                    && candidate.transition.memoryPolicy == .friendly,"Budget/reserve/profile routing changed")
        try require(result.capacity?.resources.fitsSourceLoadBudget == false
                    && result.capacity?.resources.fitsResidentLoadBudget == false,"Loading-fit negative witness lost")
        try require(candidate.transition.selection.modelInstance == 1 && candidate.transition.selection.sessionInstance == 2,
                    "Apply did not bind current native owners")
        try rejects { _ = try FriendlyApplyState.prepare(planning:planning,current:current,target:target,reply:256) }
        try rejects { _ = try FriendlyApplyState.prepare(planning:planning,current:current,target:target,reply:0) }
        for change in 0..<5 {
            var stale = current
            switch change { case 0: stale.busy = true; case 1: stale.closed = true; case 2: stale.restartRequired = true
            case 3: stale.sessionConfiguration = nil; default: stale.liveContext = try facts(session:3) }
            try rejects { _ = try FriendlyApplyState.prepare(planning:planning,current:stale,target:target,reply:128) }
        }
        let foreign = FriendlyPlanningTarget(worker:ObjectIdentifier(other),modelInstance:1,conversation:chat,
            modelKey:target.modelKey,directory:target.directory,profile:.ordinary,wholeBudget:target.wholeBudget,
            textBudget:target.textBudget,configuration:configuration,coexistenceBudget:target.coexistenceBudget)
        try rejects { _ = try FriendlyApplyState.prepare(planning:planning,current:current,target:foreign,reply:128) }
        var edited = planning; edited.edit(.budgetMiB,"769")
        try rejects { _ = try FriendlyApplyState.prepare(planning:edited,current:current,target:target,reply:128) }
        var staleRequest = planning; let different = try staleRequest.begin()
        staleRequest.observe(result,busy:false)
        try require(different.id != request.id && staleRequest.result == nil,"Old result UUID resurrected")
        try rejects { _ = try FriendlyApplyState.prepare(planning:staleRequest,current:current,target:target,reply:128) }
        var large = planning; large.edit(.upper,"1024"); let largeRequest = try large.begin()
        large.observe(try plan(largeRequest,selected:1024),busy:false)
        try rejects { _ = try FriendlyApplyState.prepare(planning:large,current:current,target:target,reply:128) }
        // Even an internally malformed point cannot authorize Apply. Wire
        // report decoding has its own controls; this checks the UI boundary.
        for variant in 0..<3 {
            let originalPlan = result.capacity!
            let point = NativeCapacityPlan.Point(capacity:256,
                footprint:NativeCapacityPlan.Footprint(admittedBytes:100*1024*1024,largestBufferBytes:100*1024*1024-7),
                fitsBudget:variant != 0,fitsBuffer:variant != 1)
            let altered = NativeCapacityPlan(resources:originalPlan.resources,
                capacity:NativeCapacityPlan.Range(minimum:16,maximum:256,limits:originalPlan.capacity.limits,
                    selected:variant == 2 ? nil:point,firstExcluded:nil))
            var bad = planning
            bad.observe(NativePreviewResult(request:request,outcome:.planned,limits:nil,capacityRequest:result.capacityRequest,
                capacity:altered,limitation:nil,status:0,message:""),busy:false)
            try rejects { _ = try FriendlyApplyState.prepare(planning:bad,current:current,target:target,reply:128) }
        }
        let auditedTarget = FriendlyPlanningTarget(worker:ObjectIdentifier(owner),modelInstance:1,conversation:chat,
            modelKey:target.modelKey,directory:target.directory,profile:.smolAudited,wholeBudget:target.wholeBudget,
            textBudget:target.textBudget,configuration:configuration,coexistenceBudget:target.coexistenceBudget)
        var audited = FriendlyPlanningState(); audited.select(auditedTarget); audited.edit(.upper,"256"); audited.edit(.chunk,"16")
        let auditedRequest = try audited.begin(); audited.observe(try plan(auditedRequest),busy:false)
        try rejects { _ = try FriendlyApplyState.prepare(planning:audited,current:current,target:auditedTarget,reply:128) }
        var state = FriendlyApplyState(); state.begin(candidate)
        var rejected = current
        rejected.transition = NativeTransitionResult(request:candidate.transition,outcome:.rejected,committed:false,
            plan:nil,candidate:nil,status:1,message:"Insufficient fresh headroom")
        state.observe(target:target,snapshot:rejected)
        try require(!state.pending && !state.committed && state.message == "Insufficient fresh headroom"
                    && planning.draft.upper == "256" && planning.draft.chunk == "16" && planning.draft.budgetMiB == "768",
                    "Rejection replaced edited values or reported success")
        let nextFacts = try facts(session:4,capacity:256)
        let nextConfiguration = try NativeSessionConfiguration(context:nextFacts,capacity:256,chunk:16,
            wholeSessionBudget:768*1024*1024,acceptedRequestID:candidate.transition.id)
        var next = current; next.liveContext = nextFacts; next.modelContext = NativeModelContext(nextFacts)
        next.sessionConfiguration = nextConfiguration; next.transition = nil
        var nextTarget = target; nextTarget.configuration = nextConfiguration
        state.begin(candidate); state.observe(target:nextTarget,snapshot:next)
        try require(state.committed && !state.pending,"Late cancellation erased durable adoption")
        state.begin(candidate); next.restartRequired = true; state.observe(target:nextTarget,snapshot:next)
        try require(state.committed && state.message.contains("Restart"),"Committed cleanup failure presented rollback")
        next.restartRequired = false
        state.begin(candidate); state.observe(target:target,snapshot:current)
        try require(!state.committed && !state.pending && state.message.contains("unchanged"),"Precommit cancellation mutated configuration")
        state.begin(candidate); state.observe(target:foreign,snapshot:next)
        try require(!state.committed && !state.pending && state.message.isEmpty,"Foreign worker inherited acceptance")
        state.begin(candidate); state.observe(target:nextTarget,snapshot:next); state.observe(target:nil,snapshot:current)
        try require(!state.committed && state.message.isEmpty,"Dismissed owner leaked a committed notice")
        var wrong = next
        wrong.sessionConfiguration = try NativeSessionConfiguration(context:nextFacts,capacity:256,chunk:16,
            wholeSessionBudget:768*1024*1024,acceptedRequestID:UUID())
        state.begin(candidate); state.observe(target:nextTarget,snapshot:wrong)
        try require(!state.committed,"Unrelated accepted UUID counted as this Apply")
        var accepted = FriendlyPlanningLoad(worker:ObjectIdentifier(owner),settings:NativeSettings(directory:target.directory),
            coexistenceBudget:target.coexistenceBudget)
        let original = accepted.settings
        accepted.complete(worker:ObjectIdentifier(owner),instance:1,chatEnabled:true)
        let promoted = accepted.target(worker:ObjectIdentifier(owner),instance:1,conversation:chat,
            loadedKey:"resident",selectedKey:"resident",configuration:nextConfiguration)
        try require(promoted?.configuration == nextConfiguration && promoted?.wholeBudget == nextConfiguration.wholeSessionBudget
                    && promoted?.coexistenceBudget == target.coexistenceBudget
                    && String(reflecting:accepted.settings) == String(reflecting:original),"Apply overwrote immutable load provenance")
        print("FRIENDLY_APPLY_CONTROLS \(checks) passed")
    }
}

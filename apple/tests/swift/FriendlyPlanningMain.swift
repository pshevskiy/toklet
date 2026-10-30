import Foundation
import CToklet

@main
@MainActor
struct FriendlyPlanningMain {
    static var checks = 0
    static func require(_ ok: Bool, _ message: String) throws {
        checks += 1
        if !ok { throw NativeContextFailure(message) }
    }
    static func rejects(_ body: () throws -> Void) throws {
        var rejected = false
        do { try body() } catch { rejected = true }
        try require(rejected,"Invalid draft accepted")
    }
    static func reply(_ request: NativePreviewRequest, failed: Bool = false) -> NativePreviewResult {
        NativePreviewResult(request:request,outcome:failed ? .failed:.metadataOnly,limits:nil,
            capacityRequest:nil,capacity:nil,limitation:failed ? nil:.backendUnavailable,
            status:failed ? 1:0,message:failed ? "synthetic stale error":"")
    }
    static func sameFunctionalSnapshot(_ before: NativeSnapshot, _ after: NativeSnapshot) -> Bool {
        // Selection invalidation advances polling revision even when preview is
        // already nil. Every other field, including preview, remains compared.
        var lhs = before, rhs = after
        lhs.revision = 0; rhs.revision = 0
        return String(reflecting:lhs) == String(reflecting:rhs)
    }
    static func main() throws {
        let one = NSObject(), two = NSObject(), conversation = UUID()
        func target(worker: NSObject? = nil, instance: UInt64 = 7, chat: UUID? = nil,
                    key: String = "same-model", directory: String = "/local/model") -> FriendlyPlanningTarget {
            FriendlyPlanningTarget(worker:ObjectIdentifier(worker ?? one),modelInstance:instance,
                conversation:chat ?? conversation,modelKey:key,directory:directory,profile:.ordinary,
                wholeBudget:512 * 1024 * 1024,textBudget:64 * 1024 * 1024)
        }
        var acceptedSettings = NativeSettings(directory:"/local/non-security-scoped/model")
        acceptedSettings.numericalBudget = 1024 * 1024 * 1024
        acceptedSettings.textBudget = 128 * 1024 * 1024
        var accepted = FriendlyPlanningLoad(worker:ObjectIdentifier(one),settings:acceptedSettings)
        func loadedTarget(_ value: FriendlyPlanningLoad, worker: NSObject? = nil, instance: UInt64 = 7,
                          selectedKey: String = "model") -> FriendlyPlanningTarget? {
            value.target(worker:ObjectIdentifier(worker ?? one),instance:instance,conversation:conversation,
                loadedKey:"model",selectedKey:selectedKey)
        }
        try require(loadedTarget(accepted) == nil,"Enqueued load was treated as loaded")
        accepted.complete(worker:ObjectIdentifier(two),instance:7,chatEnabled:true)
        try require(loadedTarget(accepted) == nil,"Wrong worker promoted accepted load")
        accepted.complete(worker:ObjectIdentifier(one),instance:0,chatEnabled:true)
        try require(loadedTarget(accepted) == nil,"Zero model instance promoted")
        accepted.complete(worker:ObjectIdentifier(one),instance:7,chatEnabled:false)
        try require(loadedTarget(accepted) == nil,"Failed Chat load promoted")
        accepted.complete(worker:ObjectIdentifier(one),instance:7,chatEnabled:true)
        try require(loadedTarget(accepted)?.directory == acceptedSettings.directory
                    && loadedTarget(accepted)?.wholeBudget == acceptedSettings.numericalBudget
                    && loadedTarget(accepted)?.textBudget == acceptedSettings.textBudget
                    && loadedTarget(accepted)?.profile == .ordinary,"Accepted input/profile lost or guessed")
        try require(loadedTarget(accepted,worker:two) == nil && loadedTarget(accepted,instance:8) == nil
                    && loadedTarget(accepted,selectedKey:"different") == nil,"Worker/model/conversation model mismatch retained load target")
        let selection = NativePreviewSelection(id:UUID(),directory:"/local/model",profile:.ordinary)
        let draft = FriendlyPlanningDraft()
        let request = try draft.request(selection:selection)
        try require(request.upper == 512 && request.chunk == 32 && request.offset == 0 && request.contextCeiling == nil,"Default draft changed Chat policy or added a ceiling")
        try require(request.loadBudget == 512 * 1024 * 1024 - UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES)
                    && request.sessionBudget == request.loadBudget,"Bridge allowance not subtracted exactly once")
        for bad in ["","0","-1","+1"," 1","1 ","1.5","NaN","١","18446744073709551616"] {
            var value = draft; value.budgetMiB = bad
            try rejects { _ = try value.request(selection:selection) }
        }
        for bad in ["6145","18446744073709551615"] {
            var value = draft; value.budgetMiB = bad
            try rejects { _ = try value.request(selection:selection) }
        }
        for bad in ["0","32769","4294967296"] {
            var value = draft; value.upper = bad
            try rejects { _ = try value.request(selection:selection) }
        }
        for bad in ["0","513","1025"] {
            var value = draft; value.chunk = bad
            try rejects { _ = try value.request(selection:selection) }
        }
        var edge = draft; edge.upper = "32768"; edge.chunk = "1024"; edge.budgetMiB = "6144"
        let maximum = try edge.request(selection:selection)
        try require(maximum.loadBudget == 6 * 1024 * 1024 * 1024 - UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES),"Maximum whole budget rejected or changed")
        edge.upper = "2048"; edge.chunk = "32"
        let hypothetical = try edge.request(selection:selection)
        let clamped = try hypothetical.capacityRequest(planningUpper:1024)
        try require(hypothetical.upper == 2048 && clamped.upper == 1024 && clamped.context == 1024
                    && NativeSettings.chatContextPositions == 512,"Draft/effective/current capacities conflated")
        var state = FriendlyPlanningState()
        try rejects { _ = try state.begin() }
        try require(state.select(target()),"Initial target not selected")
        let initialSelection = state.selection
        try require(!state.select(target()) && state.selection == initialSelection,"Same loaded target changed identity")
        let first = try state.begin()
        state.observe(nil,busy:true)
        try require(state.pending && state.request == first,"Pending nil snapshot was cleared")
        state.observe(reply(first),busy:false)
        try require(state.result?.request == first && !state.pending,"Current result was not accepted")
        // An ordinary accepted command clears native preview even while busy.
        state.observe(nil,busy:true)
        try require(state.result == nil && state.request == nil,"Ordinary work retained an old completed preview")
        for field in [FriendlyPlanningDraft.Field.upper,.chunk,.budgetMiB] {
            for failed in [false,true] {
                let old = try state.begin()
                state.edit(field,field == .chunk ? "32":"512")
                state.observe(reply(old,failed:failed),busy:false)
                try require(state.result == nil && state.request == nil,"Edit resurrected stale result/error")
            }
        }
        state.edit(.upper,String(repeating:"9",count:1000))
        try require(state.draft.upper.count == 64,"Pasted input not bounded")
        try rejects { _ = try state.begin() }
        state.edit(.upper,"1" + String(repeating:"\u{0301}",count:10000))
        try require(state.draft.upper.utf8.count <= 64,"One combining cluster exceeded the byte bound")
        try rejects { _ = try state.begin() }
        _ = state.select(nil); _ = state.select(target())
        for changed in [target(worker:two),target(instance:8),target(chat:UUID()),target(key:"other"),target(directory:"/other/model")] {
            let old = try state.begin(); _ = state.select(changed)
            state.observe(reply(old),busy:false)
            try require(state.result == nil && state.request == nil && state.selection?.id != old.selection.id,"Target change retained stale result")
            _ = state.select(target())
        }
        for failed in [false,true] {
            let old = try state.begin(); state.invalidate()
            state.observe(reply(old,failed:failed),busy:false)
            try require(state.result == nil && !state.pending,"Dismissal/lifecycle invalidation resurrected result")
        }
        let old = try state.begin(), next = try state.begin()
        state.observe(reply(old),busy:true)
        try require(state.result == nil && state.request == next,"Earlier same-target request replaced newer request")
        state.observe(reply(next),busy:false)
        try require(state.result?.request == next,"Current same-target request not accepted")
        // Full request equality, not UUID alone, binds the displayed result.
        state.invalidate(); let exact = try state.begin()
        var wrong = exact; wrong.readback = .greedyToken
        state.observe(reply(wrong),busy:false)
        try require(state.result == nil,"Mismatched request mode accepted")
        // Actual store/worker negative boundary: no loaded target, no model
        // inspection or ordinary command may be dispatched; workspace stays exact.
        let harness = HarnessStore(enabled:true)
        guard let worker = harness.worker else { throw NativeContextFailure("Missing test worker") }
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("toklet-preview-" + UUID().uuidString + ".json")
        let chat = FriendlyChatStore(harness:harness,fileURL:file)
        chat.workspace.edit("Preserve my unsaved message")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let before = try encoder.encode(chat.workspace)
        let snapshot = worker.snapshot()
        chat.showPlanning(); chat.editPlanning(.upper,"2048"); chat.previewPlanning()
        try require(!chat.planning.message.isEmpty && chat.planning.target == nil,"Unloaded store offered a preview")
        chat.invalidatePlanning(); chat.hidePlanning(); chat.observe(worker.snapshot())
        let after = worker.snapshot()
        let witness: [String:Any] = ["event":"unloaded-preview-state","before":String(reflecting:snapshot),
            "after":String(reflecting:after),"before_revision":snapshot.revision,"after_revision":after.revision]
        let witnessBytes = try JSONSerialization.data(withJSONObject:witness,options:[.sortedKeys])
        FileHandle.standardError.write(witnessBytes + Data([10]))
        var corrupted = after; corrupted.message = after.message + " injected functional mutation"
        try require(snapshot.preview == nil && after.preview == nil && after.revision >= snapshot.revision
                    && sameFunctionalSnapshot(snapshot,after) && !sameFunctionalSnapshot(snapshot,corrupted),
                    "Preview-only UI boundary changed actual worker state or comparator missed functional mutation")
        try require(try encoder.encode(chat.workspace) == before && !FileManager.default.fileExists(atPath:file.path),"Preview changed or persisted conversations/drafts")
        print("{\"passed\":true,\"checks\":\(checks),\"model_or_gpu_execution\":false,\"ui_execution\":false,\"loaded_ui_binding_qualified\":false}")
    }
}

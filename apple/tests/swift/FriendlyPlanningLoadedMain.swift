import Foundation
import CryptoKit

// Explicit Mac-only cached-model integration. No generation, download or UI.
@main
@MainActor
struct FriendlyPlanningLoadedMain {
    static func require(_ ok: Bool, _ message: String) throws {
        if !ok { throw NativeContextFailure(message) }
    }
    static func emit(_ value: [String:Any]) throws {
        let bytes = try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys])
        FileHandle.standardOutput.write(bytes + Data([10]))
    }
    static func digest(_ value: Data) -> String { SHA256.hash(data:value).map { String(format:"%02x",$0) }.joined() }
    static func engineSignature(_ state: NativeSnapshot) -> String {
        var copy = state; copy.revision = 0; copy.preview = nil
        return String(reflecting:copy)
    }
    static func settle(_ harness: HarnessStore, _ chat: FriendlyChatStore, _ worker: NativeWorker) throws {
        guard worker.waitUntilIdle(seconds:120) != nil else { throw NativeContextFailure("Worker did not reach idle") }
        harness.poll(); chat.observe(harness.state)
    }
    static func frame(_ name: String, _ worker: NativeWorker, _ chat: FriendlyChatStore, writes: Int) throws {
        let state = worker.snapshot()
        var row: [String:Any] = ["event":name,"status":state.status,"message":state.message,
            "busy":state.busy,"hasModel":state.hasModel,"chatEnabled":state.chatEnabled,
            "worker_snapshot":String(reflecting:state),"persistence_calls":writes,
            "ui_pending":chat.planning.pending,"ui_result":String(reflecting:chat.planning.result)]
        if let live = state.liveContext {
            row["model_instance"] = live.modelInstance; row["session_instance"] = live.sessionInstance
            row["active_capacity"] = live.capacity; row["active_offset"] = live.offset
            row["history"] = live.history; row["committed"] = live.committed; row["pending"] = live.pending
        }
        if let plan = state.runtimePlan { row["active_chunk"] = plan.requestedChunkSize }
        if let result = state.preview {
            row["preview_request_id"] = result.request.id.uuidString
            row["preview_selection_id"] = result.request.selection.id.uuidString
            row["preview_directory"] = result.request.selection.directory
            row["preview_profile"] = result.request.selection.profile.rawValue
            row["preview_requested_upper"] = result.request.upper; row["preview_chunk"] = result.request.chunk
            row["preview_offset"] = result.request.offset; row["preview_status"] = result.status
            row["preview_outcome"] = result.outcome.rawValue
            row["preview_load_budget"] = result.request.loadBudget; row["preview_session_budget"] = result.request.sessionBudget
            if let request = result.capacityRequest { row["effective_upper"] = request.upper }
            if let selected = result.capacity?.capacity.selected { row["estimated_capacity"] = selected.capacity }
        }
        try emit(row)
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected pinned model directory and fresh output directory") }
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let file = output.appendingPathComponent("history.json")
        let harness = HarnessStore(enabled:true)
        guard let worker = harness.worker else { throw NativeContextFailure("Missing worker") }
        var writes = 0
        let chat = FriendlyChatStore(harness:harness,fileURL:file,persist:{ data,url in
            writes += 1; try data.write(to:url,options:.atomic)
        })
        var failure: Error?
        var completed = 0
        do {
            chat.open(model)
            try settle(harness,chat,worker)
            try frame("loaded",worker,chat,writes:writes)
            let baseline = worker.snapshot()
            try require(chat.ready && baseline.status == 0 && baseline.chatEnabled,"Pinned Chat load failed: \(baseline.status) \(baseline.message)")
            guard let live = baseline.liveContext else { throw NativeContextFailure("Missing active context") }
            try require(live.capacity == 512 && live.offset == 0 && baseline.runtimePlan?.requestedChunkSize == 32,"Active Chat defaults changed")
            // Flush/cancel the existing debounce before establishing persistence
            // witnesses. No timer may race the preview write-count assertions.
            chat.edit("Keep this unsent draft exactly.")
            try require(chat.save(),"Could not establish saved workspace baseline")
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let workspace = try encoder.encode(chat.workspace), persisted = try Data(contentsOf:file), baselineWrites = writes
            let signature = engineSignature(baseline)
            chat.showPlanning()
            try require(chat.planning.target?.modelInstance == live.modelInstance
                        && chat.planning.target?.directory == model.path
                        && chat.planning.target?.profile == .ordinary,"Accepted loaded target was not promoted")
            for upper in [UInt32(512),256] {
                chat.editPlanning(.upper,String(upper))
                try require(chat.planning.result == nil && worker.snapshot().preview == nil,"Editing did not invalidate the previous preview")
                chat.previewPlanning()
                let submitted = chat.planning.request
                try settle(harness,chat,worker)
                try frame("preview-\(upper)",worker,chat,writes:writes)
                let actual = worker.snapshot()
                guard let observed = actual.preview, let displayed = chat.planning.result else { throw NativeContextFailure("Missing completed preview: \(chat.planning.message)") }
                try require(observed.request == submitted && displayed.request == observed.request
                            && observed.status == 0 && observed.outcome == .planned
                            && observed.request.upper == upper && observed.capacityRequest?.upper == upper,
                            "Actual worker result did not match the complete UI request")
                try require(actual.liveContext?.modelInstance == live.modelInstance
                            && actual.liveContext?.sessionInstance == live.sessionInstance
                            && actual.liveContext?.capacity == 512 && actual.runtimePlan?.requestedChunkSize == 32,
                            "Preview replaced or resized the active session")
                try require(engineSignature(actual) == signature,"Preview changed ordinary engine/generation state")
                try require(try encoder.encode(chat.workspace) == workspace && Data(contentsOf:file) == persisted
                            && writes == baselineWrites,"Preview changed workspace or persistence")
                completed += 1
            }
            chat.hidePlanning()
            try frame("hidden",worker,chat,writes:writes)
            try require(chat.planning.request == nil && chat.planning.result == nil && worker.snapshot().preview == nil,
                        "Dismissal retained a preview")
            try require(engineSignature(worker.snapshot()) == signature,"Dismissal changed the loaded session")
            try require(try encoder.encode(chat.workspace) == workspace && Data(contentsOf:file) == persisted
                        && writes == baselineWrites,"Dismissal changed persistence")
            try emit(["event":"workspace","before_sha256":digest(workspace),"after_sha256":digest(try encoder.encode(chat.workspace)),
                      "persisted_before_sha256":digest(persisted),"persisted_after_sha256":digest(try Data(contentsOf:file)),
                      "baseline_persistence_calls":baselineWrites,"final_persistence_calls":writes])
        } catch { failure = error }
        worker.close()
        let closed = worker.waitUntilClosed(seconds:60)
        harness.poll(); chat.observe(harness.state)
        try emit(["event":"summary","passed":failure == nil && closed && completed == 2,"completed_previews":completed,
                  "closed":closed,"error":failure.map { $0.localizedDescription } ?? "",
                  "generation_executed":false,"ui_rendered":false,"whole_device_fit_qualified":false])
        if let failure { throw failure }
        try require(closed && completed == 2,"Worker close or preview count failed")
    }
}

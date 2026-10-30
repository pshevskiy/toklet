import Foundation
import CryptoKit

// One cached, pinned ordinary Qwen3 load. Controlled worker dequeue only; no
// memory-admission override, download, successful generation or rendered UI.
private final class ChatDequeueGate: @unchecked Sendable {
    private let entered = DispatchSemaphore(value:0)
    private let released = DispatchSemaphore(value:0)
    private let lock = NSLock()
    private var expired = false
    func pause(_ command: NativeCommand) {
        guard case .chat = command else { return }
        entered.signal()
        if released.wait(timeout:.now()+30) != .success { lock.lock(); expired = true; lock.unlock() }
    }
    func wait() -> Bool { entered.wait(timeout:.now()+30) == .success }
    func resume() { released.signal() }
    var timedOut: Bool { lock.lock(); defer { lock.unlock() }; return expired }
}

@main
@MainActor
struct FriendlyApplyLoadedMain {
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func emit(_ value: [String:Any]) throws {
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]) + Data([10]))
    }
    static func digest(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    static func settle(_ harness: HarnessStore, _ chat: FriendlyChatStore, _ worker: NativeWorker) throws {
        guard worker.waitUntilIdle(seconds:120) != nil else { throw NativeContextFailure("Worker idle deadline") }
        harness.poll(); chat.observe(harness.state)
    }
    static func frame(_ label: String, _ worker: NativeWorker, _ chat: FriendlyChatStore, _ writes: Int) throws {
        let snapshot = worker.snapshot()
        var row: [String:Any] = ["event":label,"worker_snapshot":String(reflecting:snapshot),"configuration":String(reflecting:snapshot.sessionConfiguration),
            "apply":String(reflecting:chat.applying),"planning":String(reflecting:chat.planning),
            "status":snapshot.status,"message":snapshot.message,"workspace_writes":writes,
            "reply_limit":chat.workspace.outputLimit,"ui_configuration":String(reflecting:chat.currentSessionConfiguration)]
        if let configuration = snapshot.sessionConfiguration {
            row["accepted_configuration"] = ["model":configuration.modelInstance,"session":configuration.sessionInstance,
                "capacity":configuration.capacity,"chunk":configuration.chunk,"whole_budget":configuration.wholeSessionBudget,
                "request_id":configuration.acceptedRequestID?.uuidString as Any? ?? NSNull()]
        }
        if let transition = snapshot.transition {
            row["transition_request_id"] = transition.request.id.uuidString
            row["transition_committed"] = transition.committed
            if let headroom = transition.headroom {
                row["headroom"] = ["required":headroom.requiredBytes,"available":headroom.availableBytes,
                    "instrument":headroom.instrument,"fits":headroom.fits]
            }
        }
        try emit(row)
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected pinned model and fresh output directory") }
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let file = output.appendingPathComponent("history.json"), gate = ChatDequeueGate()
        let harness = HarnessStore(enabled:true)
        // Replace only the fresh, unloaded default worker with the same actual
        // implementation plus an existing cold dequeue gate. No load policy is
        // injected; FriendlyChatStore still performs ordinary real admission.
        guard let initial = harness.worker else { throw NativeContextFailure("Missing initial worker") }
        initial.close()
        try require(initial.waitUntilClosed(seconds:60),"Initial unloaded worker did not close")
        let worker = try NativeWorker(beforeCommandDequeue:{ gate.pause($0) })
        harness.worker = worker; harness.state = worker.snapshot()
        var writes = 0
        let chat = FriendlyChatStore(harness:harness,fileURL:file,persist:{ bytes,url in
            writes += 1; try bytes.write(to:url,options:.atomic)
        })
        var failure: Error?
        var completed = false
        var successfulApplications = 0, editControls = 0
        do {
            chat.open(model); try settle(harness,chat,worker); try frame("loaded",worker,chat,writes)
            let original = worker.snapshot()
            guard let configuration = original.sessionConfiguration, let live = original.liveContext else {
                throw NativeContextFailure("Missing loaded session: \(original.status) \(original.message)")
            }
            try require(chat.ready && original.status == 0 && configuration.capacity == 512 && configuration.chunk == 32,
                        "Ordinary cached model load failed")
            chat.edit("Keep this unsent draft exactly."); try require(chat.save(),"Baseline persistence failed")
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let workspace = try encoder.encode(chat.workspace), persisted = try Data(contentsOf:file), baselineWrites = writes
            chat.showPlanning(); chat.editPlanning(.upper,"64"); chat.editPlanning(.chunk,"16")
            chat.previewPlanning(); try settle(harness,chat,worker)
            try frame("reply-limit-preview",worker,chat,writes)
            try require(chat.planning.result?.capacity?.capacity.selected?.capacity == 64 && chat.workspace.outputLimit == 128,
                        "Reply-limit rejection witness missing")
            let rejectedDraft = chat.planning.draft
            let beforeRejection = worker.snapshot()
            chat.applyPlanning(); try frame("reply-limit-rejected",worker,chat,writes)
            try require(!chat.applying.pending && !chat.applying.committed && !chat.applying.message.isEmpty
                        && worker.snapshot().sessionConfiguration == configuration && chat.planning.draft == rejectedDraft,
                        "Reply preference rejection changed live settings or edited values")
            var beforeState = beforeRejection, afterState = worker.snapshot()
            beforeState.revision = 0; afterState.revision = 0
            try require(String(reflecting:beforeState) == String(reflecting:afterState),"Rejected Apply changed ordinary worker fields")
            try require(try encoder.encode(chat.workspace) == workspace && Data(contentsOf:file) == persisted && writes == baselineWrites,
                        "Rejected Apply changed workspace or persistence")
            chat.editPlanning(.upper,"256"); chat.previewPlanning(); try settle(harness,chat,worker)
            try frame("apply-preview",worker,chat,writes)
            guard let displayed = chat.planning.result, let point = displayed.capacity?.capacity.selected else {
                throw NativeContextFailure("Missing selected Apply estimate")
            }
            try require(point.capacity == 256 && point.fitsBudget && point.fitsBuffer && chat.planningApplyUnavailableReason == nil,
                        "Expected resident replacement unavailable: \(chat.planningApplyUnavailableReason ?? "unknown")")
            chat.applyPlanning()
            guard let expected = chat.applying.request else { throw NativeContextFailure("Apply was not submitted: \(chat.applying.message)") }
            guard let accepted = worker.waitUntilIdle(seconds:120) else { throw NativeContextFailure("Apply idle deadline") }
            try frame("accepted-before-ui-poll",worker,chat,writes)
            try require(accepted.transition?.committed == true && accepted.transition?.status == 0 && accepted.transition?.headroom?.fits == true && !accepted.restartRequired,
                        "Actual replacement rejected: \(String(reflecting:accepted.transition))")
            // Erase the transient result before any store observation. Hide must
            // reconcile the durable accepted UUID before clearing preview state.
            worker.cancel(); chat.hidePlanning(); harness.poll(); chat.observe(harness.state)
            try frame("late-cancel-hidden",worker,chat,writes)
            guard var applied = worker.snapshot().sessionConfiguration else { throw NativeContextFailure("Applied descriptor disappeared") }
            try require(applied.acceptedRequestID == expected.transition.id && applied.modelInstance == configuration.modelInstance
                        && applied.sessionInstance != configuration.sessionInstance && applied.capacity == 256 && applied.chunk == 16
                        && applied.wholeSessionBudget == expected.transition.wholeSessionBudget
                        && chat.currentSessionConfiguration == applied && chat.applying.committed && worker.snapshot().transition == nil,
                        "Late cancellation was mistaken for rollback")
            successfulApplications += 1
            try require(worker.snapshot().liveContext?.loadedLimit == live.loadedLimit
                        && worker.snapshot().liveContext?.history == 0
                        && String(reflecting:worker.snapshot().runtimePlan) == String(reflecting:original.runtimePlan),
                        "Replacement changed load bound/history or presented a new load plan")
            try require(try encoder.encode(chat.workspace) == workspace && Data(contentsOf:file) == persisted && writes == baselineWrites,
                        "Successful Apply or hiding changed workspace/persistence")
            try emit(["event":"apply-preservation","workspace_before":digest(workspace),"workspace_after":digest(try encoder.encode(chat.workspace)),
                "file_before":digest(persisted),"file_after":digest(try Data(contentsOf:file)),"writes_before":baselineWrites,"writes_after":writes,
                "preview_id":displayed.request.id.uuidString,"accepted_request":expected.transition.id.uuidString])

            // Each field begins with an actual fresh committed notice. Reuse
            // the one loaded model; never mutate workspace preferences. These
            // are three empty replacements in total, not model reloads.
            let edits: [(FriendlyPlanningDraft.Field,String)] = [(.upper,"255"),(.chunk,"8"),(.budgetMiB,"1535")]
            for (index,edit) in edits.enumerated() {
                chat.showPlanning()
                if index > 0 {
                    chat.editPlanning(.upper,"256"); chat.editPlanning(.chunk,"16")
                    chat.editPlanning(.budgetMiB,String(applied.wholeSessionBudget / (1024*1024)))
                    chat.previewPlanning(); try settle(harness,chat,worker)
                    try frame("notice-preview-\(index)",worker,chat,writes)
                    chat.applyPlanning()
                    guard let repeatRequest = chat.applying.request,
                          let repeatResult = worker.waitUntilIdle(seconds:120) else {
                        throw NativeContextFailure("Notice-control Apply was not completed: \(chat.applying.message)")
                    }
                    try frame("notice-accepted-\(index)",worker,chat,writes)
                    try require(repeatResult.transition?.committed == true && repeatResult.transition?.status == 0
                                && repeatResult.transition?.headroom?.fits == true && !repeatResult.restartRequired,"Notice-control replacement rejected")
                    harness.poll(); chat.observe(harness.state)
                    guard let replacement = worker.snapshot().sessionConfiguration else { throw NativeContextFailure("Missing notice-control configuration") }
                    try require(replacement.acceptedRequestID == repeatRequest.transition.id
                                && replacement.sessionInstance != applied.sessionInstance
                                && replacement.modelInstance == applied.modelInstance && replacement.capacity == applied.capacity
                                && replacement.chunk == applied.chunk && replacement.wholeSessionBudget == applied.wholeSessionBudget,
                                "Repeated empty replacement changed configuration or model")
                    applied = replacement; successfulApplications += 1
                }
                try require(chat.applying.committed && !chat.applying.message.isEmpty,"Missing committed edit-notice witness")
                chat.editPlanning(edit.0,edit.1)
                try frame("committed-edit-\(index)",worker,chat,writes)
                try require(!chat.applying.committed && !chat.applying.pending && chat.applying.message.isEmpty
                            && chat.currentSessionConfiguration == applied && worker.snapshot().sessionConfiguration == applied,
                            "Editing retained an old Applied notice or lost durable configuration")
                switch edit.0 {
                case .upper: try require(chat.planning.draft.upper == edit.1,"Upper edit lost")
                case .chunk: try require(chat.planning.draft.chunk == edit.1,"Chunk edit lost")
                case .budgetMiB: try require(chat.planning.draft.budgetMiB == edit.1,"Budget edit lost")
                }
                try require(try encoder.encode(chat.workspace) == workspace && Data(contentsOf:file) == persisted && writes == baselineWrites,
                            "Edit-notice controls changed conversation, draft, reply or persistence")
                editControls += 1
            }
            chat.hidePlanning()
            try emit(["event":"edit-preservation","applications":successfulApplications,"committed_edit_controls":editControls,
                "workspace_before":digest(workspace),"workspace_after":digest(try encoder.encode(chat.workspace)),
                "file_before":digest(persisted),"file_after":digest(try Data(contentsOf:file)),
                "writes_before":baselineWrites,"writes_after":writes])

            // Fixed pinned-codec workload: require the actual combined capacity
            // rejection, not an assumed token count or an earlier input cap.
            // An edit while dequeue is gated must survive unstarted rollback.
            let longPrompt = String(repeating:"hello ",count:140)
            chat.edit(longPrompt); try require(chat.save(),"Preflight baseline save failed")
            let priorMessages = chat.workspace.current.messages
            chat.sendOrStop()
            guard gate.wait() else {
                try frame("missing-chat-gate",worker,chat,writes)
                throw NativeContextFailure("Chat dequeue gate deadline")
            }
            let newerDraft = "Keep this newer edit while the rejected request finishes."
            chat.edit(newerDraft); gate.resume(); try settle(harness,chat,worker)
            try frame("combined-window-rejected",worker,chat,writes)
            let rejection = worker.snapshot()
            try require(!gate.timedOut && rejection.chatResult?.rejected == true
                        && rejection.chatResult?.error.hasPrefix("This request needs ") == true
                        && rejection.chatResult?.error.contains(" tokens including its reply; the current shared window is 256. ") == true
                        && rejection.emitted == 0 && rejection.ids.isEmpty && rejection.liveContext?.history == 0,
                        "Expected combined rendered-prompt/reply rejection before generation")
            try require(chat.workspace.current.draft == newerDraft && chat.workspace.outputLimit == 128
                        && String(reflecting:chat.workspace.current.messages) == String(reflecting:priorMessages)
                        && rejection.sessionConfiguration == applied,"Unstarted rejection overwrote the new draft or active configuration")
            completed = true
        } catch { failure = error }
        gate.resume(); worker.close(); let closed = worker.waitUntilClosed(seconds:60)
        harness.poll(); chat.observe(harness.state)
        try emit(["event":"summary","passed":failure == nil && completed && closed && successfulApplications == 3 && editControls == 3,"closed":closed,
            "error":failure.map { $0.localizedDescription } ?? "","loaded_controller_checked":completed,"successful_applications":successfulApplications,"committed_edit_controls":editControls,
            "successful_generation":false,"rendered_ui_checked":false,"phone_checked":false,"settings_persisted":false])
        if let failure { throw failure }
        try require(completed && closed && successfulApplications == 3 && editControls == 3,"Incomplete loaded Apply control")
    }
}

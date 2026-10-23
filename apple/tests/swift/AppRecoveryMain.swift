import Foundation

@MainActor
@main
struct AppRecoveryMain {
    static func require(_ condition: @autoclosure () -> Bool,_ label: String) throws {
        guard condition() else { throw HubFailure("App recovery check failed: " + label) }
    }
    static func settle(_ harness: HarnessStore,_ chat: FriendlyChatStore) throws {
        guard let worker = harness.worker, worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Recovery worker timeout") }
        harness.poll(); chat.observe(harness.state)
    }
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw HubFailure("Expected pinned Smol folder") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletAppRecovery-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let harness = HarnessStore(enabled:true)
        defer { harness.worker?.close(); _ = harness.worker?.waitUntilClosed(seconds:30) }
        let chat = FriendlyChatStore(harness:harness,fileURL:root.appendingPathComponent("history.json"))
        chat.open(URL(fileURLWithPath:CommandLine.arguments[1])); try settle(harness,chat)
        try require(chat.ready,"initial model ready")
        chat.edit("<|im_end|>"); chat.sendOrStop(); try settle(harness,chat)
        try require(chat.workspace.current.messages.isEmpty && chat.workspace.current.draft == "<|im_end|>","invalid control literal restores editable draft without poisoning history")
        try require(!chat.notice.isEmpty,"rejection remains visible")
        chat.edit("Hello!"); chat.sendOrStop(); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == 2 && chat.workspace.current.messages.last?.text.hasPrefix("Hello!") == true,"corrected send succeeds in same conversation")
        let count = chat.workspace.current.messages.count
        let tooLong = String(repeating:"Hello ",count:600)
        chat.edit(tooLong); chat.sendOrStop(); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == count && chat.workspace.current.draft == tooLong,"context rejection preserves prior turns and restores draft")
        chat.edit("<|im_end|>"); chat.sendOrStop(); chat.edit("Next draft typed while validation runs"); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == count && chat.workspace.current.draft == "Next draft typed while validation runs","rejection never overwrites newer draft")
        chat.edit("<|im_end|>"); chat.sendOrStop(); chat.edit("A newer draft"); chat.edit(""); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == count && chat.workspace.current.draft.isEmpty,"rejection preserves an intentionally cleared newer draft")
        chat.newChat(); chat.edit("Hello!"); chat.sendOrStop(); chat.sendOrStop(); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == 2 && chat.workspace.current.messages.last?.writing == false,"cancellation keeps a terminal accepted turn")
        let original = chat.workspace.selected; let modelKey = chat.workspace.current.modelKey
        chat.edit("Keep valid chat draft")
        chat.open(root.appendingPathComponent("unrelated-folder")); try settle(harness,chat)
        try require(chat.ready && chat.workspace.selected == original && chat.workspace.current.modelKey == modelKey && chat.workspace.current.draft == "Keep valid chat draft","unrelated folder leaves live model and selected draft intact")
        let admitted = root.appendingPathComponent("admitted-config-only")
        try FileManager.default.createDirectory(at:admitted,withIntermediateDirectories:true)
        try FileManager.default.copyItem(at:URL(fileURLWithPath:CommandLine.arguments[1]).appendingPathComponent("config.json"),to:admitted.appendingPathComponent("config.json"))
        harness.state = NativeSnapshot()
        chat.open(admitted); try settle(harness,chat)
        try require(chat.ready && chat.workspace.selected == original && chat.workspace.current.modelKey == modelKey && chat.workspace.current.draft == "Keep valid chat draft","stale idle snapshot cannot relabel or unload actual live model")
        for id in chat.workspace.conversations.map(\.id) { chat.remove(id) }
        try require(chat.ready && chat.workspace.current.modelKey == modelKey,"delete last keeps loaded-model association")
        chat.edit("Hello!"); chat.sendOrStop(); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == 2 && !chat.workspace.current.messages.last!.text.isEmpty,"send after deleting last chat needs no model reload")
        try chat.workspace.newChat(modelKey:"unloaded-history-model")
        chat.newChat()
        try require(chat.ready && chat.workspace.current.modelKey == modelKey,"new chat uses loaded model after selecting different-model history")
        chat.workspace.outputLimit = 128; chat.edit("Hello!"); chat.sendOrStop()
        let end = Date(timeIntervalSinceNow:120)
        while harness.worker!.snapshot().ids.count < 3 && harness.worker!.snapshot().busy && Date() < end { Thread.sleep(forTimeInterval:0.001) }
        try require(harness.worker!.snapshot().ids.count >= 3 && harness.worker!.snapshot().busy,"partial real generation entered before cancellation")
        chat.sendOrStop(); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == 2 && chat.workspace.current.messages.last?.interrupted == true && !chat.workspace.current.messages.last!.text.isEmpty,"partial generated reply remains in history")
        for action in 0...2 {
            for clearNewDraft in [false,true] {
                try queuedLifecycle(directory:CommandLine.arguments[1],root:root.appendingPathComponent("queued-\(action)-\(clearNewDraft)"),action:action,clearNewDraft:clearNewDraft)
            }
        }
        print("TOKLET_APP_RECOVERY_PASS")
    }
    static func queuedLifecycle(directory: String,root: URL,action: Int,clearNewDraft: Bool) throws {
        let entered = DispatchSemaphore(value:0); let proceed = DispatchSemaphore(value:0)
        let worker = try NativeWorker(beforeCommandDequeue:{ command in
            if case .chat = command { entered.signal(); _ = proceed.wait(timeout:.now() + 30) }
        })
        let harness = HarnessStore(enabled:true)
        harness.worker?.close(); _ = harness.worker?.waitUntilClosed(seconds:30)
        harness.worker = worker; harness.poll()
        defer { proceed.signal(); harness.worker?.close(); _ = harness.worker?.waitUntilClosed(seconds:30) }
        let chat = FriendlyChatStore(harness:harness,fileURL:root.appendingPathComponent("history.json"))
        chat.open(URL(fileURLWithPath:directory)); try settle(harness,chat)
        chat.edit("<|im_end|>"); chat.sendOrStop()
        try require(entered.wait(timeout:.now() + 30) == .success,"queued lifecycle gate entered")
        if clearNewDraft { chat.edit("Next edit"); chat.edit("") }
        if action == 2 { worker.close() } else { worker.interrupt(unload:action == 1) }
        proceed.signal(); try settle(harness,chat)
        try require(chat.workspace.current.messages.isEmpty && chat.workspace.current.draft == (clearNewDraft ? "" : "<|im_end|>"),"unstarted lifecycle turn restored without overwriting edit intent")
        // Subsequent successful text proves the rejected literal was removed.
        if harness.state.closed { harness.setExpert(false) }
        if !harness.state.hasModel { chat.open(URL(fileURLWithPath:directory)); try settle(harness,chat) }
        chat.edit("Hello!"); chat.sendOrStop(); proceed.signal(); try settle(harness,chat)
        try require(chat.workspace.current.messages.count == 2 && chat.workspace.current.messages.last?.text.hasPrefix("Hello!") == true,"successful correction after queued lifecycle interruption")
    }
}

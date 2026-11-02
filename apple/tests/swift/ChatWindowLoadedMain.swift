import Foundation

// Explicit local Mac test: UI controller -> rendered tokenizer -> native shared
// admission -> actual generation. No output-quality or throughput assertions.
@main
@MainActor
struct ChatWindowLoadedMain {
    static func check(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected model and fresh output folder") }
        let folder = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("Missing worker") }
        initial.close(); try check(initial.waitUntilClosed(seconds:30),"Initial close timeout")
        let worker = try NativeWorker(captureSavedPrefix:true)
        harness.worker = worker; harness.state = worker.snapshot()
        let preferences = FriendlyPreferenceStore(directory:output.appendingPathComponent("preferences"))
        let chat = FriendlyChatStore(harness:harness,fileURL:output.appendingPathComponent("chat.json"),preferences:preferences)
        var failure: Error?
        func wait() throws -> NativeSnapshot {
            guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Chat timeout") }
            harness.poll(); chat.observe(harness.state)
            return state
        }
        func send(_ digits: Int) throws -> NativeSnapshot {
            chat.edit(String(repeating:"1",count:digits))
            try check(chat.canSend,"Send unavailable: \(chat.notice)")
            chat.sendOrStop()
            return try wait()
        }
        do {
            chat.setQwen2LoadUpper(1024)
            chat.openManaged(folder,profile:.qwen,ifSelected:chat.workspace.selected)
            let loaded = try wait()
            guard let identity = loaded.acceptedIdentity else { throw NativeContextFailure("Load failed: \(loaded.message)") }
            try check(loaded.status == 0 && loaded.sessionConfiguration?.capacity == 512,"Initial capacity changed")
            chat.setReplyLimit(32)
            let rejected512 = try send(600)
            try check(rejected512.chatResult?.rejected == true && rejected512.history == 0
                && chat.workspace.current.messages.isEmpty && chat.workspace.current.draft == String(repeating:"1",count:600),
                "Over-window draft was lost or entered KV")
            chat.showPlanning(); chat.editPlanning(.upper,"1024"); chat.previewPlanning()
            _ = try wait(); try check(chat.planningApplyUnavailableReason == nil,"Growth unavailable")
            chat.applyPlanning()
            let grown = try wait()
            try check(grown.status == 0 && grown.transition?.committed == true
                && grown.acceptedIdentity == identity && grown.sessionConfiguration?.capacity == 1024,"Growth failed")
            let longer = try send(600)
            // Independent pinned Qwen template fixture has29 non-payload IDs;
            // its numeric pretokenizer emits one token for each ASCII digit.
            try check(longer.status == 0 && longer.requestBudget?.input == 629 && longer.requestBudget?.required == 661
                && longer.chatResult?.rejected == false && longer.emitted > 0
                && chat.workspace.current.messages.count == 2 && chat.workspace.current.draft.isEmpty,
                "Longer rendered Chat failed: \(longer.message)")
            chat.newChat()
            let exact = try send(963) // 963+29 rendered input +32 reply =1024.
            try check(exact.status == 0 && exact.requestBudget?.input == 992 && exact.requestBudget?.required == 1024
                && exact.chatResult?.rejected == false && exact.emitted > 0,"Exact shared boundary failed: \(exact.message)")
            let configuration = exact.sessionConfiguration, prefix = exact.savedPrefix
            let record = try preferences.url(identity.namespace), saved = try Data(contentsOf:record)
            chat.newChat()
            let overflow = try send(964) // Prompt fits1024, but prompt+reply does not.
            try check(overflow.status != 0 && overflow.chatResult?.rejected == true && overflow.requestBudget == nil
                && overflow.acceptedIdentity == identity && overflow.sessionConfiguration == configuration
                && overflow.history == exact.history && overflow.evaluated == exact.evaluated && overflow.pending == exact.pending
                && overflow.savedPrefix == prefix && prefix?.count == Int(exact.history)
                && chat.workspace.current.messages.isEmpty && chat.workspace.current.draft == String(repeating:"1",count:964)
                && (try Data(contentsOf:record)) == saved,"Shared overflow changed prefix, owners, draft or preference")
            print("CHAT_WINDOW_MAC passed longer_input=629 exact_input=992 exact_required=1024 overflow_required=1025 longer_emitted=\(longer.emitted) exact_emitted=\(exact.emitted) prefix_preserved=true")
        } catch { failure = error }
        worker.close(); try check(worker.waitUntilClosed(seconds:60),"Close timeout")
        if let failure { throw failure }
    }
}

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
            guard let initialMemory = loaded.sessionMemory,
                  case .available(let initialShared,let initialOwned,_,_,let storage) = initialMemory.reading else {
                throw NativeContextFailure("Initial owner snapshot unavailable")
            }
            try check(initialMemory.matches(loaded.liveContext,configuration:loaded.sessionConfiguration)
                && initialOwned.kvBytes == 6292128 && storage == "Fixed FP16","Initial memory snapshot is stale or has wrong storage")
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
            guard let grownMemory = grown.sessionMemory,
                  case .available(let shared,let owned,_,_,_) = grownMemory.reading else { throw NativeContextFailure("Replacement owner snapshot unavailable") }
            try check(grownMemory.matches(grown.liveContext,configuration:grown.sessionConfiguration)
                && !initialMemory.matches(grown.liveContext,configuration:grown.sessionConfiguration)
                && shared.totalBytes == initialShared.totalBytes && shared.learnedBytes == initialShared.learnedBytes
                && owned.kvBytes == 12583584,"Replacement snapshot changed shared model or retained initial KV")
            let longer = try send(600)
            // Independent pinned Qwen template fixture has29 non-payload IDs;
            // its numeric pretokenizer emits one token for each ASCII digit.
            try check(longer.status == 0 && longer.requestBudget?.input == 629 && longer.requestBudget?.required == 661
                && longer.chatResult?.rejected == false && longer.emitted > 0
                && chat.workspace.current.messages.count == 2 && chat.workspace.current.draft.isEmpty,
                "Longer rendered Chat failed: \(longer.message)")
            try check(longer.sessionMemory?.matches(longer.liveContext,configuration:longer.sessionConfiguration) == true,
                "Generation lost its historical owner snapshot")
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
            try check(overflow.sessionMemory?.configuration == exact.sessionMemory?.configuration,"Rejection changed memory snapshot binding")
            try check(worker.submit(.suspend),"Suspend submission failed")
            let suspended = try wait()
            try check(suspended.sessionMemory == nil && !suspended.hasSession,"Suspend retained an owner snapshot")
            try check(worker.submit(.replay),"Replay submission failed")
            let resumed = try wait()
            guard let resumedMemory = resumed.sessionMemory,
                  case .available(let resumedShared,let resumedOwned,_,_,_) = resumedMemory.reading else { throw NativeContextFailure("Resumed snapshot unavailable") }
            try check(resumed.status == 0 && resumedMemory.matches(resumed.liveContext,configuration:resumed.sessionConfiguration)
                && !grownMemory.matches(resumed.liveContext,configuration:resumed.sessionConfiguration)
                && resumedShared.learnedBytes == shared.learnedBytes && resumedOwned.kvBytes == owned.kvBytes
                && resumed.savedPrefix == prefix,"Resume failed to refresh snapshot without changing the retained prefix")
            print("CHAT_WINDOW_MAC passed longer_input=629 exact_input=992 exact_required=1024 overflow_required=1025 longer_emitted=\(longer.emitted) exact_emitted=\(exact.emitted) prefix_preserved=true")
        } catch { failure = error }
        worker.close(); try check(worker.waitUntilClosed(seconds:60),"Close timeout")
        try check(worker.snapshot().sessionMemory == nil,"Close retained a memory snapshot")
        if let failure { throw failure }
    }
}

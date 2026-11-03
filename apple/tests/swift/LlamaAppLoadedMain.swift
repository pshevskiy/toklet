import Foundation

// Explicit cached Mac controller journey; no UI rendering or quality assertions.
@main
@MainActor
struct LlamaAppLoadedMain {
    static func check(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func main() throws {
        guard CommandLine.arguments.count == 4, CommandLine.arguments[3] == "--experimental-llama3" else {
            throw NativeContextFailure("Expected model, fresh output directory and experimental flag")
        }
        let folder = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("Missing worker") }
        initial.close(); try check(initial.waitUntilClosed(seconds:30),"Initial close timeout")
        let worker = try NativeWorker(captureSavedPrefix:true)
        harness.worker = worker; harness.state = worker.snapshot()
        let preferences = FriendlyPreferenceStore(directory:output.appendingPathComponent("preferences"))
        let other = FriendlyModelPreference(namespace:NativeAcceptedIdentity.Profile.qwen2.namespace,capacity:256,chunk:16,reply:32,wholeBudget:HubProfile.qwen.numericalBudget)
        try preferences.save(other)
        let otherURL = try preferences.url(other.namespace), preservedOther = try Data(contentsOf:otherURL)
        let chat = FriendlyChatStore(harness:harness,fileURL:output.appendingPathComponent("chat.json"),preferences:preferences)
        var failure: Error?
        func wait() throws -> NativeSnapshot {
            for _ in 0..<6 {
                guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Llama controller deadline") }
                harness.poll(); chat.observe(harness.state)
                if !worker.snapshot().busy && !chat.loadingModel && !chat.restoringPreference { return state }
            }
            throw NativeContextFailure("Llama controller did not settle")
        }
        func send(_ text: String) throws -> NativeSnapshot {
            chat.edit(text); try check(chat.canSend,"Send unavailable: \(chat.notice)")
            chat.sendOrStop(); return try wait()
        }
        func workspaceBytes() throws -> Data { let e = JSONEncoder(); e.outputFormatting = [.sortedKeys]; return try e.encode(chat.workspace) }
        do {
            chat.openManaged(folder,profile:.llama3,ifSelected:chat.workspace.selected)
            let loaded = try wait()
            guard let identity = loaded.acceptedIdentity else { throw NativeContextFailure("No Llama identity: \(loaded.message) / \(chat.notice)") }
            try check(loaded.status == 0 && identity.profile == .llama3 && identity.loadedLimit == 512
                && loaded.sessionConfiguration?.capacity == 512 && chat.ready,"Llama route failed")
            chat.setReplyLimit(32)
            let reply = try send("Say hello in one short sentence.")
            try check(reply.status == 0 && reply.emitted > 0 && reply.chatResult?.rejected == false
                && chat.workspace.current.messages.count == 2,"Rendered send failed")
            let beforeWorkspace = try workspaceBytes(), before = worker.snapshot()
            chat.open(folder) // Ordinary folder selection must fail before teardown.
            let rejected = worker.snapshot()
            try check(!rejected.busy && rejected.hasModel && rejected.acceptedIdentity == identity
                && rejected.sessionConfiguration == before.sessionConfiguration && rejected.savedPrefix == before.savedPrefix
                && (try workspaceBytes()) == beforeWorkspace && chat.notice.contains("Models"),"Ordinary rejection damaged current chat")
            chat.newChat(); chat.setReplyLimit(128); chat.edit("Write a long story about a sailor exploring an island.")
            try check(chat.canSend,"Cancel setup unavailable"); chat.sendOrStop()
            let deadline = Date().addingTimeInterval(60)
            while worker.snapshot().busy && worker.snapshot().ids.isEmpty && Date() < deadline { Thread.sleep(forTimeInterval:0.001) }
            try check(worker.snapshot().busy && !worker.snapshot().ids.isEmpty,"No cancellable emitted token")
            harness.poll(); chat.observe(harness.state)
            try check(chat.running,"Stop action unavailable")
            chat.sendOrStop()
            let cancelled = try wait()
            let cancelFields: [String:Any] = ["status":cancelled.status,"reason":cancelled.reason,"result_reason":cancelled.chatResult?.reason ?? 0,"history":cancelled.history,"evaluated":cancelled.evaluated,"pending":cancelled.pending,"emitted":cancelled.emitted,"ids":cancelled.ids,"saved_count":cancelled.savedPrefix?.count ?? -1]
            print("LLAMA_CANCEL " + (String(data:try JSONSerialization.data(withJSONObject:cancelFields,options:[.sortedKeys]),encoding:.utf8) ?? ""))
            guard let prefix = cancelled.savedPrefix else { throw NativeContextFailure("No cancellation prefix") }
            try check(cancelled.status == 0 && cancelled.reason == 3 && cancelled.chatResult?.reason == 3
                && cancelled.chatResult?.rejected == false && cancelled.emitted > 0 && cancelled.pending <= 1
                && cancelled.history == cancelled.evaluated + cancelled.pending && prefix.count == Int(cancelled.history)
                && cancelled.ids.count == Int(cancelled.emitted) && Array(prefix.suffix(cancelled.ids.count)) == cancelled.ids,
                "Cancellation prefix failed")
            chat.newChat(); chat.setReplyLimit(32)
            let recovered = try send("Say hello.")
            try check(recovered.status == 0 && recovered.emitted > 0,"Recovery failed")
            let recordURL = try preferences.url(identity.namespace), record = try Data(contentsOf:recordURL)
            try check((try Data(contentsOf:otherURL)) == preservedOther,"Other profile settings changed")
            chat.unload(); let unloaded = try wait()
            try check(!unloaded.hasModel && chat.acceptedIdentity == nil,"Unload retained identity")
            chat.openManaged(folder,profile:.llama3,ifSelected:chat.workspace.selected)
            let reopened = try wait()
            try check(reopened.status == 0 && reopened.acceptedIdentity?.profile == .llama3
                && reopened.acceptedIdentity?.modelInstance != identity.modelInstance && chat.ready
                && chat.workspace.outputLimit == 32 && (try Data(contentsOf:recordURL)) == record
                && (try Data(contentsOf:otherURL)) == preservedOther,"Reopen or isolated preferences failed")
            chat.newChat(); let final = try send("Say hello again.")
            try check(final.status == 0 && final.emitted > 0,"Reopened chat unusable")
            let summary: [String:Any] = ["loaded":true,"rendered_send":true,"ordinary_rejection_preserved":true,"cancel_pending":cancelled.pending,"reopened":true,"preferences_isolated":true,"ui_rendered":false,"phone_qualified":false]
            let bytes = try JSONSerialization.data(withJSONObject:summary,options:[.sortedKeys])
            print("LLAMA_APP " + (String(data:bytes,encoding:.utf8) ?? ""))
        } catch { failure = error }
        worker.close(); try check(worker.waitUntilClosed(seconds:30),"Worker close timeout")
        if let failure { throw failure }
    }
}

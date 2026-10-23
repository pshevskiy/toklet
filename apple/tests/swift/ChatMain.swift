import Foundation

@main
@MainActor
struct ChatMain {
    static func checkMalformedHistory() throws {
        var workspace = ChatWorkspace(); workspace.edit("First")
        let pending = try workspace.begin()
        try workspace.newChat(); workspace.edit("Second draft")
        guard workspace.finish(NativeChatResult(id:pending.0,text:"Late first reply",reason:1,error:"")),
              workspace.current.messages.isEmpty, workspace.current.draft == "Second draft" else { throw chatInputError("Late reply entered selected conversation") }
        let valid = try JSONEncoder().encode(workspace)
        var cases = [Data(valid.dropLast()),Data(repeating:0,count:ChatWorkspace.maximumBytes + 1)]
        let original = try JSONSerialization.jsonObject(with:valid) as! [String:Any]
        var bad = original; bad["version"] = 99
        cases.append(try JSONSerialization.data(withJSONObject:bad))
        bad = original; bad["selected"] = UUID().uuidString
        cases.append(try JSONSerialization.data(withJSONObject:bad))
        var chats = original["conversations"] as! [[String:Any]]
        chats[1]["id"] = chats[0]["id"]; bad = original; bad["conversations"] = chats
        cases.append(try JSONSerialization.data(withJSONObject:bad))
        chats = original["conversations"] as! [[String:Any]]
        var messages = chats[0]["messages"] as! [[String:Any]]
        messages[0]["role"] = "tool"; chats[0]["messages"] = messages; bad = original; bad["conversations"] = chats
        cases.append(try JSONSerialization.data(withJSONObject:bad))
        for data in cases {
            var rejected = false
            do { _ = try ChatWorkspace.restore(data) } catch { rejected = true }
            guard rejected else { throw chatInputError("Malformed history was accepted") }
        }
    }
    static func checkStorageAndModes(directory: String?) throws {
        try checkMalformedHistory()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("toklet-chat-check-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:false)
        defer { try? FileManager.default.removeItem(at:folder) }
        let file = folder.appendingPathComponent("history.json")
        try Data("bad json".utf8).write(to:file)
        let harness = HarnessStore(enabled:true)
        defer { harness.worker?.close(); _ = harness.worker?.waitUntilClosed(seconds:30) }
        let chat = FriendlyChatStore(harness:harness,fileURL:file)
        guard !chat.storageAvailable else { throw chatInputError("Broken history was accepted") }
        chat.edit("This edit must stay disabled")
        chat.newChat()
        guard chat.workspace.current.draft.isEmpty, chat.workspace.conversations.count == 1 else { throw chatInputError("Blocked history accepted edits") }
        var original = ChatWorkspace(); original.edit("Original saved draft 🌍")
        try JSONEncoder().encode(original).write(to:file,options:.atomic)
        chat.retryStorage()
        guard chat.storageAvailable, chat.workspace.current.draft == "Original saved draft 🌍" else { throw chatInputError("Retry lost saved draft") }
        chat.edit("Changed draft"); chat.save()
        guard try ChatWorkspace.restore(Data(contentsOf:file)).current.draft == "Changed draft" else { throw chatInputError("Draft did not persist") }
        let blocked = folder.appendingPathComponent("blocked")
        try Data("file blocks directory".utf8).write(to:blocked)
        let unsaved = FriendlyChatStore(harness:harness,fileURL:blocked.appendingPathComponent("history.json"))
        unsaved.edit("Keep after write failure"); unsaved.save()
        guard !unsaved.storageError.isEmpty, unsaved.workspace.current.draft == "Keep after write failure" else { throw chatInputError("Write failure lost draft") }
        try FileManager.default.removeItem(at:blocked)
        unsaved.retryStorage()
        guard unsaved.storageError.isEmpty,
              try ChatWorkspace.restore(Data(contentsOf:blocked.appendingPathComponent("history.json"))).current.draft == "Keep after write failure" else { throw chatInputError("Write retry lost draft") }
        let worker = harness.worker!
        harness.setExpert(true)
        guard harness.expert, harness.worker === worker else { throw chatInputError("Mode switch replaced an active worker") }
        worker.close(); guard worker.waitUntilClosed(seconds:30) else { throw chatInputError("Worker close timed out") }
        harness.poll(); harness.setExpert(false)
        guard !harness.expert, harness.worker !== worker, !harness.state.closed else { throw chatInputError("Back from terminal Close did not restore worker") }
        if let directory {
            chat.open(URL(fileURLWithPath:directory))
            guard harness.worker?.waitUntilIdle(seconds:120) != nil else { throw chatInputError("Chat load timed out") }
            harness.poll(); chat.observe(harness.state)
            guard chat.ready else { throw chatInputError("Chat model is not ready") }
            chat.edit("Hello!"); chat.sendOrStop(); chat.enterExpert()
            guard harness.worker?.waitUntilIdle(seconds:120) != nil else { throw chatInputError("Expert cancellation timed out") }
            harness.poll(); chat.observe(harness.state)
            guard harness.expert, !chat.running, !harness.state.hasModel,
                  chat.workspace.current.messages.last?.writing == false else { throw chatInputError("Send to Expert lost completion or model cleanup") }
            harness.setExpert(false)
            guard !harness.expert else { throw chatInputError("Back to Chat failed") }
            chat.open(URL(fileURLWithPath:directory))
            guard harness.worker?.waitUntilIdle(seconds:120) != nil else { throw chatInputError("Reload timed out") }
            harness.poll(); chat.observe(harness.state)
            guard let active = harness.worker else { throw chatInputError("Missing worker") }
            requestNativeLifecycle(.memoryWarning,worker:active)
            guard active.waitUntilIdle(seconds:30) != nil else { throw chatInputError("Warning unload timed out") }
            harness.poll(); chat.observe(harness.state)
            guard !chat.ready, !harness.state.hasModel, !chat.workspace.current.messages.isEmpty else { throw chatInputError("Memory warning did not release model and preserve UI history") }
            chat.open(URL(fileURLWithPath:directory))
            guard active.waitUntilIdle(seconds:120) != nil else { throw chatInputError("Identity test load timed out") }
            harness.poll(); chat.observe(harness.state)
            // @Published emits before assignment; a stale UI snapshot must never
            // allow a rejected second load to relabel the already loaded model.
            harness.state = NativeSnapshot()
            chat.open(folder.appendingPathComponent("not-a-model"))
            harness.poll(); chat.observe(harness.state)
            guard !chat.hasChatModel else { throw chatInputError("Rejected load mislabeled the live model") }
        }
    }
    static func main() throws {
        let arguments = CommandLine.arguments
        if arguments.count == 2 && arguments[1] == "--workspace-only" {
            try checkChatWorkspace()
            try checkQueuedChatCancellation()
            try checkStorageAndModes(directory:nil)
            print("TOKLET_CHAT_WORKSPACE_PASS")
        } else if arguments.count == 4 && arguments[1] == "--raw" {
            let result = try runWorkerChecks(directory:arguments[2],metal:arguments[3] == "metal")
            print("TOKLET_RAW_CHECK_JSON=" + String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
        } else {
            guard arguments.count == 2 else { throw chatInputError("Expected a local SmolLM2 folder") }
            let result = try runNativeChatCheck(directory:arguments[1])
            try checkStorageAndModes(directory:arguments[1])
            print("TOKLET_CHAT_CHECK_JSON=" + String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
        }
    }
}

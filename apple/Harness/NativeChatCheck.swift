import Foundation

struct ChatCheckReply: Codable {
    let ids: [UInt32]
    let text: String
    let reason: UInt32
}
struct NativeChatCheck: Codable {
    let passed: Bool
    let replies: [ChatCheckReply]
    let contextRejectionPreservedPrefix: Bool
    let cancellationObserved: Bool
    let explicitSendAfterSuspend: Bool
    let released: Bool
}

func checkChatWorkspace() throws {
    func require(_ value: Bool) throws { if !value { throw chatInputError("Chat workspace contract failed") } }
    var workspace = ChatWorkspace()
    workspace.edit("Hello, 🌍!")
    let request = try workspace.begin()
    try require(workspace.current.messages.count == 2 && workspace.current.draft.isEmpty)
    try require(workspace.finish(NativeChatResult(id:request.0,text:"Hello!",reason:1,error:"")))
    try require(!workspace.finish(NativeChatResult(id:request.0,text:"Duplicate",reason:1,error:"")))
    try workspace.newChat()
    workspace.edit("Keep this draft")
    try require(!workspace.finish(NativeChatResult(id:UUID(),text:"Stale",reason:1,error:"")))
    let saved = try JSONEncoder().encode(workspace)
    let restored = try ChatWorkspace.restore(saved)
    try require(restored.current.draft == "Keep this draft" && restored.conversations[0].messages[1].text == "Hello!")
    _ = try workspace.begin()
    let interrupted = try ChatWorkspace.restore(JSONEncoder().encode(workspace))
    try require(interrupted.current.messages.last?.interrupted == true && interrupted.current.messages.last?.writing == false)
    workspace.outputLimit = 99
    do { _ = try ChatWorkspace.restore(JSONEncoder().encode(workspace)); throw chatInputError("Invalid settings were accepted") }
    catch let error as NSError where error.localizedDescription == "Invalid saved chat history" { }
}

func checkQueuedChatCancellation() throws {
    for action in 0...2 {
        let entered = DispatchSemaphore(value:0)
        let proceed = DispatchSemaphore(value:0)
        let worker = try NativeWorker(beforeCommandDequeue: { command in
            if case .chat = command {
                entered.signal(); _ = proceed.wait(timeout:.now() + 30)
            }
        })
        defer { proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let id = UUID()
        guard worker.submit(.chat(id:id,conversation:Data(),maximum:8)),
              entered.wait(timeout:.now() + 30) == .success else { throw chatInputError("Queued chat gate did not run") }
        if action == 2 { worker.close() } else { worker.interrupt(unload:action == 1) }
        proceed.signal()
        guard let state = worker.waitUntilIdle(seconds:30), let result = state.chatResult,
              result.id == id, result.reason == 3, result.text.isEmpty else { throw chatInputError("Queued chat cancellation lost its result") }
        worker.close()
        guard worker.waitUntilClosed(seconds:30), worker.snapshot().chatResult?.id == id else { throw chatInputError("Queued completion lost during close") }
    }
}

// Explicit opt-in diagnostic: no owner's prompts, history, model folders or UI
// preferences are read/written. App argv routing disables its ordinary worker.
func runNativeChatCheck(directory: String,profile: HubProfile = .smol) throws -> NativeChatCheck {
    try checkChatWorkspace()
    try checkQueuedChatCancellation()
    let worker = try NativeWorker()
    defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
    func execute(_ command: NativeCommand, cancelled: Bool = false, allowError: Bool = false) throws -> NativeSnapshot {
        guard worker.submit(command,initiallyCancelled:cancelled), let state = worker.waitUntilIdle(seconds:120),
              allowError || state.status == 0 else { throw chatInputError("Chat worker request failed") }
        return state
    }
    func data(_ content: String) throws -> Data {
        try JSONSerialization.data(withJSONObject:[["role":"user","content":content]],options:[.sortedKeys])
    }
    func reply(_ snapshot: NativeSnapshot, _ id: UUID) throws -> ChatCheckReply {
        guard let result = snapshot.chatResult, result.id == id, result.error.isEmpty else { throw chatInputError("Missing text completion") }
        return ChatCheckReply(ids:snapshot.ids,text:result.text,reason:result.reason)
    }
    var settings = NativeSettings(directory:directory)
    settings.metal = true; settings.mlx = true; settings.chat = true
    settings.capacity = 512; settings.chunk = 32; settings.numericalBudget = profile.numericalBudget; settings.textBudget = profile.textBudget
    let loaded = try execute(.load(settings))
    guard loaded.chatEnabled, loaded.hasSession else { throw chatInputError("Text model was not admitted") }
    let firstID = UUID()
    let first = try execute(.chat(id:firstID,conversation:data("Hello!"),maximum:8))
    let firstReply = try reply(first,firstID)
    guard !firstReply.ids.isEmpty, !firstReply.text.isEmpty else { throw chatInputError("No real text response") }
    let turn: [[String:String]] = [["role":"user","content":"Hello!"],["role":"assistant","content":firstReply.text],["role":"user","content":"What is 2 + 2?"]]
    let secondID = UUID()
    let second = try execute(.chat(id:secondID,conversation:JSONSerialization.data(withJSONObject:turn),maximum:8))
    let secondReply = try reply(second,secondID)
    let rejected = try execute(.chat(id:UUID(),conversation:data(String(repeating:" word",count:400)),maximum:128),allowError:true)
    let preserved = rejected.status != 0 && rejected.history == second.history && rejected.evaluated == second.evaluated && rejected.pending == second.pending
    guard preserved else { throw chatInputError("Context rejection changed numerical prefix") }
    let cancelID = UUID()
    let cancelled = try execute(.chat(id:cancelID,conversation:data("Hello!"),maximum:8),cancelled:true)
    guard cancelled.chatResult?.id == cancelID, cancelled.chatResult?.reason == 3 else { throw chatInputError("Cancellation was not observed") }
    worker.interrupt(unload:false)
    guard let suspended = worker.waitUntilIdle(seconds:30), !suspended.hasSession, suspended.chatEnabled else { throw chatInputError("Session suspend failed") }
    let resumedID = UUID()
    let resumed = try execute(.chat(id:resumedID,conversation:data("Hello!"),maximum:8))
    let resumedReply = try reply(resumed,resumedID)
    guard resumedReply.ids == firstReply.ids, resumedReply.text == firstReply.text else { throw chatInputError("Fresh text replay differs") }
    let unloaded = try execute(.unload)
    guard unloaded.chatResult?.id == resumedID else { throw chatInputError("Completed response lost during cleanup") }
    worker.close()
    guard worker.waitUntilClosed(seconds:30), worker.snapshot().workerFinished,
          !worker.snapshot().hasModel, !worker.snapshot().chatEnabled else { throw chatInputError("Text resources were not released") }
    return NativeChatCheck(passed:true,replies:[firstReply,secondReply,resumedReply],
        contextRejectionPreservedPrefix:preserved,cancellationObserved:true,explicitSendAfterSuspend:true,released:true)
}

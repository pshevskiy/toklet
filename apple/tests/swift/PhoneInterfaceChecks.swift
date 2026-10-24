import Foundation

struct PhoneInterfaceResult: Codable {
    var passed = false
    var smolReply = ""
    var qwenLongIDs: [UInt32] = []
    var qwenLongText = ""
    var longHistory: UInt32 = 0
    var longCommitted: UInt32 = 0
    var longPending: UInt32 = 0
    var longReason: UInt32 = 0
    var codeReply = ""
    var codeBlocks = 0
    var expertIDs: [UInt32] = []
    var switchedBack = false
    var released = false
}

@MainActor
func runPhoneInterfaceCheck() async throws -> PhoneInterfaceResult {
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("TokletInterfaceCheck-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at:temporary) }
    let harness = HarnessStore(enabled:true)
    guard let worker = harness.worker else { throw chatInputError("Missing phone check worker") }
    defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
    let chat = FriendlyChatStore(harness:harness,fileURL:temporary)
    func require(_ value: Bool,_ message: String) throws { if !value { throw chatInputError(message) } }
    func wait(_ condition: () -> Bool) async throws {
        let deadline = Date(timeIntervalSinceNow:120)
        while Date() < deadline {
            harness.poll(); chat.observe(harness.state)
            if condition() { return }
            if !harness.state.busy && harness.state.status != 0 { throw chatInputError(harness.state.message) }
            try await Task.sleep(for:.milliseconds(100))
        }
        throw chatInputError("Timed out: " + harness.state.message + "; " + chat.notice)
    }
    func open(_ profile: HubProfile) async throws {
        chat.open(HubStore.modelRoot.appendingPathComponent(profile.folder))
        try await wait { chat.ready && chat.modelName == profile.name }
    }
    func send(_ text: String,maximum: UInt32) async throws -> NativeSnapshot {
        chat.workspace.outputLimit = maximum; chat.edit(text)
        try require(chat.canSend,"Phone Chat rejected draft before submit")
        chat.sendOrStop(); try await wait { !chat.running && !harness.state.busy }
        try require(chat.workspace.current.messages.last?.notice.hasPrefix("Response limit reached") == true || chat.workspace.current.messages.last?.notice.isEmpty == true,"Phone Chat returned an error")
        return harness.state
    }
    var result = PhoneInterfaceResult()
    try await open(.smol)
    _ = try await send("Hello!",maximum:32)
    result.smolReply = chat.workspace.current.messages.last?.text ?? ""
    try require(!result.smolReply.isEmpty,"Empty SmolLM2 reply")
    let previousConversation = chat.workspace.selected
    try await open(.qwen)
    try require(chat.workspace.selected != previousConversation && chat.workspace.current.messages.isEmpty,"Model switch mixed conversation histories")
    let long = try await send("Write a long adventure story about a cat exploring an old city. Include many scenes, detailed descriptions, and dialogue. Continue the story for at least 2000 words.",maximum:256)
    result.qwenLongIDs = long.ids; result.qwenLongText = chat.workspace.current.messages.last?.text ?? ""
    result.longHistory = long.history; result.longCommitted = long.evaluated; result.longPending = long.pending; result.longReason = long.reason
    try require(long.ids.count == 256 && long.emitted == 256 && long.reason == 1 && long.history == 321 && long.evaluated == 320 && long.pending == 1,"Long phone Chat count/state mismatch")
    chat.newChat()
    _ = try await send("Write a Python function add(a, b) that returns their sum. Put the code in a fenced Markdown block labelled python, then give one sentence explaining the function.",maximum:256)
    result.codeReply = chat.workspace.current.messages.last?.text ?? ""
    for block in ChatBlock.parse(result.codeReply) {
        if case .code(let language,let text) = block {
            result.codeBlocks += 1
            try require(CodeHighlight.spans(text,language:language).map(\.text).joined() == text,"Highlighting changed phone code text")
        }
    }
    try require(result.codeBlocks > 0,"Model did not return a fenced code block")
    try await open(.smol); result.switchedBack = chat.ready && chat.modelName == HubProfile.smol.name
    chat.enterExpert(); try await wait { harness.expert && !harness.state.hasModel && !harness.state.busy }
    guard let tiny = Bundle.main.url(forResource:"Tiny",withExtension:nil) else { throw chatInputError("Missing tiny fixture") }
    harness.submit(.load(NativeSettings(directory:tiny.path))); try await wait { harness.state.hasSession && !harness.state.busy }
    harness.submit(.generate(operation:0,tokens:[1,7,13,3],maximum:2,stops:[])); try await wait { !harness.state.busy }
    result.expertIDs = harness.state.ids; try require(result.expertIDs == [47,47],"Expert fixture output mismatch")
    let raw = result.expertIDs.map(String.init).joined(separator:", ")
    try require(String(CodeHighlight.attributed(raw,language:"Token IDs").characters) == raw,"Expert output highlighting changed IDs")
    harness.worker?.interrupt(unload:true); try await wait { !harness.state.hasModel && !harness.state.busy }
    result.released = true; result.passed = true
    return result
}

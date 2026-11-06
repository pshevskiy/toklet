import Foundation
import CToklet

@main
@MainActor
struct OfficialGemmaWindowLoadedMain {
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func main() async throws {
        try require(CommandLine.arguments.count == 3, "Expected exact Gemma folder and fresh workspace")
        let folder = URL(fileURLWithPath:CommandLine.arguments[1],isDirectory:true)
        let root = URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        let harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("Missing worker") }
        initial.close(); try require(initial.waitUntilClosed(seconds:30),"Initial worker did not close")
        let worker = try NativeWorker(captureSavedPrefix:true)
        harness.worker = worker; harness.state = worker.snapshot()
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let chat = FriendlyChatStore(harness:harness,fileURL:root.appendingPathComponent("chat.json"),
            preferences:FriendlyPreferenceStore(directory:root.appendingPathComponent("prefs")))
        func wait() async throws -> NativeSnapshot {
            let deadline = ContinuousClock.now + .seconds(240)
            while ContinuousClock.now < deadline {
                harness.poll(); chat.observe(harness.state)
                let snapshot = worker.snapshot()
                if !snapshot.busy && !chat.loadingModel { return snapshot }
                try await Task.sleep(for:.milliseconds(20))
            }
            throw NativeContextFailure("Managed Gemma worker did not settle")
        }
        func kv(_ state: NativeSnapshot) throws -> UInt64 {
            guard let report = state.sessionMemory,
                  report.matches(state.liveContext,configuration:state.sessionConfiguration),
                  case .officialGemma(_,_,let value,_) = report.reading else { throw NativeContextFailure("Official memory unavailable") }
            return value
        }

        let conversation = chat.workspace.selected
        chat.setGemmaCapacity(384); chat.setGemmaChunk(16); chat.setGemmaKVCeiling("0")
        chat.openManaged(folder,profile:.gemmaBF16,ifSelected:conversation)
        let initiallyRejected = try await wait()
        try require(!initiallyRejected.hasModel && chat.notice.contains("KV bytes"),
                    "The zero ceiling should reject before opening the model")
        chat.setGemmaKVCeiling("16")
        let plan = try chat.gemmaWindowChoice().plan()
        try require(plan == 14_745_600,"384/16 plan differs")
        chat.openManaged(folder,profile:.gemmaBF16,ifSelected:conversation)
        let loaded = try await wait()
        let actualKV = try kv(loaded)
        try require(loaded.status == TOKLET_OK && chat.ready &&
                    loaded.modelContext?.declaredTotal == 131_072 && loaded.modelContext?.loadedLimit == 512 &&
                    loaded.sessionConfiguration?.capacity == 384 && loaded.sessionConfiguration?.chunk == 16 &&
                    actualKV == plan,"Selected context failed to open: \(loaded.message) \(chat.notice)")
        chat.setReplyLimit(32)
        chat.edit("Name the capital of France.")
        try require(chat.canSend,"Gemma Chat is unavailable")
        chat.sendOrStop()
        let before = try await wait()
        try require(before.status == TOKLET_OK && before.chatResult?.text == "The capital of France is **Paris**.","Initial Chat differs")
        chat.edit("Draft stays")
        let messages = chat.workspace.current.messages.count
        chat.setGemmaCapacity(256); chat.setGemmaChunk(8)
        for ceiling in ["0","16"] {
            chat.setGemmaKVCeiling(ceiling)
            chat.openManaged(folder,profile:.gemmaBF16,ifSelected:conversation)
            let retained = try await wait()
            let expected = ceiling == "0" ? "KV bytes" : "already open"
            try require(chat.notice.contains(expected) && retained.acceptedIdentity == before.acceptedIdentity &&
                        retained.sessionConfiguration == before.sessionConfiguration &&
                        retained.savedPrefix == before.savedPrefix && chat.workspace.current.draft == "Draft stays" &&
                        chat.workspace.current.messages.count == messages,"Next-open edit changed the current owner, prefix or draft")
        }
        try require(worker.submit(.suspend),"Suspend unavailable")
        let suspended = try await wait()
        try require(suspended.hasModel && !suspended.hasSession && suspended.acceptedIdentity == before.acceptedIdentity,"Suspend lost the model owner")
        try require(worker.submit(.replay),"Replay unavailable")
        let replayed = try await wait()
        let replayKV = try kv(replayed)
        try require(replayed.status == TOKLET_OK && replayed.savedPrefix == before.savedPrefix &&
                    replayed.sessionConfiguration?.capacity == 384 && replayed.sessionConfiguration?.chunk == 16 &&
                    replayKV == plan,"Replay failed to restore the frozen context")
        print("Gemma next-open plan and owners: \(actualKV) bytes, rejected settings preserved the old prefix")
    }
}

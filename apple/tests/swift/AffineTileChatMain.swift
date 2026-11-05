import Foundation

// The normal Mac stores and worker run against an isolated verified cache.
@main
@MainActor
struct AffineTileChatMain {
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }

    static func record(_ event: String, _ fields: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: ["event": event, "fields": fields], options: [.sortedKeys])
        try FileHandle.standardOutput.write(contentsOf: Data(("AFFINE_CHAT " + String(decoding: data, as: UTF8.self) + "\n").utf8))
    }

    static func main() async throws {
        try require(CommandLine.arguments.count == 2, "Expected an isolated cache root")
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        try require(!FileManager.default.fileExists(atPath: root.appendingPathComponent("chat-v1.json").path),
                    "The diagnostic conversation must start fresh")
        let coordinator = MacCoordinator(root: root)
        let harness = coordinator.harness, chat = coordinator.chat
        guard let idleWorker = harness.worker else { throw NativeContextFailure("Missing Mac worker") }
        idleWorker.close()
        try require(idleWorker.waitUntilClosed(seconds: 30), "Initial worker close timed out")
        let worker = try NativeWorker(captureSavedPrefix: true)
        harness.worker = worker
        harness.state = worker.snapshot()

        func wait() throws -> NativeSnapshot {
            for _ in 0..<6 {
                guard let state = worker.waitUntilIdle(seconds: 120) else {
                    throw NativeContextFailure("Mac Chat deadline")
                }
                harness.poll()
                chat.observe(harness.state)
                if !worker.snapshot().busy && !chat.loadingModel && !chat.restoringPreference {
                    return state
                }
            }
            throw NativeContextFailure("Mac Chat did not settle")
        }

        func send(_ prompt: String) throws -> NativeSnapshot {
            chat.edit(prompt)
            try require(chat.canSend, "Send unavailable: \(chat.notice)")
            chat.sendOrStop()
            let state = try wait()
            try require(state.status == 0 && state.emitted > 0
                && state.chatResult?.rejected == false && state.chatResult?.error.isEmpty == true
                && [1, 2].contains(state.reason), "Chat reply failed: \(state.message)")
            return state
        }

        func raw(_ operation: UInt32, _ tokens: [UInt32], _ maximum: UInt32) throws -> NativeSnapshot {
            try require(worker.submit(.generate(operation: operation, tokens: tokens,
                                                maximum: maximum, stops: [151645, 151643])),
                        "Raw diagnostic request unavailable")
            let state = try wait()
            try require(state.status == 0, "Raw diagnostic request failed: \(state.message)")
            return state
        }

        func renderedPrompt(_ state: NativeSnapshot) throws -> [UInt32] {
            let sampled = Int(state.emitted) + (state.reason == 2 ? 1 : 0)
            guard let history = state.savedPrefix, history.count == Int(state.history),
                  history.count > sampled else {
                throw NativeContextFailure("Rendered Chat prefix unavailable")
            }
            return Array(history.dropLast(sampled))
        }

        var failure: Error?
        do {
            let profile = HubProfile.qwen
            var folder: URL?
            coordinator.hub.use(profile) { folder = $0 }
            await coordinator.hub.waitForIdle()
            guard let folder else { throw NativeContextFailure("Pinned cache open failed") }
            try ModelHub.verify(folder, profile: profile)
            chat.openManaged(folder, profile: profile, ifSelected: chat.workspace.selected)
            let loaded = try wait()
            try require(loaded.acceptedIdentity?.profile == .qwen2
                && loaded.acceptedIdentity?.loadedLimit == 512 && chat.ready, "Wrong accepted Qwen identity")
            try record("loaded", ["admitted_bytes": loaded.admittedBytes,
                                   "repository": profile.repository, "revision": profile.revision])
            chat.setReplyLimit(32)
            let first = try send("Say hello in one short sentence.")
            let second = try send("Repeat your greeting in one short sentence.")
            guard let a = first.savedPrefix, let b = second.savedPrefix else {
                throw NativeContextFailure("The diagnostic needs saved prefixes")
            }
            let shared = zip(a, b).prefix(while: { $0.0 == $0.1 }).count
            try require(shared > 0 && chat.workspace.current.messages.count == 4, "Continuation prefix differs")
            try record("continuation", ["shared_prefix": shared, "first_emitted": first.emitted,
                                         "second_emitted": second.emitted])
            chat.newChat()
            chat.setReplyLimit(128)
            chat.edit("Write a long story about a sailor exploring an island.")
            try require(chat.canSend, "Stop prompt unavailable")
            chat.sendOrStop()
            let deadline = Date().addingTimeInterval(60)
            while worker.snapshot().busy && worker.snapshot().ids.isEmpty && Date() < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            try require(worker.snapshot().busy && !worker.snapshot().ids.isEmpty, "No active reply to stop")
            harness.poll()
            chat.observe(harness.state)
            try require(chat.running, "Stop action unavailable")
            chat.sendOrStop()
            let stopped = try wait()
            try require(stopped.status == 0 && stopped.reason == 3 && stopped.chatResult?.reason == 3
                && stopped.emitted > 0 && stopped.history == stopped.evaluated + stopped.pending
                && stopped.pending <= 1, "Stop did not preserve a valid prefix")
            try record("stopped", ["emitted": stopped.emitted, "history": stopped.history,
                                    "evaluated": stopped.evaluated, "pending": stopped.pending])
            chat.newChat()
            chat.setReplyLimit(32)
            let recovered = try send("Say hello again.")
            try require(chat.workspace.current.messages.count == 2, "New Chat inherited prior turns")
            try record("reset_reply", ["emitted": recovered.emitted])

            // Friendly Chat intentionally resets each rendered turn. Separately
            // exercise the same worker's exact-prefix path with its real codec
            // outputs and a committed prefix containing no sampled tokens.
            let prefix = try renderedPrompt(first)
            let extended = try renderedPrompt(second)
            try require(extended.starts(with: prefix) && extended.count - prefix.count >= 4,
                        "The fixture needs an eligible exact-prefix extension")
            let primed = try raw(3, prefix, 0)
            try require(Int(primed.evaluated) == prefix.count && primed.pending == 0,
                        "Raw prefix did not commit")
            let reused = try raw(0, extended, 1)
            let fresh = try raw(3, extended, 1)
            try require(reused.ids == fresh.ids && reused.stopToken == fresh.stopToken
                && reused.reason == fresh.reason && reused.savedPrefix == fresh.savedPrefix
                && reused.history == fresh.history && reused.evaluated == fresh.evaluated
                && reused.pending == fresh.pending && Int(reused.history) == extended.count + 1,
                        "Raw reuse differs from a fresh reset")
            try require(chat.workspace.current.messages.count == 2, "Raw witness changed Chat history")
            try record("raw_reuse", ["prefix": prefix.count, "extended": extended.count,
                                     "fresh_output_state_equal": true])
            try ModelHub.verify(folder, profile: profile)
        } catch {
            failure = error
        }
        let closed = await coordinator.shutdown(terminating: true)
        try require(closed, "Mac coordinator shutdown failed")
        if let failure { throw failure }
        try record("complete", ["closed": closed, "ui_rendered": false])
    }
}

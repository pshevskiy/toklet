import Foundation

// The normal Mac stores and worker run against an isolated verified cache.
@main
@MainActor
struct Qwen15MacChatMain {
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }

    static func record(_ event: String, _ fields: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: ["event": event, "fields": fields], options: [.sortedKeys])
        try FileHandle.standardOutput.write(contentsOf: Data(("QWEN15_MAC_CHAT " + String(decoding: data, as: UTF8.self) + "\n").utf8))
    }

    static func main() async throws {
        let download = CommandLine.arguments.count == 3 && CommandLine.arguments[2] == "--download"
        try require(CommandLine.arguments.count == 2 || download, "Expected an isolated cache root and optional --download")
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        try require(!FileManager.default.fileExists(atPath: root.appendingPathComponent("chat-v1.json").path),
                    "The diagnostic conversation must start fresh")
        let coordinator = MacCoordinator(root: root)
        let harness = coordinator.harness, chat = coordinator.chat
        guard let worker = harness.worker else { throw NativeContextFailure("Missing Mac worker") }

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

        var failure: Error?
        do {
            let profile = HubProfile.qwen15
            try require(HubProfile.catalog.contains(profile), "Missing managed Mac profile")
            var folder: URL?
            if download {
                coordinator.hub.query = "Qwen2.5-1.5B-Instruct-8bit"
                coordinator.hub.search()
                await coordinator.hub.waitForIdle()
                try require(coordinator.hub.results.contains { $0.profile == profile },
                            "Pinned profile absent from search: \(coordinator.hub.message)")
                coordinator.hub.downloadAndUse(profile) { folder = $0 }
            } else {
                coordinator.hub.use(profile) { folder = $0 }
            }
            await coordinator.hub.waitForIdle()
            guard let folder else { throw NativeContextFailure("Pinned cache open failed: \(coordinator.hub.message)") }
            try ModelHub.verify(folder, profile: profile)
            chat.openManaged(folder, profile: profile, ifSelected: chat.workspace.selected)
            let loaded = try wait()
            try require(loaded.acceptedIdentity?.profile == .qwen15
                && loaded.acceptedIdentity?.loadedLimit == 512
                && loaded.sessionConfiguration?.chunk == 32 && loaded.runtimePlan?.attention == .reference
                && chat.ready, "Wrong accepted Qwen identity or Reference plan")
            try record("loaded", ["admitted_bytes": loaded.admittedBytes,
                                   "repository": profile.repository, "revision": profile.revision, "downloaded": download])
            chat.setReplyLimit(32)
            let first = try send("Say hello in one short sentence.")
            let second = try send("Repeat your greeting in one short sentence.")
            try require(chat.workspace.current.messages.count == 4, "Continuation lost turns")
            try record("continuation", ["first_emitted": first.emitted, "second_emitted": second.emitted])
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

            chat.showPlanning()
            chat.editPlanning(.upper, "256")
            chat.editPlanning(.chunk, "32")
            chat.previewPlanning()
            let preview = try wait()
            try require(preview.preview?.capacity?.capacity.selected?.capacity == 256
                && chat.planningApplyUnavailableReason == nil, "Preview could not prepare the control request")
            chat.applyPlanning()
            let applied = try wait()
            try require(preview.preview?.capacity?.resources.attention == .reference
                && applied.transition?.committed == true
                && applied.sessionConfiguration?.capacity == 256
                && applied.modelContext?.modelInstance == preview.modelContext?.modelInstance
                && applied.history == 0 && chat.ready, "Reference replacement did not preserve the model")
            try record("applied", ["capacity": applied.sessionConfiguration?.capacity ?? 0])
            chat.unload()
            let unloaded = try wait()
            try require(!unloaded.hasSession && !unloaded.hasModel && !chat.ready, "Unload retained live state")
            try record("unloaded", [:])
            chat.openManaged(folder, profile: profile, ifSelected: chat.workspace.selected)
            let reopened = try wait()
            try require(reopened.acceptedIdentity?.namespace == loaded.acceptedIdentity?.namespace
                && reopened.acceptedIdentity?.loadedLimit == 512
                && reopened.acceptedIdentity?.modelInstance != loaded.acceptedIdentity?.modelInstance
                && reopened.sessionConfiguration?.capacity == 256
                && reopened.sessionConfiguration?.chunk == 32 && chat.ready,
                        "Reopen changed the accepted profile")
            chat.newChat()
            let reply = try send("Say hello one more time.")
            try record("reopened", ["emitted": reply.emitted, "capacity": reopened.sessionConfiguration?.capacity ?? 0])
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

import Foundation
import Combine

// One window owns one worker, Hub operation and history file. The Mac app does
// not map ordinary focus loss/file panels onto iOS cancellation or suspension.
@MainActor
final class MacCoordinator: ObservableObject {
    let harness: HarnessStore
    let chat: FriendlyChatStore
    let hub: HubStore
    private let trace: (String) -> Void
    func record(_ event: String) { trace(event) }
    @Published private(set) var shuttingDown = false
    private var terminating = false
    private var windowIntent = UUID()
    private(set) var windowOpenRequested = false
    private var shutdownTask: Task<Bool,Never>?

    static func dataRoot(arguments: [String],standard: URL) throws -> URL {
        let indices = arguments.indices.filter { arguments[$0] == "--data-root" }
        guard indices.count <= 1 else { throw HubFailure("Use --data-root only once.") }
        guard let index = indices.first else { return standard }
        guard index + 1 < arguments.count, arguments[index + 1].hasPrefix("/") else {
            throw HubFailure("--data-root requires an absolute local folder path.")
        }
        return URL(fileURLWithPath:arguments[index + 1],isDirectory:true).standardizedFileURL
    }
    // Reopening the single window receives a fresh worker after terminal close;
    // it never reuses a cached successful shutdown task for that new worker.
    func windowClosed() {
        record("window.intent.closed")
        windowOpenRequested = false; windowIntent = UUID()
    }
    func reopen() async {
        record("window.reopen.begin")
        guard !Task.isCancelled, !terminating else { record("window.reopen.refused"); return }
        let intent = UUID(); windowIntent = intent; windowOpenRequested = true
        if let shutdownTask { _ = await shutdownTask.value }
        guard !Task.isCancelled, !terminating, windowOpenRequested, windowIntent == intent else { record("window.reopen.stale"); return }
        if harness.worker?.snapshot().closed == true { harness.setExpert(harness.expert) }
        shutdownTask = nil; shuttingDown = false
        record("window.reopen.ready")
    }
    private func persistForQuit() -> Bool {
        // A damaged restored file blocked all editing and must stay untouched.
        guard chat.storageAvailable else { return true }
        guard chat.save() else {
            record("history.save.failed")
            chat.notice = "Your latest conversation could not be saved. Fix the storage problem and retry saving before quitting."
            return false
        }
        return true
    }
    init(root: URL,service: ModelHub? = nil,trace: @escaping (String) -> Void = { _ in }) {
        self.trace = trace
        harness = HarnessStore(enabled:true)
        chat = FriendlyChatStore(harness:harness,fileURL:root.appendingPathComponent("chat-v1.json"))
        hub = HubStore(hub:service ?? ModelHub(root:root.appendingPathComponent("Models",isDirectory:true)))
    }
    func shutdown(terminating: Bool = false) async -> Bool {
        record("shutdown.request.quit=\(terminating)")
        self.terminating = self.terminating || terminating
        windowClosed()
        if let shutdownTask { return await shutdownTask.value }
        shuttingDown = true
        let task = Task { @MainActor in
            hub.cancel()
            await hub.waitForIdle()
            guard let worker = harness.worker else { return persistForQuit() }
            worker.close()
            // waitUntilClosed belongs off the UI thread. Hub has already joined
            // its own cancelled transfer before the process may terminate.
            let closed = await withCheckedContinuation { continuation in
                DispatchQueue.global(qos:.userInitiated).async {
                    continuation.resume(returning:worker.waitUntilClosed(seconds:120))
                }
            }
            harness.poll(); chat.observe(harness.state)
            let saved = persistForQuit()
            if !closed { chat.notice = "The engine is still closing. Please wait and try quitting again." }
            return closed && saved
        }
        shutdownTask = task
        let result = await task.value
        record("shutdown.complete.success=\(result)")
        if !result {
            shutdownTask = nil; shuttingDown = false; self.terminating = false
            if harness.worker?.snapshot().closed == true { harness.setExpert(harness.expert) }
        }
        return result
    }
}

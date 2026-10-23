import Foundation

@MainActor
@main
struct MacMain {
    static func require(_ value: @autoclosure () throws -> Bool,_ label: String) throws {
        guard try value() else { throw HubFailure("Mac check failed: " + label) }
    }
    static func waitIdle(_ worker: NativeWorker) async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async { continuation.resume(returning:worker.waitUntilIdle(seconds:120) != nil) }
        }
    }
    static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletMacChecks-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        defer { try? FileManager.default.removeItem(at:root) }
        let standard = root.appendingPathComponent("standard")
        try require(try MacCoordinator.dataRoot(arguments:["Toklet"],standard:standard) == standard,"default root unchanged")
        for args in [["Toklet","--data-root"],["Toklet","--data-root","relative"],["Toklet","--data-root","/one","--data-root","/two"]] {
            var rejected = false
            do { _ = try MacCoordinator.dataRoot(arguments:args,standard:standard) } catch { rejected = true }
            try require(rejected,"malformed isolated root refused")
        }
        try require(!FileManager.default.fileExists(atPath:standard.path),"argument parsing creates no owner stores")
        actor Started {
            var started = false
            func signal() { started = true }
            func isStarted() -> Bool { started }
        }
        let started = Started()
        let service = ModelHub(root:root.appendingPathComponent("download/Models"),fetch:{ _,destination,_,_ in
            try Data("partial".utf8).write(to:destination,options:.withoutOverwriting)
            await started.signal()
            try await Task.sleep(for:.seconds(120))
        },freeBytes:{ _ in Int64.max / 2 })
        let download = MacCoordinator(root:root.appendingPathComponent("download"),service:service)
        download.chat.edit("Keep this Mac draft")
        download.hub.download()
        let end = ContinuousClock.now.advanced(by:.seconds(10))
        while !(await started.isStarted()), ContinuousClock.now < end { try await Task.sleep(for:.milliseconds(10)) }
        let didStart = await started.isStarted()
        try require(didStart,"download entered before quit")
        async let first = download.shutdown()
        async let second = download.shutdown()
        let values = await (first,second)
        try require(values.0 && values.1,"concurrent quit joins one shutdown")
        try require(!download.hub.busy && download.harness.worker?.snapshot().closed == true,"download and worker joined")
        try require(try FileManager.default.contentsOfDirectory(atPath:service.root.path).isEmpty,"incomplete download removed before quit completes")
        let saved = try ChatWorkspace.restore(Data(contentsOf:root.appendingPathComponent("download/chat-v1.json")))
        try require(saved.current.draft == "Keep this Mac draft","draft persisted on quit")
        let oldWorker = download.harness.worker
        await download.reopen()
        try require(download.harness.worker !== oldWorker && download.harness.worker?.snapshot().closed == false,"reopen uses fresh worker after terminal close")
        let reclosed = await download.shutdown()
        try require(reclosed && download.harness.worker?.snapshot().closed == true,"new worker is not covered by cached old shutdown")
        actor Drain {
            var entered = false; var cancelled = false; var released = false
            var waiter: CheckedContinuation<Void,Never>?
            func block() async {
                entered = true
                if !released { await withCheckedContinuation { waiter = $0 } }
            }
            func markCancelled() { cancelled = true }
            func release() { released = true; waiter?.resume(); waiter = nil }
        }
        let drain = Drain()
        let heldService = ModelHub(root:root.appendingPathComponent("held/Models"),fetch:{ _,destination,_,_ in
            try Data("partial".utf8).write(to:destination,options:.withoutOverwriting)
            try await withTaskCancellationHandler {
                await drain.block(); try Task.checkCancellation()
            } onCancel: { Task { await drain.markCancelled() } }
        },freeBytes:{ _ in Int64.max / 2 })
        let held = MacCoordinator(root:root.appendingPathComponent("held"),service:heldService)
        held.hub.download()
        let gateEnd = ContinuousClock.now.advanced(by:.seconds(10))
        while !(await drain.entered), ContinuousClock.now < gateEnd { try await Task.sleep(for:.milliseconds(10)) }
        let entered = await drain.entered; try require(entered,"held Hub transfer entered")
        let closing = Task { await held.shutdown() }
        while !(await drain.cancelled), ContinuousClock.now < gateEnd { try await Task.sleep(for:.milliseconds(10)) }
        let cancelled = await drain.cancelled; try require(cancelled,"shutdown held in cancelled Hub drain")
        let heldWorker = held.harness.worker
        let reopening = Task { await held.reopen() }
        while !held.windowOpenRequested, ContinuousClock.now < gateEnd { try await Task.sleep(for:.milliseconds(10)) }
        try require(held.windowOpenRequested,"reopen awaits joined shutdown")
        reopening.cancel(); held.windowClosed()
        let closingAgain = Task { await held.shutdown() }
        await drain.release()
        let firstClosed = await closing.value; let secondClosed = await closingAgain.value
        await reopening.value
        try require(firstClosed && secondClosed && !held.windowOpenRequested,"close-reopen-close intent remains closed")
        try require(held.harness.worker === heldWorker && heldWorker?.snapshot().closed == true,"cancelled reopen creates no replacement worker")
        let unwritable = root.appendingPathComponent("unwritable")
        try Data("keep parent".utf8).write(to:unwritable)
        let unsaved = MacCoordinator(root:unwritable)
        unsaved.chat.edit("Unsaved draft must survive failed quit")
        let failedQuit = await unsaved.shutdown(terminating:true)
        try require(!failedQuit && !unsaved.shuttingDown && !unsaved.chat.storageError.isEmpty,"write failure cancels termination with visible error")
        try require(unsaved.chat.workspace.current.draft == "Unsaved draft must survive failed quit","failed quit keeps unsaved state")
        try require(try Data(contentsOf:unwritable) == Data("keep parent".utf8),"failed quit preserves obstruction")
        try FileManager.default.removeItem(at:unwritable)
        let retried = await unsaved.shutdown(terminating:true)
        try require(retried,"quit can retry after storage repair")
        let recovered = try ChatWorkspace.restore(Data(contentsOf:unwritable.appendingPathComponent("chat-v1.json")))
        try require(recovered.current.draft == "Unsaved draft must survive failed quit","retry saves original draft")
        let reopened = MacCoordinator(root:root.appendingPathComponent("download"))
        try require(reopened.chat.workspace.current.draft == saved.current.draft && reopened.harness.worker?.snapshot().closed == false,"fresh process owner restores history with new worker")
        let reopenedClosed = await reopened.shutdown()
        try require(reopenedClosed,"reopened worker closes")
        let broken = root.appendingPathComponent("broken")
        try FileManager.default.createDirectory(at:broken,withIntermediateDirectories:false)
        let damaged = broken.appendingPathComponent("chat-v1.json")
        let bytes = Data("invalid history".utf8); try bytes.write(to:damaged)
        let blocked = MacCoordinator(root:broken)
        try require(!blocked.chat.storageAvailable,"damaged history blocks editing")
        let blockedClosed = await blocked.shutdown()
        try require(blockedClosed,"damaged history owner still closes")
        try require(try Data(contentsOf:damaged) == bytes,"quit never replaces damaged history")
        if CommandLine.arguments.count == 2 {
            let live = MacCoordinator(root:root.appendingPathComponent("generation"))
            guard let worker = live.harness.worker else { throw HubFailure("Mac worker unavailable") }
            live.chat.open(URL(fileURLWithPath:CommandLine.arguments[1]))
            let loaded = await waitIdle(worker)
        try require(loaded,"model opened")
            live.harness.poll(); live.chat.observe(live.harness.state)
            try require(live.chat.ready,"chat ready before active quit")
            live.chat.workspace.outputLimit = 128; live.chat.edit("Hello!"); live.chat.sendOrStop()
            try require(live.chat.running,"generation accepted before quit")
            let liveClosed = await live.shutdown()
        try require(liveClosed,"active generation joined on quit")
            try require(worker.snapshot().closed && !live.chat.running,"chat terminal observed and worker released")
            let state = try ChatWorkspace.restore(Data(contentsOf:root.appendingPathComponent("generation/chat-v1.json")))
            try require(state.current.messages.count == 2 && !state.current.messages.last!.writing,"saved response is terminal")
        }
        print("TOKLET_MAC_CHECKS_PASS")
    }
}

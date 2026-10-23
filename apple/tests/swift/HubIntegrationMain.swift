import Foundation

@MainActor
@main
struct HubIntegrationMain {
    static func main() async throws {
        if (3...4).contains(CommandLine.arguments.count) && CommandLine.arguments[1] == "--network" {
            guard let profile = HubProfile.named(CommandLine.arguments.count == 4 ? CommandLine.arguments[3] : "smol") else { throw HubFailure("Unknown test profile") }
            let result = try await runHubSelfTest(root:URL(fileURLWithPath:CommandLine.arguments[2]),profile:profile)
            print("TOKLET_HUB_CHECK_JSON=" + String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
            return
        }
        try await controllerChecks()
        print("TOKLET_HUB_CONTROLLER_PASS")
    }
    static func require(_ condition: @autoclosure () -> Bool,_ label: String) throws {
        guard condition() else { throw HubFailure("Controller check failed: " + label) }
    }
    static func controllerChecks() async throws {
        let budget = HubAdmission(unifiedGPU:true,physicalMemory:6 * 1024 * 1024 * 1024,availableMemory:1024 * 1024 * 1024)
        try budget.check(.smol)
        do { try budget.check(.qwen); throw HubFailure("Qwen admitted with Smol-only allowance") }
        catch let error as HubFailure where error.message.contains("Not enough memory") { }
        try require(HubModel(id:HubProfile.qwen.repository).profile == .qwen && HubModel(id:"mlx-community/Qwen2.5-0.5B-Instruct-4bit").profile == nil,"Qwen eight-bit only")
        for profile in HubProfile.catalog { try profile.validate() }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletHubController-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let model = HubProfile.smol
        let inventory = try JSONSerialization.data(withJSONObject:["sha":model.revision,"siblings":model.files.map { ["rfilename":$0.name,"size":$0.bytes] as [String:Any] }])
        let rows = Data("[{\"id\":\"\(model.repository)\"},{\"id\":\"other/unsupported\"}]".utf8)
        let admission: @Sendable () -> HubAdmission = { HubAdmission(unifiedGPU:true,physicalMemory:6 * 1024 * 1024 * 1024,availableMemory:1024 * 1024 * 1024) }
        final class Progress: @unchecked Sendable {
            let lock = NSLock(); private var callback: (@Sendable (Int64) -> Void)?
            func set(_ callback: @escaping @Sendable (Int64) -> Void) { lock.lock(); self.callback = callback; lock.unlock() }
            func emit() { lock.lock(); let callback = callback; lock.unlock(); callback?(999999) }
        }
        let progress = Progress()
        let hub = ModelHub(root:root.appendingPathComponent("normal"),fetch:{ url,destination,_,callback in
            if url.path == "/api/models" { try rows.write(to:destination) }
            else if url.path.contains("/api/models/") { try inventory.write(to:destination) }
            else { progress.set(callback); callback(42); throw HubFailure("Injected interrupted file") }
        },freeBytes:{ _ in Int64.max / 2 })
        let store = HubStore(hub:hub,admission:admission)
        store.select(.qwen); store.select(.smol); await store.waitForIdle()
        try require(store.profile == .qwen && !store.deviceAllowed,"busy profile selection ignored and per-profile admission")
        store.select(.smol); await store.waitForIdle()
        store.refresh(); await store.waitForIdle()
        try require(!store.busy && store.localURL == nil && store.deviceAllowed,"refresh absence and device")
        store.query = "SmolLM2"; store.search(); await store.waitForIdle()
        try require(store.results.count == 2 && store.results[0].profile == .smol && store.results[1].profile == nil,"search support separation")
        store.download(); store.search(); await store.waitForIdle()
        try require(!store.busy && store.message == "Injected interrupted file" && store.localURL == nil,"duplicate ignored and failed download not exposed")
        store.search(); progress.emit(); await store.waitForIdle()
        await Task.yield()
        try require(store.completed == 0 && store.total == 0,"late old-operation progress ignored")
        var opened = false
        store.open { _ in opened = true }; await store.waitForIdle()
        try require(!opened && store.localURL == nil && !store.message.isEmpty,"missing snapshot cannot open")
        let denied = HubStore(hub:hub,admission:{ HubAdmission(unifiedGPU:false,physicalMemory:0,availableMemory:0) })
        denied.download(); await denied.waitForIdle()
        try require(!denied.deviceAllowed && !denied.message.isEmpty,"fresh device admission")
        actor Started {
            var continuation: CheckedContinuation<Void,Never>?
            var started = false
            func signal() { started = true; continuation?.resume(); continuation = nil }
            func wait() async { if !started { await withCheckedContinuation { continuation = $0 } } }
        }
        actor VerificationGate {
            var started = false
            var startWaiter: CheckedContinuation<Void,Never>?
            var release: CheckedContinuation<Void,Never>?
            func verify() async -> URL? {
                started = true; startWaiter?.resume(); startWaiter = nil
                await withCheckedContinuation { release = $0 }
                return URL(fileURLWithPath:"/verified-controller-fixture")
            }
            func wait() async { if !started { await withCheckedContinuation { startWaiter = $0 } } }
            func finish() { release?.resume(); release = nil }
        }
        for choice in ["expert","another-model","unloaded","unchanged"] {
            let gate = VerificationGate()
            let delayed = HubStore(hub:hub,admission:admission,verify:{ _ in await gate.verify() })
            var loads = 0; var selected = "original"
            delayed.open { _ in loads += 1; selected = "downloaded-model" }
            await gate.wait()
            if choice != "unchanged" { delayed.modelChoiceChanged(); selected = choice }
            await gate.finish(); await delayed.waitForIdle()
            if choice == "unchanged" { try require(loads == 1 && selected == "downloaded-model","baseline verified open reaches worker boundary") }
            else {
                try require(loads == 0 && selected == choice && delayed.message.contains("cancelled"),"no late load after " + choice)
            }
        }
        let started = Started()
        let slow = ModelHub(root:root.appendingPathComponent("cancel"),fetch:{ _,_,_,_ in
            await started.signal(); try await Task.sleep(for:.seconds(60))
        },freeBytes:{ _ in Int64.max / 2 })
        let cancelled = HubStore(hub:slow,admission:admission)
        cancelled.download(); await started.wait(); cancelled.cancel(); await cancelled.waitForIdle()
        try require(!cancelled.busy && cancelled.localURL == nil && cancelled.message.contains("cancelled"),"background/explicit cancellation joins operation")
    }
}

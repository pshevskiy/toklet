import Foundation
import Combine
import Metal
import Darwin
#if os(iOS)
import os
#endif

struct HubDeviceSnapshot: Codable, Sendable {
    let unifiedGPU: Bool
    let physicalMemory: UInt64
    let availableMemory: UInt64
    let instrument: String
    var admission: HubAdmission { HubAdmission(unifiedGPU:unifiedGPU,physicalMemory:physicalMemory,availableMemory:availableMemory) }
    static func current() throws -> HubDeviceSnapshot {
        let unified = MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true
        let physical = ProcessInfo.processInfo.physicalMemory
        #if os(iOS)
        let available = UInt64(os_proc_available_memory())
        let instrument = "os_proc_available_memory: current process dirty-memory allowance"
        #else
        let budget = min(physical / 2,6 * 1024 * 1024 * 1024)
        let footprint = try NativeFootprint.read().physicalBytes
        let available = budget > footprint ? budget - footprint : 0
        let instrument = "Configured Mac process allowance min(RAM/2,6GiB) minus TASK_VM_INFO footprint; not free system RAM"
        #endif
        return HubDeviceSnapshot(unifiedGPU:unified,physicalMemory:physical,availableMemory:available,instrument:instrument)
    }
}

@MainActor
final class HubStore: ObservableObject {
    @Published var query = ""
    @Published private(set) var profile: HubProfile
    @Published private(set) var results: [HubModel] = []
    @Published private(set) var phase = ""
    @Published private(set) var message = ""
    @Published private(set) var completed: Int64 = 0
    @Published private(set) var total: Int64 = 0
    @Published private(set) var localURL: URL?
    @Published private(set) var deviceMessage = ""
    @Published private(set) var deviceAllowed = false
    private let hub: ModelHub
    private let admission: @Sendable () throws -> HubAdmission
    private let verify: @Sendable (HubProfile) async throws -> URL?
    private var task: Task<Void,Never>?
    private var operation = UUID()
    var busy: Bool { task != nil }
    var downloading: Bool { total > 0 && busy }
    static var modelRoot: URL {
        FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0].appendingPathComponent("TokletModels",isDirectory:true)
    }
    init(profile: HubProfile = .smol,hub: ModelHub? = nil,admission: @escaping @Sendable () throws -> HubAdmission = { try HubDeviceSnapshot.current().admission },
         verify: (@Sendable (HubProfile) async throws -> URL?)? = nil) {
        let service = hub ?? ModelHub(root:Self.modelRoot)
        self.profile = profile; self.hub = service; self.admission = admission
        self.verify = verify ?? { try await service.verified($0) }
    }
    private enum Action { case refresh, search(String), download, open }
    func select(_ selected: HubProfile) {
        guard !busy, HubProfile.catalog.contains(selected) else { return }
        profile = selected; localURL = nil; deviceAllowed = false; refresh()
    }
    func refresh() { start(.refresh) }
    func search() { start(.search(query)) }
    func download() { start(.download) }
    func open(_ completion: @escaping @MainActor (URL) -> Void) { start(.open,opened:completion) }
    func cancel() {
        guard let task else { return }
        operation = UUID(); phase = "Stopping…"; task.cancel()
    }
    func modelChoiceChanged() { cancel() }
    func waitForIdle() async { await task?.value }
    private func checkDevice() throws -> HubAdmission {
        let value = try admission()
        do { try value.check(profile); deviceAllowed = true; deviceMessage = "Estimated fit on this device. Opening checks memory again." }
        catch { deviceAllowed = false; deviceMessage = error.localizedDescription; throw error }
        return value
    }
    private func start(_ action: Action,opened: (@MainActor (URL) -> Void)? = nil) {
        guard task == nil else { return }
        operation = UUID(); let id = operation
        completed = 0; total = 0; message = ""
        switch action {
        case .refresh: phase = "Checking saved model…"
        case .search: phase = "Searching Hugging Face…"
        case .download: phase = "Downloading model…"; total = profile.totalBytes
        case .open: phase = "Verifying model…"
        }
        let selected = profile
        task = Task { [self] in
            defer { task = nil; phase = "" }
            do {
                try Task.checkCancellation()
                switch action {
                case .refresh:
                    _ = try? checkDevice()
                    let folder = try await verify(selected)
                    try Task.checkCancellation(); localURL = folder
                case .search(let query):
                    let rows = try await hub.search(query)
                    try Task.checkCancellation(); results = rows
                    if rows.isEmpty { message = "No public MLX models found. Try a different name." }
                case .download:
                    let device = try checkDevice()
                    let folder = try await hub.download(selected,admission:device,progress:{ [weak self] current,total in
                        Task { @MainActor [weak self] in
                            guard let self, self.operation == id, self.task != nil else { return }
                            self.completed = current; self.total = total
                        }
                    })
                    try Task.checkCancellation(); localURL = folder
                    message = "Download verified. Open the model to start chatting."
                case .open:
                    _ = try checkDevice()
                    guard let folder = try await verify(selected) else { throw HubFailure("Download this model first.") }
                    try Task.checkCancellation(); _ = try checkDevice()
                    guard operation == id else { throw CancellationError() }
                    localURL = folder; opened?(folder)
                    message = "Model verified."
                }
            } catch is CancellationError { message = "Operation cancelled. Any complete saved model is kept." }
            catch {
                if case .refresh = action { localURL = nil }
                if case .open = action { localURL = nil }
                message = error.localizedDescription
            }
        }
    }
}

struct HubSelfTestResult: Codable {
    let passed: Bool
    let repository: String
    let revision: String
    let snapshotBytes: Int64
    let reusedSnapshot: Bool
    let offlineController: HubControllerCheck
    let searchFoundProfile: Bool
    let savedSnapshotVerified: Bool
    let device: HubDeviceSnapshot
    let chat: NativeChatCheck
}

func runHubSelfTest(root: URL,profile: HubProfile = .smol) async throws -> HubSelfTestResult {
    let hub = ModelHub(root:root)
    let rows = try await hub.search(profile.key == "qwen" ? "Qwen2.5-0.5B" : "SmolLM2-135M")
    guard rows.contains(where:{ $0.id == profile.repository }) else { throw HubFailure("Search did not return the qualified profile.") }
    let existing = try await hub.verified(profile)
    let device = try HubDeviceSnapshot.current()
    let folder = try await hub.download(profile,admission:device.admission,progress:{ _,_ in })
    guard let verified = try await hub.verified(profile), verified.path == folder.path else { throw HubFailure("Downloaded model failed offline validation.") }
    try Task.checkCancellation()
    try HubDeviceSnapshot.current().admission.check(profile)
    let chat = try runNativeChatCheck(directory:folder.path,profile:profile)
    let controller = try await checkDownloadedHubController(root:root,expected:chat.replies[0],profile:profile)
    return HubSelfTestResult(passed:true,repository:profile.repository,revision:profile.revision,
        snapshotBytes:profile.totalBytes,reusedSnapshot:existing != nil,offlineController:controller,
        searchFoundProfile:true,savedSnapshotVerified:true,device:device,chat:chat)
}

struct HubControllerCheck: Codable, Sendable {
    let passed: Bool
    let text: String
    let ids: [UInt32]
    let released: Bool
}

// Explicit diagnostic only. Uses a fresh temporary history and refuses all
// network calls while exercising the actual downloaded-model UI controllers.
@MainActor
func checkDownloadedHubController(root: URL,expected: ChatCheckReply,profile: HubProfile = .smol) async throws -> HubControllerCheck {
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("TokletHubChat-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:temporary) }
    let hub = HubStore(profile:profile,hub:ModelHub(root:root,fetch:{ _,_,_,_ in throw HubFailure("Offline controller attempted network access.") }))
    let harness = HarnessStore(enabled:true)
    guard let worker = harness.worker else { throw HubFailure("Missing controller worker.") }
    defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
    let chat = FriendlyChatStore(harness:harness,fileURL:temporary.appendingPathComponent("history.json"))
    hub.refresh(); await hub.waitForIdle()
    guard hub.localURL != nil, hub.deviceAllowed else { throw HubFailure("Downloaded folder or device not admitted by controller: " + hub.message) }
    hub.open { folder in chat.open(folder) }; await hub.waitForIdle()
    guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Downloaded model load timed out.") }
    harness.poll(); chat.observe(harness.state)
    guard chat.ready else { throw HubFailure("Downloaded model did not load through controller: " + harness.state.message) }
    chat.workspace.outputLimit = 32; chat.edit("Hello!"); chat.sendOrStop()
    guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Controller reply timed out.") }
    harness.poll(); chat.observe(harness.state)
    let text = chat.workspace.current.messages.last?.text ?? ""
    let ids = harness.state.ids
    guard !chat.running, text.hasPrefix(expected.text), Array(ids.prefix(expected.ids.count)) == expected.ids else { throw HubFailure("Controller reply differs from native text reference.") }
    chat.unload()
    guard worker.waitUntilIdle(seconds:30) != nil else { throw HubFailure("Controller unload timed out.") }
    worker.close()
    guard worker.waitUntilClosed(seconds:30), !worker.snapshot().hasModel else { throw HubFailure("Controller resources remain live.") }
    return HubControllerCheck(passed:true,text:text,ids:ids,released:true)
}

// A fresh-transfer witness owns only this newly created temporary tree. It
// never renames, removes or replaces the normal downloaded-model directory.
func runFreshHubSelfTest(profile: HubProfile = .smol) async throws -> HubSelfTestResult {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletHubDownload-" + UUID().uuidString,isDirectory:true)
    guard Darwin.mkdir(root.path,0o700) == 0 else { throw HubFailure("Could not create exclusive download-test directory.") }
    do {
        let result = try await runHubSelfTest(root:root,profile:profile)
        guard !result.reusedSnapshot else { throw HubFailure("Fresh download test unexpectedly reused a snapshot.") }
        try FileManager.default.removeItem(at:root)
        return result
    } catch {
        let original = error
        do {
            if FileManager.default.fileExists(atPath:root.path) { try FileManager.default.removeItem(at:root) }
        } catch { throw HubFailure("Download test failed: " + original.localizedDescription + "; temporary cleanup failed: " + error.localizedDescription) }
        throw original
    }
}

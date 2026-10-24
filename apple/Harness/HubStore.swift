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
    @Published private(set) var blocksSelection = false
    @Published private(set) var opening = false
    @Published private(set) var savedProfiles: Set<String> = []
    private let hub: ModelHub
    let modelsRoot: URL
    private let admission: @Sendable () throws -> HubAdmission
    private let verify: @Sendable (HubProfile) async throws -> URL?
    private var task: Task<Void,Never>?
    private var operation = UUID()
    private var cancelling = false
    var busy: Bool { task != nil }
    var downloading: Bool { total > 0 && busy }
    static var modelRoot: URL {
        FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0].appendingPathComponent("TokletModels",isDirectory:true)
    }
    init(profile: HubProfile = .smol,hub: ModelHub? = nil,admission: @escaping @Sendable () throws -> HubAdmission = { try HubDeviceSnapshot.current().admission },
         verify: (@Sendable (HubProfile) async throws -> URL?)? = nil) {
        let service = hub ?? ModelHub(root:Self.modelRoot)
        self.profile = profile; self.hub = service; self.admission = admission
        modelsRoot = hub == nil ? Self.modelRoot : service.root
        self.verify = verify ?? { try await service.verified($0) }
    }
    private enum Action { case inventory, refresh, search(String), download, open, use(replacing: Bool) }
    func select(_ selected: HubProfile) {
        guard !blocksSelection, HubProfile.catalog.contains(selected) else { return }
        profile = selected; localURL = nil; deviceAllowed = false; refresh()
    }
    func refresh() { start(.refresh) }
    func inventory() { start(.inventory) }
    func use(_ selected: HubProfile,replacingLoadedModel: Bool = false,opened: @escaping @MainActor (URL) -> Void) {
        guard !blocksSelection, HubProfile.catalog.contains(selected) else { return }
        profile = selected; localURL = nil; deviceAllowed = false
        start(.use(replacing:replacingLoadedModel),opened:opened)
    }
    func downloadAndUse(_ selected: HubProfile,opened: @escaping @MainActor (URL) -> Void) {
        guard !blocksSelection, HubProfile.catalog.contains(selected) else { return }
        profile = selected; localURL = nil; deviceAllowed = false
        start(.download,opened:opened)
    }
    func search() { start(.search(query)) }
    func download() { start(.download) }
    func open(_ completion: @escaping @MainActor (URL) -> Void) { start(.open,opened:completion) }
    func cancel() {
        guard let task else { return }
        cancelling = true; phase = "Stopping…"; task.cancel()
    }
    func modelChoiceChanged() { cancel() }
    func waitForIdle() async { await task?.value }
    private func checkDevice() throws -> HubAdmission {
        do {
            let value = try admission()
            try value.check(profile)
            deviceAllowed = true; deviceMessage = "Estimated fit on this device. Opening checks memory again."
            return value
        } catch { deviceAllowed = false; deviceMessage = error.localizedDescription; throw error }
    }

    private func start(_ action: Action,opened: (@MainActor (URL) -> Void)? = nil) {
        guard !blocksSelection else { return }
        // A slow automatic hash check must not lock search or model selection.
        // Join the cancelled operation before using the service again, and only
        // publish results for the latest choice. Downloads/opens stay exclusive.
        let previous = task
        previous?.cancel()
        operation = UUID(); let id = operation
        cancelling = false
        completed = 0; total = 0; message = ""
        switch action { case .open, .use: opening = true; default: opening = false }
        switch action {
        case .inventory: phase = "Checking saved models…"
        case .refresh: phase = "Checking saved model…"
        case .search(let query):
            phase = "Searching Hugging Face…"
            results = HubProfile.matching(query).map { HubModel(id:$0.repository) }
        case .download: phase = "Downloading model…"; total = profile.totalBytes; blocksSelection = true
        case .open: phase = "Verifying model…"; blocksSelection = true
        case .use: phase = "Opening " + profile.name + "…"
        }
        let selected = profile
        task = Task { [self] in
            defer { if operation == id { task = nil; phase = ""; blocksSelection = false; opening = false } }
            await previous?.value
            do {
                try Task.checkCancellation()
                guard operation == id else { throw CancellationError() }
                switch action {
                case .inventory:
                    let saved = try await hub.savedProfiles()
                    try Task.checkCancellation()
                    guard operation == id else { throw CancellationError() }
                    savedProfiles = saved
                case .refresh:
                    let folder = try await verify(selected)
                    try Task.checkCancellation()
                    guard operation == id else { throw CancellationError() }
                    localURL = folder
                    // Verification can outlast unloading a previous GPU model.
                    // Display current admission after the await, just as open does.
                    _ = try? checkDevice()
                case .search(let query):
                    let rows = try await hub.search(query)
                    try Task.checkCancellation()
                    guard operation == id else { throw CancellationError() }
                    let supported = HubProfile.matching(query).map { HubModel(id:$0.repository) }
                    let pinned = Set(supported.map(\.id))
                    results = supported + rows.filter { !pinned.contains($0.id) }
                    if results.isEmpty { message = "No public MLX models found. Try a different name." }
                case .download:
                    let device = try checkDevice()
                    let folder = try await hub.download(selected,admission:device,progress:{ [weak self] current,total in
                        Task { @MainActor [weak self] in
                            guard let self, self.operation == id, self.task != nil, !self.cancelling else { return }
                            self.completed = current; self.total = total
                        }
                    })
                    try Task.checkCancellation(); localURL = folder
                    savedProfiles.insert(selected.repository)
                    if let opened { opened(folder); message = "" }
                    else { message = "Download verified. Open the model to start chatting." }
                case .open, .use:
                    let replacing: Bool
                    if case .use(let value) = action { replacing = value } else { replacing = false }
                    // A replacement releases its old numerical owner before
                    // beginLoad checks fresh memory. Remaining memory while
                    // that owner is resident is not the replacement budget.
                    if !replacing { _ = try checkDevice() }
                    guard let folder = try await verify(selected) else { throw HubFailure("Download this model first.") }
                    try Task.checkCancellation()
                    if !replacing { _ = try checkDevice() }
                    guard operation == id else { throw CancellationError() }
                    localURL = folder; savedProfiles.insert(selected.repository); opened?(folder)
                    message = ""
                }
            } catch is CancellationError {
                // cancel() also invalidates callbacks, but keeps ownership until
                // the task joins. Replaced tasks cannot clear the latest state.
                if operation == id { message = "Operation cancelled. Any complete saved model is kept." }
            }
            catch {
                guard operation == id else { return }
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
    let rows = try await hub.search(profile.searchQuery)
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
    guard hub.localURL != nil, hub.deviceAllowed else { throw HubFailure("Downloaded folder or device not admitted by controller: " + hub.message + "; " + hub.deviceMessage) }
    hub.open { folder in chat.open(folder) }; await hub.waitForIdle()
    guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Downloaded model load timed out.") }
    harness.poll(); chat.observe(harness.state)
    guard chat.ready && chat.modelName == profile.name else { throw HubFailure("Downloaded model did not load through controller: " + harness.state.message) }
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

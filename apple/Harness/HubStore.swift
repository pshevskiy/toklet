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
        let headroom = try NativeFootprint.readHeadroom()
        return HubDeviceSnapshot(unifiedGPU:unified,physicalMemory:physical,
            availableMemory:headroom.availableBytes,instrument:headroom.instrument)
    }
}

enum HubRankingTask: String, CaseIterable, Hashable, Sendable {
    case balanced = "Balanced", coding = "Coding", fast = "Fast replies"
    var explanation: String {
        switch self {
        case .balanced: "Balances model capacity, estimated speed and memory headroom."
        case .coding: "Favors model capacity. Coding quality is not measured by this score."
        case .fast: "Favors smaller models for faster replies, with memory headroom."
        }
    }
    // Capacity / speed / memory. These express preferences, not measured quality.
    var weights: (Double,Double,Double) {
        switch self {
        case .balanced: (0.65,0.15,0.20)
        case .coding: (0.80,0.05,0.15)
        case .fast: (0.15,0.65,0.20)
        }
    }
}

struct HubRecommendation: Identifiable, Equatable, Sendable {
    let profile: HubProfile
    let capacity, speed, headroom, score: Double
    let active: Bool
    var id: String { profile.repository }
    var displayScore: Int { Int(score.rounded()) }
    var breakdown: String {
        "Capacity \(Int(capacity.rounded())) · Speed \(Int(speed.rounded())) · Headroom \(Int(headroom.rounded()))"
    }
}

enum HubRecommendations {
    // Rank only the pinned Chat catalog admitted by the existing device policy.
    // Native compatibility, exact allocation planning and loading remain final.
    // A resident model is included without demanding space for a second copy.
    static func ranked(_ admission: HubAdmission,task: HubRankingTask = .balanced,active: HubProfile? = nil) -> [HubRecommendation] {
        guard admission.unifiedGPU, admission.physicalMemory >= 2 * 1024 * 1024 * 1024 else { return [] }
        let smallest = Double(HubProfile.catalog.map(\.totalBytes).min() ?? 1)
        let largest = Double(HubProfile.catalog.map(\.totalBytes).max() ?? 1)
        let range = log(max(largest / smallest,1))
        let weights = task.weights
        return Array(HubProfile.catalog.filter { profile in
            profile == active || (try? admission.check(profile)) != nil
        }.map { profile in
            let bytes = Double(profile.totalBytes)
            // Relative stored-weight size is a capacity proxy, not answer quality.
            let capacity = range > 0 ? 35 + 65 * log(bytes / smallest) / range : 50
            // Relative bandwidth pressure only; no fabricated tok/s or TTFT.
            let speed = 100 * sqrt(smallest / bytes)
            let available = Double(admission.availableMemory)
            let required = Double(profile.processAllowance)
            // Conservative even for a resident profile: low free allowance is
            // never shown as perfect headroom because loading is skipped.
            let headroom = available > 0 ? 100 * max(0,available - required) / available : 0
            let score = capacity * weights.0 + speed * weights.1 + headroom * weights.2
            return HubRecommendation(profile:profile,capacity:capacity,speed:speed,headroom:headroom,score:score,active:profile == active)
        }.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.id < $1.id
        }.prefix(5))
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
    @Published private(set) var recommendations: [HubRecommendation] = []
    @Published private(set) var recommendationTask: HubRankingTask = .balanced
    @Published private(set) var recommendationMessage = ""
    @Published private(set) var discoveredModels: [HubModel] = []
    @Published private(set) var discoveryMessage = ""
    @Published private(set) var findingBest = false
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
    private enum Action { case inventory, refresh, search(String), discover(HubProfile?), download, open, use(replacing: Bool) }
    func select(_ selected: HubProfile) {
        guard !blocksSelection, HubProfile.catalog.contains(selected) else { return }
        profile = selected; localURL = nil; deviceAllowed = false; refresh()
    }
    func refresh() { start(.refresh) }
    // Opening the sheet must not replace a discovery started by its first tap.
    func inventory() { if !busy { start(.inventory) } }
    func findBest(task: HubRankingTask? = nil,active: HubProfile? = nil) {
        guard !blocksSelection else { return }
        if let task {
            recommendationTask = task
            rank(active:active)
            // Preferences rerank with fresh memory without restarting the web
            // request or cancelling a download/verification already in flight.
            if !discoveryMessage.isEmpty { return }
        }
        rank(active:active)
        start(.discover(active))
    }
    private func rank(active: HubProfile?) {
        do {
            recommendations = HubRecommendations.ranked(try admission(),task:recommendationTask,active:active)
            recommendationMessage = recommendations.isEmpty
                ? "No supported models fit the current memory allowance. Unload a model or close other apps, then try again."
                : "\(recommendations.count) supported model(s) available. \(recommendationTask.explanation) Up to 5 results."
            if active != nil { recommendationMessage += " The active model is included. Unload it to assess more replacements." }
            if HubProfile.catalog.count < 5 { recommendationMessage += " Toklet currently supports \(HubProfile.catalog.count) downloadable chat models." }
        } catch {
            recommendations = []; recommendationMessage = "Could not check this device: " + error.localizedDescription
        }
    }
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
        if findingBest { discoveryMessage = "Hugging Face search stopped. Tap Find best model to retry." }
        findingBest = false
        switch action {
        case .inventory: phase = "Checking saved models…"
        case .refresh: phase = "Checking saved model…"
        case .search(let query):
            phase = "Searching Hugging Face…"
            results = HubProfile.matching(query).map { HubModel(id:$0.repository) }
        case .discover:
            findingBest = true
            phase = "Searching Hugging Face…"
            discoveryMessage = "Searching Hugging Face… Supported models are shown below while searching."
            discoveredModels = []
        case .download: phase = "Downloading model…"; total = profile.totalBytes; blocksSelection = true
        case .open: phase = "Verifying model…"; blocksSelection = true
        case .use: phase = "Opening " + profile.name + "…"
        }
        let selected = profile
        task = Task { [self] in
            defer { if operation == id { task = nil; phase = ""; blocksSelection = false; opening = false; findingBest = false } }
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
                case .discover(let active):
                    let saved = try await hub.savedProfiles()
                    try Task.checkCancellation()
                    guard operation == id else { throw CancellationError() }
                    savedProfiles = saved
                    let discovery = try await hub.discover()
                    try Task.checkCancellation()
                    guard operation == id else { throw CancellationError() }
                    discoveredModels = discovery.models
                    discoveryMessage = "Hugging Face: \(discovery.models.count) public MLX models found. "
                        + (discovery.models.isEmpty ? "Try searching by name below. " : "Compatibility and memory determine which can run. ")
                        + (discovery.failedRequests > 0 ? "Some searches failed; tap Find best model to retry." : "Updated " + Date().formatted(date:.omitted,time:.shortened) + ".")
                    rank(active:active)
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
                if operation == id {
                    message = "Operation cancelled. Any complete saved model is kept."
                    if case .discover = action { discoveryMessage = "Hugging Face search stopped. Tap Find best model to retry." }
                }
            }
            catch {
                guard operation == id else { return }
                if case .refresh = action { localURL = nil }
                if case .open = action { localURL = nil }
                if case .discover = action {
                    discoveryMessage = "Hugging Face search failed. Showing the supported offline catalog. " + error.localizedDescription
                }
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
    let memory: ChatCheckMemory
}

// Explicit diagnostic only. Uses a fresh temporary history and refuses all
// network calls while exercising the actual downloaded-model UI controllers.
@MainActor
func checkDownloadedHubController(root: URL,expected: ChatCheckReply,profile: HubProfile = .smol) async throws -> HubControllerCheck {
    let memoryBefore = try NativeFootprint.read()
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
    let memoryLoaded = try NativeFootprint.read()
    chat.workspace.outputLimit = 32; chat.edit("Hello!"); chat.sendOrStop()
    guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Controller reply timed out.") }
    harness.poll(); chat.observe(harness.state)
    let text = chat.workspace.current.messages.last?.text ?? ""
    let ids = harness.state.ids
    guard !chat.running, text.hasPrefix(expected.text), Array(ids.prefix(expected.ids.count)) == expected.ids else { throw HubFailure("Controller reply differs from native text reference.") }
    let memoryGenerated = try NativeFootprint.read()
    chat.unload()
    guard worker.waitUntilIdle(seconds:30) != nil else { throw HubFailure("Controller unload timed out.") }
    worker.close()
    guard worker.waitUntilClosed(seconds:30), !worker.snapshot().hasModel else { throw HubFailure("Controller resources remain live.") }
    let memoryReleased = try NativeFootprint.read()
    return HubControllerCheck(passed:true,text:text,ids:ids,released:true,
        memory:ChatCheckMemory(before:memoryBefore,loaded:memoryLoaded,generated:memoryGenerated,released:memoryReleased))
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

import Foundation
import Combine

struct LocalBenchmarkSample: Identifiable, Sendable {
    let id: Int
    let metrics: GenerationMetrics
    let stopReason: UInt32
}

@MainActor
final class FriendlyChatStore: ObservableObject {
    @Published var workspace = ChatWorkspace()
    @Published private(set) var modelName = "Toklet"
    @Published private(set) var loadingModel = false
    @Published private(set) var loadingName = ""
    @Published var notice = ""
    @Published var storageError = ""
    @Published var folders: [URL] = []
    @Published private(set) var benchmarkRunning = false
    @Published private(set) var benchmarkStatus = ""
    @Published private(set) var benchmarkModel = ""
    @Published private(set) var benchmarkSamples: [LocalBenchmarkSample] = []
    private var benchmarkRequest: UUID?
    private var benchmarkStep = 0
    private var benchmarkStopping = false
    // A fixed offline workload, independent of all owner history and drafts.
    static let benchmarkPrompt = "Write a detailed paragraph about the seasons of the year. Describe spring, summer, autumn and winter."
    static let benchmarkMaximum: UInt32 = 64
    private weak var harness: HarnessStore?
    private var request: UUID?
    private var draftEdit = UUID()
    private var requestDraftEdit: UUID?
    private var pendingFolder: URL?
    private var scopedFolder: URL?
    private var loadedKey = ""
    private var loadedModelInstance: UInt64?
    // Exact accepted load input, separate from catalog/path guesses or security scope.
    private var planningLoad: FriendlyPlanningLoad?
    private var planningVisible = false
    private weak var planningWorker: NativeWorker?
    @Published private(set) var planning = FriendlyPlanningState()
    @Published private(set) var applying = FriendlyApplyState()
    @Published private(set) var currentSessionConfiguration: NativeSessionConfiguration?
    private var saving: Task<Void,Never>?
    private var storageBlocked = false
    private let file: URL
    private let admission: () throws -> HubAdmission
    private let persist: (Data,URL) throws -> Void

    init(harness: HarnessStore, fileURL: URL? = nil, admission: @escaping () throws -> HubAdmission = { try HubDeviceSnapshot.current().admission },
         persist: ((Data,URL) throws -> Void)? = nil) {
        self.harness = harness
        self.admission = admission
        self.persist = persist ?? { bytes,file in
            #if os(iOS)
            try bytes.write(to:file,options:[.atomic,.completeFileProtectionUntilFirstUserAuthentication])
            #else
            try bytes.write(to:file,options:.atomic)
            #endif
        }
        file = fileURL ?? FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("Toklet/chat-v1.json")
        restore()
        refreshFolders()
    }
    var running: Bool { request != nil || benchmarkRunning }
    var storageAvailable: Bool { !storageBlocked }
    var hasChatModel: Bool {
        guard let harness else { return false }
        return harness.state.hasModel && harness.state.chatEnabled && loadedKey == workspace.current.modelKey
            && loadedModelInstance != nil && harness.state.modelContext?.modelInstance == loadedModelInstance
    }
    var ready: Bool {
        guard let harness else { return false }
        return hasChatModel && harness.state.modelContext != nil && !loadingModel && !harness.state.closed && !harness.state.restartRequired && !harness.state.busy && !running
    }
    var availableOutputLimits: [UInt32] {
        // Before choosing/opening this conversation's model these are saved
        // preference choices, not advertised model capability or admission.
        guard hasChatModel else {
            return ChatWorkspace.outputLimits.filter { $0 <= NativeGenerationPolicy.maximumChatOutput }
        }
        guard let context = harness?.state.modelContext else { return [] }
        let limits = NativeGenerationPolicy.limits(.chat,model:context)
        return ChatWorkspace.outputLimits.filter { UInt64($0) <= limits.maximumOutput }
    }
    var outputLimitAvailable: Bool { availableOutputLimits.contains(workspace.outputLimit) }
    var canSend: Bool { ready && outputLimitAvailable && !workspace.current.draft.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && !storageBlocked }
    private var documents: URL { FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0] }
    private func key(_ url: URL) -> String {
        let prefix = documents.path + "/"
        return url.path.hasPrefix(prefix) ? "documents:" + url.path.dropFirst(prefix.count) : url.path
    }
    func restore() {
        do {
            if FileManager.default.fileExists(atPath:file.path) {
                let handle = try FileHandle(forReadingFrom:file)
                defer { try? handle.close() }
                workspace = try ChatWorkspace.restore(handle.read(upToCount:ChatWorkspace.maximumBytes + 1) ?? Data())
            }
            storageBlocked = false; storageError = ""
        } catch { storageBlocked = true; storageError = "Could not open saved chats: " + error.localizedDescription }
    }
    @discardableResult
    func save() -> Bool {
        saving?.cancel()
        guard !storageBlocked else { return false }
        do {
            let bytes = try JSONEncoder().encode(workspace)
            guard bytes.count <= ChatWorkspace.maximumBytes else { throw chatInputError("Chat history is full") }
            try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
            try persist(bytes,file)
            storageError = ""; return true
        } catch { storageError = "Could not save chats: " + error.localizedDescription; return false }
    }
    func retryStorage() { if storageBlocked { restore() } else { save() } }
    func edit(_ text: String) {
        guard !storageBlocked else { return }
        guard text.utf8.count <= NativeGenerationPolicy.maximumDraftBytes else { notice = "A message can contain up to 16 KiB of text."; return }
        draftEdit = UUID()
        workspace.edit(text)
        saving?.cancel()
        saving = Task { [weak self] in
            do { try await Task.sleep(for:.milliseconds(400)) } catch { return }
            self?.save()
        }
    }
    func newChat() {
        guard !storageBlocked, !running, !loadingModel, harness?.state.busy == false else { return }
        do { try workspace.newChat(modelKey:activeModelKey); invalidatePlanning(); refreshPlanning(); notice = ""; save() }
        catch { notice = error.localizedDescription }
    }
    func select(_ id: UUID) {
        guard !storageBlocked, !running, !loadingModel, harness?.state.busy == false, workspace.conversations.contains(where: { $0.id == id }) else { return }
        workspace.selected = id; invalidatePlanning(); refreshPlanning(); notice = ""; save()
    }
    func remove(_ id: UUID) {
        guard !storageBlocked, !running, !loadingModel, harness?.state.busy == false else { return }
        let previous = workspace
        let last = workspace.conversations.count == 1 && workspace.current.id == id
        workspace.remove(id)
        if last, let activeModelKey { workspace.conversations[workspace.selectedIndex].modelKey = activeModelKey }
        finishDeletion(previous)
    }
    func removeAll() {
        guard !storageBlocked, !running, !loadingModel, harness?.state.busy == false else { return }
        let previous = workspace
        workspace.removeAll(modelKey:activeModelKey); finishDeletion(previous)
    }
    private func finishDeletion(_ previous: ChatWorkspace) {
        invalidatePlanning()
        if save() { notice = "" }
        else {
            workspace = previous
            notice = "Your conversations were kept because saving failed. Please try deleting again."
        }
        refreshPlanning()
    }
    func startBenchmark() {
        guard ready, let harness else { return }
        benchmarkSamples = []; benchmarkModel = modelName
        benchmarkStep = 0; benchmarkStopping = false; benchmarkRunning = true
        submitBenchmark(harness)
    }
    func stopBenchmark() {
        guard benchmarkRunning else { return }
        benchmarkStopping = true; benchmarkStatus = "Stopping benchmark…"
        harness?.worker?.cancel()
    }
    private func submitBenchmark(_ harness: HarnessStore) {
        do {
            let bytes = try JSONSerialization.data(withJSONObject:[["role":"user","content":Self.benchmarkPrompt]],options:[.sortedKeys])
            let id = UUID()
            invalidatePlanning()
            guard harness.worker?.submit(.chat(id:id,conversation:bytes,maximum:Self.benchmarkMaximum)) == true else {
                throw HubFailure("The engine is busy. Try the benchmark again when it is ready.")
            }
            benchmarkRequest = id
            benchmarkStatus = benchmarkStep == 0 ? "Warming up…" : "Running sample \(benchmarkStep) of 3…"
            harness.poll()
        } catch { finishBenchmark(error.localizedDescription) }
    }
    private func finishBenchmark(_ message: String) {
        benchmarkRequest = nil; benchmarkRunning = false; benchmarkStatus = message
    }
    private func observeBenchmark(_ state: NativeSnapshot) {
        guard benchmarkRunning, !state.busy else { return }
        guard !state.closed, state.chatEnabled, state.hasSession, harness?.expert == false else {
            finishBenchmark("Benchmark stopped because the chat model was released."); return
        }
        guard let result = state.chatResult, result.id == benchmarkRequest else {
            if state.closed || !state.chatEnabled { finishBenchmark("Benchmark stopped because the model was released.") }
            return
        }
        if benchmarkStopping || result.reason == 3 { finishBenchmark("Benchmark stopped. Incomplete runs are excluded."); return }
        guard result.error.isEmpty, !result.rejected, state.status == 0 else {
            finishBenchmark("Benchmark failed: " + (result.error.isEmpty ? state.message : result.error)); return
        }
        let metrics = GenerationMetrics(state)
        guard metrics.tokens > 0, metrics.tokens == Int(state.emitted), metrics.firstTokenSeconds != nil,
              metrics.totalSeconds != nil, (1...2).contains(result.reason) else {
            finishBenchmark("Benchmark returned no valid timed output. Try again with another model."); return
        }
        if benchmarkStep > 0 { benchmarkSamples.append(LocalBenchmarkSample(id:benchmarkStep,metrics:metrics,stopReason:result.reason)) }
        benchmarkStep += 1
        if benchmarkStep == 4 { finishBenchmark("Complete. 3 measured runs; warm-up excluded.") }
        else if let harness { submitBenchmark(harness) }
    }
    private var activeModelKey: String? {
        guard let harness, harness.state.hasModel, harness.state.chatEnabled, !loadedKey.isEmpty else { return nil }
        return loadedKey
    }
    func sendOrStop() {
        guard let harness else { return }
        if benchmarkRunning { stopBenchmark(); return }
        if running { harness.worker?.cancel(); return }
        guard canSend else { return }
        do {
            let previous = workspace
            let pending = try workspace.begin()
            invalidatePlanning()
            guard harness.worker?.submit(.chat(id:pending.0,conversation:pending.1,maximum:workspace.outputLimit)) == true else {
                workspace = previous; notice = "The engine is busy. Your draft is kept; please try again."
                save(); return
            }
            request = pending.0; requestDraftEdit = draftEdit; notice = ""; save(); harness.poll()
        } catch { notice = error.localizedDescription }
    }
    func observe(_ state: NativeSnapshot) {
        observePlanning(state)
        observeBenchmark(state)
        if loadingModel && !state.busy && pendingFolder == nil {
            loadingModel = false
            loadedModelInstance = state.chatEnabled ? state.modelContext?.modelInstance : nil
            if let worker = harness?.worker {
                planningLoad?.complete(worker:ObjectIdentifier(worker),instance:loadedModelInstance,chatEnabled:state.chatEnabled)
            } else { planningLoad = nil }
            if !state.chatEnabled { planningLoad = nil }
            observePlanning(state)
            if !state.chatEnabled { notice = state.message.isEmpty ? "The model could not be opened. Choose a model to try again." : state.message }
        }
        if let result = state.chatResult, result.id == request {
            workspace.finish(result,restoreRejectedDraft:requestDraftEdit == draftEdit)
            if result.rejected { notice = result.error }
            request = nil; requestDraftEdit = nil; save()
        }
        if state.closed, let id = request {
            workspace.finish(NativeChatResult(id:id,text:"",reason:3,error:"Response stopped. Reload a model to continue."))
            request = nil; requestDraftEdit = nil; save()
        }
        if !state.busy && !state.hasModel {
            scopedFolder?.stopAccessingSecurityScopedResource(); scopedFolder = nil; loadedKey = ""; loadedModelInstance = nil; planningLoad = nil
            if modelName != "Toklet" { modelName = "Toklet" }
            if let folder = pendingFolder, harness?.expert == false {
                pendingFolder = nil; beginLoad(folder)
            }
        }
    }
    // Delayed Hub verification is scoped to the chat which requested it.
    func open(_ folder: URL,ifSelected conversation: UUID) {
        guard workspace.selected == conversation else { return }
        open(folder)
    }
    func open(_ folder: URL) {
        guard !storageBlocked, !running, !loadingModel, let harness, !harness.state.busy, !harness.state.closed else { return }
        invalidatePlanning()
        notice = ""
        do {
            guard let actual = harness.worker?.snapshot(), !actual.busy, !actual.closed,
                  actual.hasModel == harness.state.hasModel,
                  actual.modelContext?.modelInstance == harness.state.modelContext?.modelInstance else {
                harness.poll()
                throw HubFailure("The engine state changed. Wait for it to finish and choose the model again.")
            }
            // Validate the selected folder's bounded pinned configuration before
            // changing history or unloading a working model. Full tensor/text
            // admission still runs at the native load boundary.
            let scoped = folder.startAccessingSecurityScopedResource()
            defer { if scoped { folder.stopAccessingSecurityScopedResource() } }
            let profile = try HubProfile.identify(folder)
            if hasChatModel && loadedKey == key(folder) { notice = "This model is already open."; return }
            if workspace.current.modelKey != key(folder), !workspace.current.messages.isEmpty { try workspace.newChat(modelKey:key(folder)) }
            workspace.conversations[workspace.selectedIndex].modelKey = key(folder)
            save()
            if harness.state.hasModel {
                loadingName = profile.name; loadingModel = true; pendingFolder = folder; harness.worker?.interrupt(unload:true); harness.poll()
            } else { beginLoad(folder) }
        } catch { notice = error.localizedDescription }
    }
    private func beginLoad(_ folder: URL) {
        guard let harness, let worker = harness.worker else { return }
        let actual = worker.snapshot()
        guard !actual.busy, !actual.closed, !actual.hasModel else {
            loadingModel = false
            notice = "The engine is busy. Wait for it to finish before opening another model."
            return
        }
        scopedFolder?.stopAccessingSecurityScopedResource(); scopedFolder = nil
        if folder.startAccessingSecurityScopedResource() { scopedFolder = folder }
        let profile: HubProfile
        do {
            profile = try HubProfile.identify(folder)
            try admission().check(profile)
        } catch {
            loadingModel = false
            scopedFolder?.stopAccessingSecurityScopedResource(); scopedFolder = nil
            notice = error.localizedDescription; return
        }
        var settings = NativeSettings(directory:folder.path)
        settings.metal = true; settings.mlx = true; settings.chat = true
        settings.capacity = NativeSettings.chatContextPositions; settings.chunk = 32
        settings.numericalBudget = profile.numericalBudget; settings.textBudget = profile.textBudget
        if worker.submit(.load(settings)) {
            planningLoad = FriendlyPlanningLoad(worker:ObjectIdentifier(worker),settings:settings,coexistenceBudget:profile.processAllowance)
            loadingName = profile.name; loadingModel = true; loadedKey = key(folder); modelName = profile.name
            harness.poll()
        } else {
            loadingModel = false
            scopedFolder?.stopAccessingSecurityScopedResource(); scopedFolder = nil
            notice = "The engine is busy. Your model selection was not loaded."
        }
    }
    func unload() { invalidatePlanning(); pendingFolder = nil; loadingModel = false; harness?.worker?.interrupt(unload:true); harness?.poll() }
    func enterExpert() { invalidatePlanning(); pendingFolder = nil; loadingModel = false; save(); harness?.setExpert(true) }
    // Preview is never serialized into ChatWorkspace or used by beginLoad.
    private func planningTarget(_ state: NativeSnapshot, requireVisible: Bool = true) -> FriendlyPlanningTarget? {
        guard (!requireVisible || planningVisible), let harness, !harness.expert, !loadingModel,
              let worker = harness.worker, let accepted = planningLoad,
              accepted.worker == ObjectIdentifier(worker), state.hasModel, state.chatEnabled, !state.closed,
              let instance = state.modelContext?.modelInstance, instance > 0,
              instance == loadedModelInstance, !loadedKey.isEmpty, loadedKey == workspace.current.modelKey else { return nil }
        return accepted.target(worker:ObjectIdentifier(worker),instance:instance,
            conversation:workspace.selected,loadedKey:loadedKey,selectedKey:workspace.current.modelKey,
            configuration:state.sessionConfiguration)
    }
    private func observePlanning(_ state: NativeSnapshot) {
        // Polling revisions are not authority: reconcile against the actual
        // owning worker again before making a result visible.
        let actual = harness?.worker?.snapshot() ?? state
        let currentWorker = harness?.worker
        let currentIdentity = currentWorker.map { ObjectIdentifier($0) }
        if planningLoad?.worker != currentIdentity { planningLoad = nil }
        let active = planningTarget(actual,requireVisible:false)
        if let prior = planning.target,
           active?.worker != prior.worker || active?.modelInstance != prior.modelInstance
            || active?.conversation != prior.conversation || active?.modelKey != prior.modelKey {
            applying.invalidate()
        }
        // Accepted configuration is durable even when cancellation erased the
        // transient result. Reconcile it before retiring the old preview.
        applying.observe(target:active,snapshot:actual)
        currentSessionConfiguration = active?.configuration
        if planning.select(planningVisible ? active : nil) {
            planningWorker?.selectPreview(nil); planningWorker?.selectTransition(nil)
            planningWorker = harness?.worker
            planningWorker?.selectPreview(nil); planningWorker?.selectTransition(nil)
        }
        planning.observe(actual.preview,busy:actual.busy)
    }
    private func refreshPlanning() {
        if let state = harness?.worker?.snapshot() { observePlanning(state) }
        else { _ = planning.select(nil); applying.invalidate(); currentSessionConfiguration = nil }
    }
    func showPlanning() { planningVisible = true; refreshPlanning() }
    func hidePlanning() { planningVisible = false; invalidatePlanning(); _ = planning.select(nil) }
    func invalidatePlanning() {
        // Serialize invalidation against acceptance, then observe any commit
        // which won the race. Never cancel unrelated generation.
        planningWorker?.selectTransition(nil)
        if planningWorker !== harness?.worker { harness?.worker?.selectTransition(nil) }
        if let actual = harness?.worker?.snapshot() {
            applying.observe(target:planningTarget(actual,requireVisible:false),snapshot:actual)
            currentSessionConfiguration = planningTarget(actual,requireVisible:false)?.configuration
        }
        if !applying.committed { applying.invalidate() }
        planning.invalidate()
        planningWorker?.selectPreview(nil)
        if planningWorker !== harness?.worker { harness?.worker?.selectPreview(nil) }
    }
    func editPlanning(_ field: FriendlyPlanningDraft.Field, _ text: String) {
        invalidatePlanning(); applying.invalidate(); planning.edit(field,text)
    }
    func previewPlanning() {
        guard let harness, let worker = harness.worker else { planning.reject("The engine is unavailable."); return }
        let actual = worker.snapshot()
        observePlanning(actual)
        guard planningVisible, !actual.busy, !actual.closed, !running, !loadingModel,
              planning.target != nil else { planning.reject("Wait for an idle loaded chat model before previewing."); return }
        do {
            let request = try planning.begin()
            guard worker.selectPreview(request.selection), worker.submit(.preview(request)) else {
                worker.selectPreview(nil); planning.reject("The engine changed or is busy. Try Preview again."); return
            }
            harness.poll()
        } catch { worker.selectPreview(nil) }
    }
    var planningApplyUnavailableReason: String? {
        guard let worker = harness?.worker else { return "The engine is unavailable." }
        let actual = worker.snapshot()
        guard planningVisible, !running, !loadingModel, !applying.pending,
              let target = planningTarget(actual) else { return "Open an idle chat session before applying." }
        do {
            _ = try FriendlyApplyState.prepare(planning:planning,current:actual,target:target,reply:workspace.outputLimit)
            return nil
        } catch { return error.localizedDescription }
    }
    func applyPlanning() {
        guard let harness, let worker = harness.worker else { applying.reject("The engine is unavailable."); return }
        let actual = worker.snapshot()
        observePlanning(actual)
        guard planningVisible, !running, !loadingModel, !applying.pending,
              let target = planningTarget(actual) else { applying.reject("Wait for an idle active chat session before applying."); return }
        do {
            let request = try FriendlyApplyState.prepare(planning:planning,current:actual,target:target,reply:workspace.outputLimit)
            applying.begin(request)
            guard worker.selectTransition(request.transition.selection),
                  worker.submit(.replaceWithEmptySession(request.transition)) else {
                worker.selectTransition(nil)
                applying.reject("The engine changed or is busy. Your current chat and edited values are kept.")
                return
            }
            harness.poll()
        } catch { applying.reject(error.localizedDescription) }
    }

    func refreshFolders() {
        let candidates = [documents.appendingPathComponent("M8Inputs/smollm/Model"),documents.appendingPathComponent("Model")]
        folders = candidates.filter { FileManager.default.fileExists(atPath:$0.appendingPathComponent("config.json").path) }
    }
}

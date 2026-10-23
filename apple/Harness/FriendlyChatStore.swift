import Foundation
import Combine

@MainActor
final class FriendlyChatStore: ObservableObject {
    @Published var workspace = ChatWorkspace()
    @Published var notice = ""
    @Published var storageError = ""
    @Published var folders: [URL] = []
    private weak var harness: HarnessStore?
    private var request: UUID?
    private var pendingFolder: URL?
    private var scopedFolder: URL?
    private var loadedKey = ""
    private var saving: Task<Void,Never>?
    private var storageBlocked = false
    private let file: URL

    init(harness: HarnessStore, fileURL: URL? = nil) {
        self.harness = harness
        file = fileURL ?? FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("Toklet/chat-v1.json")
        restore()
        refreshFolders()
    }
    var running: Bool { request != nil }
    var storageAvailable: Bool { !storageBlocked }
    var hasChatModel: Bool {
        guard let harness else { return false }
        return harness.state.hasModel && harness.state.chatEnabled && loadedKey == workspace.current.modelKey
    }
    var ready: Bool {
        guard let harness else { return false }
        return hasChatModel && !harness.state.closed && !harness.state.busy && !running
    }
    var canSend: Bool { ready && !workspace.current.draft.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && !storageBlocked }
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
    func save() {
        saving?.cancel()
        guard !storageBlocked else { return }
        do {
            let bytes = try JSONEncoder().encode(workspace)
            guard bytes.count <= ChatWorkspace.maximumBytes else { throw chatInputError("Chat history is full") }
            try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
            #if os(iOS)
            try bytes.write(to:file,options:[.atomic,.completeFileProtectionUntilFirstUserAuthentication])
            #else
            try bytes.write(to:file,options:.atomic)
            #endif
            storageError = ""
        } catch { storageError = "Could not save chats: " + error.localizedDescription }
    }
    func retryStorage() { if storageBlocked { restore() } else { save() } }
    func edit(_ text: String) {
        guard !storageBlocked else { return }
        guard text.utf8.count <= 16384 else { notice = "A message can contain up to 16 KiB of text."; return }
        workspace.edit(text)
        saving?.cancel()
        saving = Task { [weak self] in
            do { try await Task.sleep(for:.milliseconds(400)) } catch { return }
            self?.save()
        }
    }
    func newChat() {
        guard !storageBlocked, !running, harness?.state.busy == false else { return }
        do { try workspace.newChat(); notice = ""; save() }
        catch { notice = error.localizedDescription }
    }
    func select(_ id: UUID) {
        guard !storageBlocked, !running, harness?.state.busy == false, workspace.conversations.contains(where: { $0.id == id }) else { return }
        workspace.selected = id; notice = ""; save()
    }
    func remove(_ id: UUID) {
        guard !storageBlocked, !running, harness?.state.busy == false else { return }
        workspace.remove(id); save()
    }
    func sendOrStop() {
        guard let harness else { return }
        if running { harness.worker?.cancel(); return }
        guard canSend else { return }
        do {
            let previous = workspace
            let pending = try workspace.begin()
            guard harness.worker?.submit(.chat(id:pending.0,conversation:pending.1,maximum:workspace.outputLimit)) == true else {
                workspace = previous; notice = "The engine is busy. Your draft is kept; please try again."
                save(); return
            }
            request = pending.0; notice = ""; save(); harness.poll()
        } catch { notice = error.localizedDescription }
    }
    func observe(_ state: NativeSnapshot) {
        if let result = state.chatResult, result.id == request {
            workspace.finish(result); request = nil; save()
        }
        if state.closed, let id = request {
            workspace.finish(NativeChatResult(id:id,text:"",reason:3,error:"Response stopped. Reload a model to continue."))
            request = nil; save()
        }
        if !state.busy && !state.hasModel {
            scopedFolder?.stopAccessingSecurityScopedResource(); scopedFolder = nil; loadedKey = ""
            if let folder = pendingFolder, harness?.expert == false {
                pendingFolder = nil; beginLoad(folder)
            }
        }
    }
    func open(_ folder: URL) {
        guard !storageBlocked, !running, let harness, !harness.state.busy, !harness.state.closed else { return }
        notice = ""
        do {
            if workspace.current.modelKey != key(folder), !workspace.current.messages.isEmpty { try workspace.newChat(modelKey:key(folder)) }
            workspace.conversations[workspace.selectedIndex].modelKey = key(folder)
            save()
            if harness.state.hasModel {
                pendingFolder = folder; harness.worker?.interrupt(unload:true); harness.poll()
            } else { beginLoad(folder) }
        } catch { notice = error.localizedDescription }
    }
    private func beginLoad(_ folder: URL) {
        guard let harness, let worker = harness.worker else { return }
        let actual = worker.snapshot()
        guard !actual.busy, !actual.closed, !actual.hasModel else {
            notice = "The engine is busy. Wait for it to finish before opening another model."
            return
        }
        scopedFolder?.stopAccessingSecurityScopedResource(); scopedFolder = nil
        if folder.startAccessingSecurityScopedResource() { scopedFolder = folder }
        var settings = NativeSettings(directory:folder.path)
        settings.metal = true; settings.mlx = true; settings.chat = true
        settings.capacity = 512; settings.chunk = 32; settings.numericalBudget = 512 * 1024 * 1024
        if worker.submit(.load(settings)) {
            loadedKey = key(folder)
            harness.poll()
        } else {
            scopedFolder?.stopAccessingSecurityScopedResource(); scopedFolder = nil
            notice = "The engine is busy. Your model selection was not loaded."
        }
    }
    func unload() { pendingFolder = nil; harness?.worker?.interrupt(unload:true); harness?.poll() }
    func enterExpert() { pendingFolder = nil; save(); harness?.setExpert(true) }
    func refreshFolders() {
        let candidates = [documents.appendingPathComponent("M8Inputs/smollm/Model"),documents.appendingPathComponent("Model")]
        folders = candidates.filter { FileManager.default.fileExists(atPath:$0.appendingPathComponent("config.json").path) }
    }
}

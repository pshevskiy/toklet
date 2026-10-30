import Foundation
import CryptoKit
import CToklet

// User choices only. The caller supplies an actual accepted model/codec
// namespace; a file, path or record never authenticates a loaded model.
struct FriendlyModelPreference: Equatable {
    static let maximumBytes = 4096
    let namespace: NativeAcceptedIdentity.Namespace
    let capacity, chunk, reply: UInt32
    let wholeBudget: UInt64

    static func checkNamespace(_ value: NativeAcceptedIdentity.Namespace) throws {
        guard value.version == 1, value.manifestAlgorithm == "toklet-full-manifest-v1",
              value.manifestSHA256 == NativeAcceptedIdentity.manifestSHA256,
              value.backend == "metal_native_dense_v1", value.load == "qwen3_audited_v1",
              value.text == "qwen3-plain-nonthinking-v1", value.planning == "native_mlx_planning_v1" else {
            throw NativeContextFailure("Saved settings use an unsupported model policy.")
        }
    }
    func validate() throws {
        try Self.checkNamespace(namespace)
        guard (1...32768).contains(capacity), (1...1024).contains(chunk), chunk <= capacity,
              wholeBudget > UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES), wholeBudget <= 6 * 1024 * 1024 * 1024,
              ChatWorkspace.outputLimits.contains(reply), reply < capacity else {
            throw NativeContextFailure("Saved settings are outside the supported range. Your record is kept.")
        }
    }
    static func validateReply(_ reply: UInt32, capacity: UInt64, model: NativeModelContext) throws {
        guard capacity > 0, ChatWorkspace.outputLimits.contains(reply),
              UInt64(reply) <= min(capacity - 1,model.declaredOutput ?? .max,UInt64(NativeGenerationPolicy.maximumChatOutput)) else {
            throw NativeContextFailure("That reply limit does not fit the current window. Increase context or choose a shorter reply; your previous choice is kept.")
        }
        // This leaves room for a nonempty prompt, not a tokenizer template.
        // The rendered conversation plus reply is admitted separately on Send.
    }
    func validate(model: NativeModelContext) throws {
        try validate()
        guard UInt64(capacity) <= model.loadedLimit else {
            throw NativeContextFailure("Saved context exceeds this opened model's limit. Current settings and the saved record are kept.")
        }
        try Self.validateReply(reply,capacity:UInt64(capacity),model:model)
    }
    static func filename(_ namespace: NativeAcceptedIdentity.Namespace) throws -> String {
        try checkNamespace(namespace)
        var bytes = Data("toklet-preference-key-v1\0".utf8)
        func integer(_ value: UInt64) { var value = value.littleEndian; withUnsafeBytes(of:&value) { bytes.append(contentsOf:$0) } }
        integer(1); integer(namespace.version)
        for text in [namespace.manifestAlgorithm,namespace.manifestSHA256,namespace.backend,namespace.load,namespace.text,namespace.planning] {
            integer(UInt64(text.utf8.count)); bytes.append(contentsOf:text.utf8)
        }
        return SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined() + ".json"
    }
    func encoded() throws -> Data {
        try validate()
        let object: [String:Any] = ["schema":1,"namespace":["version":namespace.version,
            "manifest_algorithm":namespace.manifestAlgorithm,"manifest_sha256":namespace.manifestSHA256,
            "backend":namespace.backend,"load":namespace.load,"text":namespace.text,"planning":namespace.planning],
            "capacity":capacity,"chunk":chunk,"whole_budget":wholeBudget,"reply":reply]
        let data = try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys])
        guard data.count <= Self.maximumBytes else { throw NativeContextFailure("Saved settings are too large.") }
        return data
    }
    static func decode(_ data: Data, namespace expected: NativeAcceptedIdentity.Namespace) throws -> Self {
        try checkNamespace(expected)
        let object = try GenerationJSON.object(data,maximumBytes:maximumBytes,maximumStringBytes:128)
        try GenerationJSON.keys(object,["schema","namespace","capacity","chunk","whole_budget","reply"])
        guard try GenerationJSON.integer(object["schema"]) == 1 else { throw NativeContextFailure("Unknown settings version.") }
        let n = try GenerationJSON.dictionary(object["namespace"])
        try GenerationJSON.keys(n,["version","manifest_algorithm","manifest_sha256","backend","load","text","planning"])
        let namespace = try NativeAcceptedIdentity.Namespace(version:GenerationJSON.integer(n["version"]),
            manifestAlgorithm:GenerationJSON.string(n["manifest_algorithm"],maximumBytes:64),
            manifestSHA256:GenerationJSON.hex(n["manifest_sha256"],count:64),
            backend:GenerationJSON.string(n["backend"],maximumBytes:64),load:GenerationJSON.string(n["load"],maximumBytes:64),
            text:GenerationJSON.string(n["text"],maximumBytes:64),planning:GenerationJSON.string(n["planning"],maximumBytes:64))
        guard namespace == expected,
              let capacity = UInt32(exactly:try GenerationJSON.positive(object["capacity"])),
              let chunk = UInt32(exactly:try GenerationJSON.positive(object["chunk"])),
              let reply = UInt32(exactly:try GenerationJSON.positive(object["reply"])) else {
            throw NativeContextFailure("Saved settings do not match this accepted model.")
        }
        let value = try Self(namespace:namespace,capacity:capacity,chunk:chunk,reply:reply,
                             wholeBudget:GenerationJSON.positive(object["whole_budget"]))
        try value.validate(); return value
    }
}

// One bounded file per engine-allowlisted namespace, no directory scan or paths
// from records. MainActor serializes this app's writers. Atomic replacement is
// not a claim of power-loss durability or hostile concurrent-filesystem safety.
@MainActor
final class FriendlyPreferenceStore {
    let directory: URL
    private let persist: (Data,URL) throws -> Void
    init(directory: URL, persist: ((Data,URL) throws -> Void)? = nil) {
        self.directory = directory
        self.persist = persist ?? { bytes,url in
            #if os(iOS)
            try bytes.write(to:url,options:[.atomic,.completeFileProtectionUntilFirstUserAuthentication])
            #else
            try bytes.write(to:url,options:.atomic)
            #endif
        }
    }
    func url(_ namespace: NativeAcceptedIdentity.Namespace) throws -> URL {
        directory.appendingPathComponent(try FriendlyModelPreference.filename(namespace))
    }
    private func read(_ path: URL) throws -> Data? {
        guard FileManager.default.fileExists(atPath:path.path) else { return nil }
        return try ModelHub.withRegular(path,maximum:Int64(FriendlyModelPreference.maximumBytes)) { handle,size in
            let bytes = try handle.read(upToCount:FriendlyModelPreference.maximumBytes + 1) ?? Data()
            guard bytes.count == size, bytes.count <= FriendlyModelPreference.maximumBytes else {
                throw NativeContextFailure("Saved settings changed or exceed their size limit.")
            }
            return bytes
        }
    }
    func load(_ namespace: NativeAcceptedIdentity.Namespace) throws -> FriendlyModelPreference? {
        guard let bytes = try read(url(namespace)) else { return nil }
        return try FriendlyModelPreference.decode(bytes,namespace:namespace)
    }
    func saveAccepted(_ value: FriendlyModelPreference) -> String {
        do { try save(value); return "Current chat settings saved for this managed model." }
        catch { return "Current settings could not be saved: " + error.localizedDescription }
    }
    func save(_ value: FriendlyModelPreference) throws {
        let bytes = try value.encoded(), path = try url(value.namespace)
        // Do not silently replace a damaged/unknown record with defaults.
        if let previous = try read(path) {
            let accepted = try FriendlyModelPreference.decode(previous,namespace:value.namespace)
            if accepted == value { return }
        }
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        try persist(bytes,path)
    }
}

// This owner correlation is never stored on disk. One intent covers either an
// explicit Apply save or an automatic restore; it is retired on owner changes.
struct FriendlyPreferenceIntent {
    let worker: ObjectIdentifier
    let conversation, loadRequest, editGeneration: UUID
    let identity: NativeAcceptedIdentity
    enum Acceptance {
        case replacement(NativeTransitionRequest)
        case current(NativeSessionConfiguration)
    }
    let acceptance: Acceptance
    let value: FriendlyModelPreference
    let restoring: Bool

    func owns(worker: ObjectIdentifier, conversation: UUID, snapshot: NativeSnapshot) -> Bool {
        self.worker == worker && self.conversation == conversation
            && snapshot.acceptedLoadRequestID == loadRequest && snapshot.acceptedIdentity == identity
            && snapshot.modelContext?.modelInstance == identity.modelInstance && snapshot.hasModel && snapshot.chatEnabled
            && !snapshot.closed
    }
    enum Resolution: Equatable {
        case waiting, discarded, rejected, quarantine, newerEdit
        case restoreReply(NativeModelContext), save(NativeModelContext)
    }
    // Only terminal healthy facts can produce a persistence/reply effect.
    // This resolver does not mutate the model, workspace, record or draft.
    func resolve(worker: ObjectIdentifier?, conversation: UUID, expert: Bool,
                 editGeneration: UUID, snapshot: NativeSnapshot) -> Resolution {
        guard !expert, let worker, owns(worker:worker,conversation:conversation,snapshot:snapshot) else { return .discarded }
        guard !snapshot.busy else { return .waiting }
        guard committed(snapshot) else { return .rejected }
        guard !snapshot.restartRequired, let context = snapshot.modelContext,
              context.configuredCapacity == UInt64(value.capacity) else { return .quarantine }
        if restoring { return self.editGeneration == editGeneration ? .restoreReply(context) : .newerEdit }
        return .save(context)
    }
    // Use the same captured snapshot as terminal reconciliation. Reading again
    // can observe cleanup completing and accidentally discard an unsaved intent.
    func retainDuringCleanup(_ snapshot: NativeSnapshot) -> Bool { snapshot.busy && committed(snapshot) }
    func committed(_ snapshot: NativeSnapshot) -> Bool {
        guard let current = snapshot.sessionConfiguration else { return false }
        switch acceptance {
        case .current(let accepted): return current == accepted
        case .replacement(let request):
            return current.modelInstance == request.selection.modelInstance
                && current.sessionInstance != request.selection.sessionInstance && current.acceptedRequestID == request.id
                && current.capacity == value.capacity && current.chunk == value.chunk && current.wholeSessionBudget == value.wholeBudget
        }
    }
}

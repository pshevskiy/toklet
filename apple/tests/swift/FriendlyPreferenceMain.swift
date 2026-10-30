import Foundation
import CToklet

@main
@MainActor
struct FriendlyPreferenceMain {
    static var names: [String] = []
    static func check(_ name: String, _ value: Bool) throws {
        names.append(name)
        if !value { throw NativeContextFailure("Preference control: " + name) }
    }
    static func rejects(_ name: String, _ body: () throws -> Void) throws {
        var rejected = false
        do { try body() } catch { rejected = true }
        try check(name,rejected)
    }
    static func namespace() -> NativeAcceptedIdentity.Namespace {
        .init(version:1,manifestAlgorithm:"toklet-full-manifest-v1",manifestSHA256:NativeAcceptedIdentity.manifestSHA256,
              backend:"metal_native_dense_v1",load:"qwen3_audited_v1",text:"qwen3-plain-nonthinking-v1",planning:"native_mlx_planning_v1")
    }
    static func facts(capacity: UInt64 = 512, session: UInt64 = 2) throws -> NativeContextFacts {
        var raw = TokletContextInfo()
        raw.abi_version = TOKLET_ABI_VERSION; raw.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        raw.model_instance = 1; raw.session_instance = session; raw.backend = TOKLET_CONTEXT_METAL
        raw.engine_global_position_ceiling = 32768; raw.loaded_position_limit = 512; raw.session_capacity = capacity
        raw.numerical_and_bridge_admitted_bytes = 80*1024*1024
        return try NativeContextFacts(raw)
    }
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw NativeContextFailure("Expected exclusive temporary directory") }
        let directory = URL(fileURLWithPath:CommandLine.arguments[1])
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        let n = namespace(), model = NativeModelContext(try facts()), budget: UInt64 = 1536*1024*1024
        let value = FriendlyModelPreference(namespace:n,capacity:256,chunk:16,reply:128,wholeBudget:budget)
        let bytes = try value.encoded()
        try check("record-roundtrip",try FriendlyModelPreference.decode(bytes,namespace:n) == value)
        try check("budget-includes-bridge-once",try FriendlyModelPreference.decode(bytes,namespace:n).wholeBudget == budget)
        try check("key-stable",try FriendlyModelPreference.filename(n) == FriendlyModelPreference.filename(namespace()))
        try check("filename-bounded",try FriendlyModelPreference.filename(n).utf8.count == 69)
        try check("qwen-key-golden-unchanged",try FriendlyModelPreference.filename(n) == "76000af6228a780641701064c74675be166531ffa2fb6221f132027df7d6bc73.json")
        let smolNamespace = NativeAcceptedIdentity.Profile.smol.namespace
        let smolValue = FriendlyModelPreference(namespace:smolNamespace,capacity:256,chunk:16,reply:64,wholeBudget:512*1024*1024)
        let smolBytes = try smolValue.encoded()
        let qwen2Namespace = NativeAcceptedIdentity.Profile.qwen2.namespace
        let qwen2Value = FriendlyModelPreference(namespace:qwen2Namespace,capacity:256,chunk:16,reply:64,wholeBudget:1024*1024*1024)
        let qwen2Bytes = try qwen2Value.encoded()
        try check("qwen2-record-roundtrip",try FriendlyModelPreference.decode(qwen2Bytes,namespace:qwen2Namespace) == qwen2Value)
        let threeKeys = try [n,smolNamespace,qwen2Namespace].map { try FriendlyModelPreference.filename($0) }
        try check("three-profiles-distinct-keys",Set(threeKeys).count == 3)
        try rejects("qwen2-record-cannot-restore-as-qwen3") { _ = try FriendlyModelPreference.decode(qwen2Bytes,namespace:n) }
        try rejects("qwen2-record-cannot-restore-as-smol") { _ = try FriendlyModelPreference.decode(qwen2Bytes,namespace:smolNamespace) }
        try rejects("qwen3-record-cannot-restore-as-qwen2") { _ = try FriendlyModelPreference.decode(bytes,namespace:qwen2Namespace) }
        try rejects("smol-record-cannot-restore-as-qwen2") { _ = try FriendlyModelPreference.decode(smolBytes,namespace:qwen2Namespace) }

        try check("smol-record-roundtrip",try FriendlyModelPreference.decode(smolBytes,namespace:smolNamespace) == smolValue)
        try check("profiles-have-distinct-keys",try FriendlyModelPreference.filename(smolNamespace) != FriendlyModelPreference.filename(n))
        try rejects("qwen-record-cannot-restore-as-smol") { _ = try FriendlyModelPreference.decode(bytes,namespace:smolNamespace) }
        try rejects("smol-record-cannot-restore-as-qwen") { _ = try FriendlyModelPreference.decode(smolBytes,namespace:n) }
        let mixed = NativeAcceptedIdentity.Namespace(version:1,manifestAlgorithm:n.manifestAlgorithm,
            manifestSHA256:smolNamespace.manifestSHA256,backend:n.backend,load:n.load,text:smolNamespace.text,planning:n.planning)
        try rejects("mixed-profile-key-rejected") { _ = try FriendlyModelPreference.filename(mixed) }
        let object = try JSONSerialization.jsonObject(with:bytes) as! [String:Any]
        for (key,bad) in [("schema",2),("capacity",0),("capacity",32769),("chunk",0),("chunk",257),
                          ("chunk",1025),("reply",256),("reply",31),("whole_budget",TOKLET_MODEL_SESSION_BRIDGE_BYTES),
                          ("whole_budget",UInt64(6)*1024*1024*1024+1),("reply",true)] as [(String,Any)] {
            var wrong = object; wrong[key] = bad
            try rejects("invalid-\(key)-\(bad)") { _ = try FriendlyModelPreference.decode(JSONSerialization.data(withJSONObject:wrong),namespace:n) }
        }
        var unknown = object; unknown["future"] = 1
        try rejects("unknown-key") { _ = try FriendlyModelPreference.decode(JSONSerialization.data(withJSONObject:unknown),namespace:n) }
        var wrongNamespace = object, ns = object["namespace"] as! [String:Any]; ns["planning"] = "future"; wrongNamespace["namespace"] = ns
        try rejects("policy-mismatch") { _ = try FriendlyModelPreference.decode(JSONSerialization.data(withJSONObject:wrongNamespace),namespace:n) }
        let text = String(decoding:bytes,as:UTF8.self)
        for (label,replacement) in [("duplicate","\"reply\":128,\"reply\":128"),("float","\"reply\":128.0"),
                                   ("overflow","\"reply\":18446744073709551616")] {
            try rejects(label) { _ = try FriendlyModelPreference.decode(Data(text.replacingOccurrences(of:"\"reply\":128",with:replacement).utf8),namespace:n) }
        }
        try rejects("oversized") { _ = try FriendlyModelPreference.decode(Data(repeating:32,count:4097),namespace:n) }
        try rejects("reply32-capacity32") { try FriendlyModelPreference.validateReply(32,capacity:32,model:model) }
        try FriendlyModelPreference.validateReply(32,capacity:33,model:model)
        try check("reply32-capacity33-is-mode-only",NativeGenerationPolicy.limits(.chat,model:NativeModelContext(try facts(capacity:33))).maximumOutput == 32)
        try rejects("loaded-bound") { try FriendlyModelPreference(namespace:n,capacity:1024,chunk:16,reply:32,wholeBudget:budget).validate(model:model) }
        var writes = 0
        let store = FriendlyPreferenceStore(directory:directory,persist:{ data,url in writes += 1; try data.write(to:url,options:.atomic) })
        try check("absent",try store.load(n) == nil)
        try store.save(value)
        let path = try store.url(n), original = try Data(contentsOf:path)
        try check("one-write",writes == 1 && (try store.load(n)) == value)
        try store.save(value); try check("identical-no-write",writes == 1)
        let reopened = FriendlyPreferenceStore(directory:directory)
        try check("reopened-record",try reopened.load(n) == value)
        // A separate injected store writes Smol without changing this test's
        // original Qwen write counter or bytes. Both use the same file protocol.
        try reopened.save(smolValue)
        try check("shared-store-keeps-qwen",try Data(contentsOf:path) == original && reopened.load(n) == value)
        try check("shared-store-loads-smol",try reopened.load(smolNamespace) == smolValue)
        let preservedSmol = try Data(contentsOf:reopened.url(smolNamespace))
        try reopened.save(qwen2Value)
        try check("shared-store-loads-qwen2",try reopened.load(qwen2Namespace) == qwen2Value)
        try check("qwen2-save-keeps-qwen3",try Data(contentsOf:path) == original)
        try check("qwen2-save-keeps-smol",try Data(contentsOf:reopened.url(smolNamespace)) == preservedSmol)

        let failure = FriendlyPreferenceStore(directory:directory,persist:{ _,_ in throw NativeContextFailure("Injected atomic write failure") })
        let other = FriendlyModelPreference(namespace:n,capacity:256,chunk:8,reply:64,wholeBudget:budget)
        try rejects("write-failure") { try failure.save(other) }
        try check("write-failure-keeps-old",try Data(contentsOf:path) == original)
        for (label,bad) in [("corrupt",Data("{bad".utf8)),("oversized",Data(repeating:32,count:4097))] {
            try bad.write(to:path,options:.atomic)
            try rejects("read-\(label)") { _ = try store.load(n) }
            try rejects("save-keeps-\(label)") { try store.save(other) }
            try check("bytes-kept-\(label)",try Data(contentsOf:path) == bad && writes == 1)
        }
        try original.write(to:path,options:.atomic)
        let worker = NSObject(), wrongWorker = NSObject(), conversation = UUID(), loadID = UUID(), edit = UUID()
        let identity = NativeAcceptedIdentity(namespace:n,assets:[],modelInstance:1,loadedLimit:512,sourceRanges:3,sourceReads:3)
        let selection = NativeTransitionSelection(id:UUID(),modelInstance:1,sessionInstance:2)
        let request = NativeTransitionRequest(id:UUID(),selection:selection,capacity:256,chunk:16,wholeSessionBudget:budget,
            coexistenceBudget:2304*1024*1024,replayReserve:0,textReserve:256*1024*1024,applicationReserve:NativeTransitionHeadroomCheck.applicationReserve)
        let intent = FriendlyPreferenceIntent(worker:ObjectIdentifier(worker),conversation:conversation,loadRequest:loadID,
            editGeneration:edit,identity:identity,acceptance:.replacement(request),value:value,restoring:true)
        var snapshot = NativeSnapshot(); snapshot.hasModel = true; snapshot.hasSession = true; snapshot.chatEnabled = true
        snapshot.acceptedIdentity = identity; snapshot.acceptedLoadRequestID = loadID; snapshot.modelContext = model
        let newFacts = try facts(capacity:256,session:3)
        snapshot.sessionConfiguration = try NativeSessionConfiguration(context:newFacts,capacity:256,chunk:16,wholeSessionBudget:budget,acceptedRequestID:request.id)
        try check("exact-owner",intent.owns(worker:ObjectIdentifier(worker),conversation:conversation,snapshot:snapshot))
        try check("wrong-worker",!intent.owns(worker:ObjectIdentifier(wrongWorker),conversation:conversation,snapshot:snapshot))
        try check("wrong-conversation",!intent.owns(worker:ObjectIdentifier(worker),conversation:UUID(),snapshot:snapshot))
        try check("durable-commit",intent.committed(snapshot))
        snapshot.transition = nil; snapshot.busy = true
        try check("commit-visible-before-terminal",intent.committed(snapshot) && snapshot.busy)
        let capturedBusy = snapshot
        var terminal = snapshot; terminal.busy = false
        try check("cleanup-between-observations-retains-captured-intent",intent.retainDuringCleanup(capturedBusy) && !intent.retainDuringCleanup(terminal))
        snapshot.acceptedLoadRequestID = UUID()
        try check("wrong-load",!intent.owns(worker:ObjectIdentifier(worker),conversation:conversation,snapshot:snapshot))
        snapshot.acceptedLoadRequestID = loadID; snapshot.closed = true
        try check("closed-owner",!intent.owns(worker:ObjectIdentifier(worker),conversation:conversation,snapshot:snapshot))
        snapshot.closed = false
        snapshot.sessionConfiguration = try NativeSessionConfiguration(context:newFacts,capacity:256,chunk:16,wholeSessionBudget:budget,acceptedRequestID:UUID())
        try check("wrong-accepted-request",!intent.committed(snapshot))
        try check("synthetic-intent-does-not-write",try Data(contentsOf:path) == original && writes == 1)
        terminal.modelContext = NativeModelContext(newFacts)
        func outcome(_ state: NativeSnapshot, _ generation: UUID) -> FriendlyPreferenceIntent.Resolution {
            intent.resolve(worker:ObjectIdentifier(worker),conversation:conversation,expert:false,editGeneration:generation,snapshot:state)
        }
        try check("restore-waits-for-retirement",outcome(capturedBusy,edit) == .waiting)
        try check("restore-terminal-reply",outcome(terminal,edit) == .restoreReply(NativeModelContext(newFacts)))
        try check("restore-newer-global-edit",outcome(terminal,UUID()) == .newerEdit)
        try check("restore-expert-discard",intent.resolve(worker:ObjectIdentifier(worker),conversation:conversation,expert:true,editGeneration:edit,snapshot:terminal) == .discarded)
        try check("restore-stale-worker-discard",intent.resolve(worker:ObjectIdentifier(wrongWorker),conversation:conversation,expert:false,editGeneration:edit,snapshot:terminal) == .discarded)
        var quarantine = terminal; quarantine.restartRequired = true
        try check("restore-quarantine-no-effect",outcome(quarantine,edit) == .quarantine)
        var rejected = terminal; rejected.modelContext = model
        rejected.sessionConfiguration = try NativeSessionConfiguration(context:facts(),capacity:512,chunk:32,wholeSessionBudget:budget)
        let unchangedDefault = String(reflecting:rejected)
        try check("restore-rejected-terminal",outcome(rejected,edit) == .rejected)
        try check("restore-rejected-keeps-default-file",String(reflecting:rejected) == unchangedDefault && (try Data(contentsOf:path)) == original && writes == 1)
        let explicit = FriendlyPreferenceIntent(worker:ObjectIdentifier(worker),conversation:conversation,loadRequest:loadID,
            editGeneration:edit,identity:identity,acceptance:.replacement(request),value:value,restoring:false)
        try check("explicit-save-waits",explicit.resolve(worker:ObjectIdentifier(worker),conversation:conversation,expert:false,editGeneration:edit,snapshot:capturedBusy) == .waiting)
        try check("explicit-save-survives-later-draft",explicit.resolve(worker:ObjectIdentifier(worker),conversation:conversation,expert:false,editGeneration:UUID(),snapshot:terminal) == .save(NativeModelContext(newFacts)))
        let committedBefore = String(reflecting:terminal)
        let failedStatus = failure.saveAccepted(other)
        try check("postcommit-storage-failure-keeps-owner-file",failedStatus.hasPrefix("Current settings could not be saved:") && String(reflecting:terminal) == committedBefore && (try Data(contentsOf:path)) == original)
        var mismatch = terminal; mismatch.modelContext = model
        try check("terminal-context-mismatch-no-save",outcome(mismatch,edit) == .quarantine)
        let harness = HarnessStore(enabled:false), workspaceFile = directory.appendingPathComponent("ordinary-chat.json")
        let chat = FriendlyChatStore(harness:harness,fileURL:workspaceFile,preferences:store)
        chat.setReplyLimit(64)
        let workspaceBytes = try Data(contentsOf:workspaceFile)
        try check("ordinary-no-identity-no-record-save",chat.workspace.outputLimit == 64 && chat.acceptedIdentity == nil && writes == 1 && (try Data(contentsOf:path)) == original)
        chat.setReplyLimit(31)
        try check("rejected-reply-keeps-workspace-record",chat.workspace.outputLimit == 64 && (try Data(contentsOf:workspaceFile)) == workspaceBytes && writes == 1 && (try Data(contentsOf:path)) == original)
        guard names.count == 77 else { throw NativeContextFailure("Preference control inventory changed") }
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:["passed":true,"controls":names,
            "count":names.count,"model_loaded":false,"rendered_admission_tested":false],options:[.sortedKeys]) + Data([10]))
    }
}

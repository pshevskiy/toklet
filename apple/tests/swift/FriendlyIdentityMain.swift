import Foundation

private final class IdentityDequeueGate: @unchecked Sendable {
    private let entered = DispatchSemaphore(value:0), released = DispatchSemaphore(value:0)
    private let lock = NSLock()
    private var captured: NativeSettings?
    private var expired = false
    func pause(_ command: NativeCommand) {
        guard case .load(let settings) = command else { return }
        lock.lock(); captured = settings; lock.unlock(); entered.signal()
        if released.wait(timeout:.now()+30) != .success { lock.lock(); expired = true; lock.unlock() }
    }
    func wait() -> Bool { entered.wait(timeout:.now()+30) == .success }
    func resume() { released.signal() }
    var settings: NativeSettings? { lock.lock(); defer { lock.unlock() }; return captured }
    var timedOut: Bool { lock.lock(); defer { lock.unlock() }; return expired }
}

@main
@MainActor
struct FriendlyIdentityMain {
    static func main() throws {
        var checks = 0
        func require(_ ok: Bool, _ message: String) throws {
            checks += 1
            if !ok { throw NativeContextFailure(message) }
        }
        let folder = URL(fileURLWithPath:"/never-opened/Qwen3-0.6B-8bit")
        try require(NativeSettings(directory:folder.path).loadIdentity == .ordinary,"Path recognition selected audited routing")
        try require(try FriendlyLoadRequest.managedMode(.qwen3) == .managedQwen3,"Explicit managed Qwen routing missing")
        try require(try FriendlyLoadRequest.managedMode(.smol) == .managedSmol,"Explicit managed Smol routing missing")
        let unknown = HubProfile(repository:"unsupported/model",revision:String(repeating:"a",count:40),files:[])
        var unknownRejected = false
        do { _ = try FriendlyLoadRequest.managedMode(unknown) } catch { unknownRejected = true }
        try require(unknownRejected,"Unknown managed profile was promoted")
        let modified = HubProfile(repository:HubProfile.qwen3.repository,revision:HubProfile.qwen3.revision,files:[])
        var rejected = false
        do { _ = try FriendlyLoadRequest.managedMode(modified) } catch { rejected = true }
        try require(rejected,"Partial catalog recognition was treated as managed authority")
        var workspace = ChatWorkspace()
        let request = FriendlyLoadRequest(folder:folder,mode:.managedQwen3,conversation:workspace.selected)
        try require(request.isCurrent(in:workspace),"Fresh request correlation failed")
        try workspace.newChat(modelKey:"another")
        try require(!request.isCurrent(in:workspace),"Changed selected conversation retained delayed request")
        workspace.selected = request.conversation
        try require(request.isCurrent(in:workspace),"Original conversation correlation failed")
        workspace.conversations.removeAll { $0.id == request.conversation }
        try require(!request.isCurrent(in:workspace),"Removed conversation retained delayed request")
        let smolRequest = FriendlyLoadRequest(folder:folder,mode:.managedSmol,conversation:request.conversation)
        try require(FriendlyLoadRequest.matches(.managedSmol,profile:.smol),"Exact Smol route rejected")
        try require(!FriendlyLoadRequest.matches(.managedSmol,profile:.qwen3),"Smol route accepted Qwen metadata")
        try require(!FriendlyLoadRequest.matches(.managedQwen3,profile:.smol),"Qwen route accepted Smol metadata")
        try require(FriendlyLoadRequest.matches(.ordinary,profile:.smol),"External ordinary route changed")
        let fakeSmol = HubProfile(repository:HubProfile.smol.repository,revision:HubProfile.smol.revision,files:[])
        try require(!FriendlyLoadRequest.matches(.managedSmol,profile:fakeSmol),"Partial Smol catalog recognized")
        let first = NSObject(), second = NSObject()
        let accepting = FriendlyAcceptingLoad(request:request,worker:ObjectIdentifier(first))
        var originalWorkspace = ChatWorkspace()
        originalWorkspace.conversations[0].id = request.conversation; originalWorkspace.selected = request.conversation
        try require(accepting.isCurrent(worker:ObjectIdentifier(first),workspace:originalWorkspace),"Pending route lost original worker")
        try require(!accepting.isCurrent(worker:ObjectIdentifier(second),workspace:originalWorkspace),"Pending unload redirected route to replacement worker")
        var snapshot = NativeSnapshot(); snapshot.acceptedLoadRequestID = request.id
        try require(accepting.matches(worker:ObjectIdentifier(first),snapshot:snapshot),"Exact accepting worker/request rejected")
        try require(!accepting.matches(worker:ObjectIdentifier(second),snapshot:snapshot),"Replacement worker accepted stale load")
        snapshot.acceptedLoadRequestID = UUID()
        try require(!accepting.matches(worker:ObjectIdentifier(first),snapshot:snapshot),"Same worker's later request accepted stale completion")
        snapshot.acceptedLoadRequestID = nil
        try require(!accepting.matches(worker:ObjectIdentifier(first),snapshot:snapshot),"Missing accepted request promoted")
        let smolAccepting = FriendlyAcceptingLoad(request:smolRequest,worker:ObjectIdentifier(first))
        snapshot.acceptedLoadRequestID = smolRequest.id
        try require(smolAccepting.matches(worker:ObjectIdentifier(first),snapshot:snapshot),"Smol completion lost captured request")
        try require(!smolAccepting.matches(worker:ObjectIdentifier(second),snapshot:snapshot),"Smol completion redirected to another worker")
        snapshot.acceptedLoadRequestID = request.id
        try require(!smolAccepting.matches(worker:ObjectIdentifier(first),snapshot:snapshot),"Other profile request accepted as Smol")
        var smolSettings = NativeSettings(directory:folder.path); smolSettings.loadIdentity = .managedSmol
        var smolLoad = FriendlyPlanningLoad(worker:ObjectIdentifier(first),settings:smolSettings)
        smolLoad.complete(worker:ObjectIdentifier(first),instance:7,chatEnabled:true)
        try require(smolLoad.target(worker:ObjectIdentifier(first),instance:7,conversation:request.conversation,
            loadedKey:"same",selectedKey:"same")?.profile == .smolAudited,"Smol preview promoted to ordinary numerical authority")
        try require(try FriendlyLoadRequest.managedMode(.qwen) == .managedQwen2,"Explicit managed Qwen2 routing missing")
        try require(FriendlyLoadRequest.matches(.managedQwen2,profile:.qwen),"Exact Qwen2 route rejected")
        try require(!FriendlyLoadRequest.matches(.managedQwen2,profile:.qwen3),"Qwen3 metadata accepted for Qwen2")
        try require(!FriendlyLoadRequest.matches(.managedQwen3,profile:.qwen),"Qwen2 metadata accepted for Qwen3")
        try require(!FriendlyLoadRequest.matches(.managedSmol,profile:.qwen),"Qwen2 metadata accepted for Smol")
        let fakeQwen2 = HubProfile(repository:HubProfile.qwen.repository,revision:HubProfile.qwen.revision,files:[])
        try require(!FriendlyLoadRequest.matches(.managedQwen2,profile:fakeQwen2),"Partial Qwen2 catalog recognized")
        var qwen2Settings = NativeSettings(directory:folder.path); qwen2Settings.loadIdentity = .managedQwen2
        var qwen2Load = FriendlyPlanningLoad(worker:ObjectIdentifier(first),settings:qwen2Settings)
        qwen2Load.complete(worker:ObjectIdentifier(first),instance:7,chatEnabled:true)
        try require(qwen2Load.target(worker:ObjectIdentifier(first),instance:7,conversation:request.conversation,
            loadedKey:"same",selectedKey:"same")?.profile == .ordinary,"Qwen2 ordinary numerical estimator route changed")
        try require(try FriendlyLoadRequest.managedMode(.mistral) == .managedMistral,"Explicit managed Mistral routing missing")
        try require(FriendlyLoadRequest.matches(.managedMistral,profile:.mistral),"Exact Mistral route rejected")
        for mode in [NativeLoadIdentity.managedQwen2,.managedQwen3,.managedSmol] {
            try require(!FriendlyLoadRequest.matches(mode,profile:.mistral),"Mistral accepted for other managed mode")
        }
        for other in [HubProfile.qwen,.qwen3,.smol] {
            try require(!FriendlyLoadRequest.matches(.managedMistral,profile:other),"Other profile accepted as Mistral")
        }
        let fakeMistral = HubProfile(repository:HubProfile.mistral.repository,revision:HubProfile.mistral.revision,files:[])
        try require(!FriendlyLoadRequest.matches(.managedMistral,profile:fakeMistral),"Partial Mistral catalog recognized")
        var mistralSettings = NativeSettings(directory:folder.path); mistralSettings.loadIdentity = .managedMistral
        var mistralLoad = FriendlyPlanningLoad(worker:ObjectIdentifier(first),settings:mistralSettings)
        mistralLoad.complete(worker:ObjectIdentifier(first),instance:7,chatEnabled:true)
        try require(mistralLoad.target(worker:ObjectIdentifier(first),instance:7,conversation:request.conversation,
            loadedKey:"same",selectedKey:"same")?.profile == .ordinary,"Mistral advisory estimator route changed")
        // Existing queue hook blocks before dispatch. Each lifecycle operation
        // supersedes an audited request before any path, model or codec read.
        for (action,mode) in [(0,NativeLoadIdentity.managedQwen3),(1,.managedQwen3),(2,.managedQwen3),
                              (0,.managedSmol),(1,.managedSmol),(2,.managedSmol),
                              (0,.managedQwen2),(1,.managedQwen2),(2,.managedQwen2),
                              (0,.managedMistral),(1,.managedMistral),(2,.managedMistral)] {
            let gate = IdentityDequeueGate()
            let worker = try NativeWorker(beforeCommandDequeue:{ gate.pause($0) })
            var settings = NativeSettings(directory:folder.path)
            settings.metal = true; settings.mlx = true; settings.chat = true
            settings.loadIdentity = mode; settings.loadRequestID = request.id
            var failure: Error?
            do {
                try require(worker.submit(.load(settings)),"Queued routed load rejected")
                try require(gate.wait(),"Load dequeue gate deadline")
                try require(gate.settings?.loadIdentity == mode && gate.settings?.loadRequestID == request.id,
                            "Captured audited intent changed before dequeue")
                if action == 2 { worker.close() } else { worker.interrupt(unload:action == 0) }
                gate.resume()
                if action == 2 { try require(worker.waitUntilClosed(seconds:30),"Close deadline") }
                else { try require(worker.waitUntilIdle(seconds:30) != nil,"Cleanup deadline") }
                let actual = worker.snapshot()
                try require(!actual.hasModel && !actual.hasSession && actual.acceptedIdentity == nil && actual.acceptedLoadRequestID == nil && !gate.timedOut,
                            "Superseded load published owner or identity")
            } catch { failure = error }
            gate.resume(); worker.close()
            try require(worker.waitUntilClosed(seconds:30),"Final owning thread did not close")
            if let failure { throw failure }
        }
        let worker = try NativeWorker()
        var failure: Error?
        do {
            var invalid = NativeSettings(directory:folder.path); invalid.loadIdentity = .managedQwen3; invalid.loadRequestID = UUID()
            try require(worker.submit(.load(invalid)),"Invalid-route test could not enqueue")
            guard let result = worker.waitUntilIdle(seconds:30) else { throw NativeContextFailure("Invalid-route idle deadline") }
            try require(result.status != 0 && result.message.contains("Audited managed identity requires") && !result.hasModel
                        && result.acceptedIdentity == nil && result.acceptedLoadRequestID == nil,"Invalid audited request fell back to ordinary loading")
        } catch { failure = error }
        worker.close(); try require(worker.waitUntilClosed(seconds:30),"Rejected-route worker did not close")
        if let failure { throw failure }
        guard checks == 116 else { throw NativeContextFailure("Control inventory changed") }
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:["checks":checks,"passed":true,"model_loads":0,"stale_queue_actions":12],options:[.sortedKeys]) + Data([10]))
    }
}

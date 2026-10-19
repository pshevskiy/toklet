import Foundation

struct WorkerCheckState: Codable, Sendable {
    let name: String
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
    let saved: Int
    let reason: UInt32
}
struct WorkerCheckResult: Codable, Sendable {
    let passed: Bool
    let backend: String
    let states: [WorkerCheckState]
    let tokens: [UInt32]
    let closed: Bool
    let released: Bool
    let physicalDeviceQualified: Bool
}
private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain:"TokletChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func settled(_ worker: NativeWorker) throws -> NativeSnapshot {
    let deadline = Date(timeIntervalSinceNow: 120)
    while Date() < deadline {
        let state = worker.snapshot()
        if !state.busy { return state }
        Thread.sleep(forTimeInterval: 0.002)
    }
    throw NSError(domain:"TokletChecks",code:2,userInfo:[NSLocalizedDescriptionKey:"Worker deadline"])
}

// Shared by the Mac executable and the simulator app's explicit self-test mode.
// These checks do not perform physical-device/performance qualification.
func runWorkerChecks(directory: String, metal: Bool) throws -> WorkerCheckResult {
    var owner: NativeWorker? = try NativeWorker()
    weak let weakOwner = owner
    var states: [WorkerCheckState] = []
    var ids: [UInt32] = []
    do {
        let worker = owner!
        defer { worker.close() }
        func perform(_ command: NativeCommand, _ name: String) throws -> NativeSnapshot {
            try expect(worker.submit(command),"Command was rejected: \(name)")
            let state = try settled(worker)
            try expect(state.status == 0,"\(name): \(state.message)")
            states.append(WorkerCheckState(name:name,history:state.history,evaluated:state.evaluated,
                                           pending:state.pending,saved:state.savedHistory,reason:state.reason))
            return state
        }
        var settings = NativeSettings(directory: directory)
        settings.metal = metal
        var state = try perform(.load(settings),"load")
        try expect(state.hasModel && state.hasSession,"Missing loaded native owners")
        state = try perform(.generate(operation:0,tokens:[1,7,13,3],maximum:2,stops:[]),"prompt")
        try expect(state.ids == [47,47] && state.history == 6 && state.evaluated == 5 && state.pending == 1,"Prompt oracle/state")
        ids += state.ids
        state = try perform(.generate(operation:2,tokens:[],maximum:2,stops:[]),"continue")
        try expect(state.ids == [47,47] && state.history == 8 && state.evaluated == 7 && state.pending == 1,"Continuation oracle/state")
        ids += state.ids
        state = try perform(.suspend,"suspend")
        try expect(state.hasModel && !state.hasSession && state.savedHistory == 8,"Suspend must retain only immutable model/prefix")
        try expect(worker.submit(.replay,initiallyCancelled:true),"Precancelled replay rejected")
        state = try settled(worker)
        try expect(state.status == 0 && state.reason == 3 && !state.hasSession && state.savedHistory == 8,"Precancelled replay lost retained prefix")
        states.append(WorkerCheckState(name:"replay-cancelled",history:state.history,evaluated:state.evaluated,
                                       pending:state.pending,saved:state.savedHistory,reason:state.reason))
        state = try perform(.replay,"replay")
        try expect(state.history == 8 && state.evaluated == 8 && state.pending == 0,"Replay eagerly commits full sampled history")
        state = try perform(.generate(operation:2,tokens:[],maximum:1,stops:[]),"after-replay")
        try expect(state.ids == [47] && state.history == 9 && state.pending == 1,"Replay continuation oracle")
        _ = try perform(.reset,"reset")
        state = try perform(.generate(operation:0,tokens:[1,7,13,3],maximum:4,stops:[],autoCancelAfter:1),"cancel")
        try expect(state.ids == [47] && state.reason == 3 && state.history == 5 && state.evaluated == 4 && state.pending == 1,"Deterministic callback cancellation")
        state = try perform(.generate(operation:2,tokens:[],maximum:1,stops:[]),"resume")
        try expect(state.ids == [47] && state.history == 6 && state.pending == 1,"Resume after cancellation")
        // Cross-thread lifecycle path supersedes queued work and clears replay
        // state on full unload. No native resource is freed from this caller.
        worker.interrupt(unload:true)
        state = try settled(worker)
        try expect(state.status == 0 && !state.hasModel && !state.hasSession && state.savedHistory == 0,"Full lifecycle unload")
        worker.close()
        try expect(worker.waitUntilClosed(seconds:120),"Close deadline")
        try expect(worker.snapshot().workerFinished,"Worker failed to acknowledge close")
    }
    owner = nil
    let releaseDeadline = Date(timeIntervalSinceNow: 5)
    while weakOwner != nil && Date() < releaseDeadline { Thread.sleep(forTimeInterval:0.002) }
    try expect(weakOwner == nil,"Worker/thread retain cycle after close")
    return WorkerCheckResult(passed:true,backend:metal ? "metal" : "cpu",states:states,tokens:ids,
                             closed:true,released:true,physicalDeviceQualified:false)
}

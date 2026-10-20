import Foundation

struct WorkerCheckState: Codable, Sendable {
    let name: String
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
    let saved: Int
    let reason: UInt32
}
struct WorkerTimingCheck: Codable, Sendable {
    let name: String
    let submittedUptimeNanoseconds: UInt64
    let completedUptimeNanoseconds: UInt64
    let tokens: [UInt32]
    let events: [NativeTokenTiming]
}
struct WorkerCheckResult: Codable, Sendable {
    let passed: Bool
    let backend: String
    let states: [WorkerCheckState]
    let timingChecks: [WorkerTimingCheck]
    let tokens: [UInt32]
    let closed: Bool
    let released: Bool
    let physicalDeviceQualified: Bool
}
private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain:"TokletChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func settled(_ worker: NativeWorker) throws -> NativeSnapshot {
    guard let state = worker.waitUntilIdle(seconds:120) else {
        throw NSError(domain:"TokletChecks",code:2,userInfo:[NSLocalizedDescriptionKey:"Worker deadline"])
    }
    try expect(state.tokenTimings.count == state.ids.count,"Missing native token timing")
    for (index,timing) in state.tokenTimings.enumerated() {
        try expect(timing.index == UInt32(index) && timing.nativeElapsedSeconds.isFinite
                   && timing.nativeElapsedSeconds >= 0,"Invalid native token timing")
        if index > 0 {
            let previous = state.tokenTimings[index-1]
            try expect(timing.nativeElapsedSeconds >= previous.nativeElapsedSeconds
                       && timing.callbackUptimeNanoseconds >= previous.callbackUptimeNanoseconds,
                       "Token clocks moved backwards")
        }
    }
    return state
}

// Shared by the Mac executable and the simulator app's explicit self-test mode.
// These checks do not perform physical-device/performance qualification.
func runWorkerChecks(directory: String, metal: Bool) throws -> WorkerCheckResult {
    var owner: NativeWorker? = try NativeWorker()
    weak let weakOwner = owner
    var states: [WorkerCheckState] = []
    var ids: [UInt32] = []
    var timingChecks: [WorkerTimingCheck] = []
    do {
        let worker = owner!
        defer { worker.close() }
        try expect(worker.waitUntilIdle(seconds:0)?.busy == false,"Idle completion query")
        try expect(worker.waitUntilIdle(seconds:.nan) == nil
                   && worker.waitUntilIdle(seconds:-1) == nil
                   && worker.waitUntilIdle(seconds:3601) == nil,"Invalid wait deadline admitted")
        func perform(_ command: NativeCommand, _ name: String) throws -> NativeSnapshot {
            let submitted = DispatchTime.now().uptimeNanoseconds
            try expect(worker.submit(command),"Command was rejected: \(name)")
            let state = try settled(worker)
            let completed = DispatchTime.now().uptimeNanoseconds
            try expect(state.tokenTimings.allSatisfy {
                $0.callbackUptimeNanoseconds >= submitted && $0.callbackUptimeNanoseconds <= completed
            },"Callback clocks outside completed request")
            timingChecks.append(WorkerTimingCheck(name:name,submittedUptimeNanoseconds:submitted,
                                completedUptimeNanoseconds:completed,tokens:state.ids,events:state.tokenTimings))
            try expect(state.status == 0,"\(name): \(state.message)")
            states.append(WorkerCheckState(name:name,history:state.history,evaluated:state.evaluated,
                                           pending:state.pending,saved:state.savedHistory,reason:state.reason))
            return state
        }
        var settings = NativeSettings(directory: directory)
        settings.metal = metal
        var state = try perform(.load(settings),"load")
        try expect(state.hasModel && state.hasSession,"Missing loaded native owners")
        try expect(state.tokenTimings.isEmpty,"Load retained previous timing events")
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
        state = try perform(.reset,"reset")
        try expect(state.ids.isEmpty && state.tokenTimings.isEmpty,"Reset retained old emitted events")
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
    return WorkerCheckResult(passed:true,backend:metal ? "metal" : "cpu",states:states,timingChecks:timingChecks,tokens:ids,
                             closed:true,released:true,physicalDeviceQualified:false)
}

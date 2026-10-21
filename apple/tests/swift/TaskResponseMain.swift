import Foundation
import Darwin

private func expectTask(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw NSError(domain:"TokletTaskChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func rejectTask(_ body: () throws -> Void) throws {
    var rejected = false;do { try body() } catch { rejected = true }
    try expectTask(rejected,"Invalid task sequence accepted")
}
private final class TaskCommands {
    var prefix: [UInt32] = []
    var calls = 0;var generated = 0;var stopAt = 0;var failAt = 0
    var cancelled = false
    var corrupt: ((inout NativeSnapshot) -> Void)?
    func active() throws { try expectTask(!cancelled,"Latched cancellation") }
    func perform(_ command: NativeCommand) throws -> NativeSnapshot {
        calls += 1
        var state = NativeSnapshot();state.hasModel = true;state.hasSession = true
        switch command {
        case .reset: prefix = [];generated = 0
        case .generate(let operation,let tokens,let maximum,let stops,let cancel):
            try expectTask(cancel == 0 && maximum <= 128 && maximum > 0,"Unbounded command")
            if generated == 0 { try expectTask(operation == 0 && tokens == [1,7,13],"Prompt repeated/differs");prefix = tokens }
            else { try expectTask(operation == 2 && tokens.isEmpty,"Continuation changed prefix") }
            if failAt > 0 && generated >= failAt { throw NSError(domain:"InjectedCommand",code:1) }
            state.reason = 1
            for index in 0..<Int(maximum) {
                generated += 1
                if generated == stopAt {
                    prefix.append(stops[0]);state.reason = 2;state.stopToken = stops[0];break
                }
                let token = UInt32(generated % 50);prefix.append(token);state.ids.append(token)
                state.tokenTimings.append(NativeTokenTiming(index:UInt32(index),nativeElapsedSeconds:Double(index)/10,
                    callbackUptimeNanoseconds:UInt64(generated)))
            }
            state.pending = 1;corrupt?(&state)
        default: throw NSError(domain:"UnexpectedCommand",code:1)
        }
        state.history = UInt32(prefix.count);state.evaluated = state.history-state.pending;state.savedHistory = prefix.count
        return state
    }
    func collect(maximum: Int = 256, limit: Int = 128,
                 onChunk: (NativeTaskChunk) throws -> Void = { _ in }) throws -> NativeTaskResponse {
        try collectNativeTaskResponse(prompt:[1,7,13],maximum:maximum,capacity:259,vocabulary:64,
            stops:[62,63],commandLimit:limit,checkActive:active,perform:perform,onChunk:onChunk)
    }
}
private func simulatedTaskChecks() throws -> [String] {
    var names: [String] = []
    do {
        let model = TaskCommands();var rows: [NativeTaskChunk] = []
        let result = try model.collect(onChunk:{ rows.append($0) })
        try expectTask(result.ids == (1...256).map { UInt32($0%50) } && result.sampled == 256
            && result.reason == 1 && result.stopToken == nil && result.history == 259
            && result.evaluated == 258 && result.pending == 1 && result.chunks == 2,"256 token response differs")
        try expectTask(rows.map(\.requested) == [128,128] && rows.map(\.history) == [131,259]
            && rows.allSatisfy { $0.tokenTimings.first?.index == 0 } && model.calls == 3,"Chunk boundary differs")
        names.append("bounded-256-continuation")
    }
    for boundary in [1,128,129,256] {
        let model = TaskCommands();model.stopAt = boundary
        let result = try model.collect()
        try expectTask(result.ids.count == boundary-1 && result.sampled == boundary && result.reason == 2
            && result.stopToken == 62 && result.history == UInt32(boundary+3)
            && result.evaluated == UInt32(boundary+2) && result.pending == 1
            && result.chunks == (boundary+127)/128 && model.prefix.last == 62,"Hidden stop budget/prefix differs")
        names.append("hidden-stop-\(boundary)")
    }
    do {
        let model = TaskCommands();var rows = 0
        try rejectTask { _ = try model.collect(onChunk:{ _ in rows += 1;model.cancelled = true }) }
        try expectTask(rows == 1 && model.calls == 2 && model.prefix.count == 131,"Cancel between commands resumed")
        names.append("latched-between-chunks")
    }
    do {
        let model = TaskCommands();model.corrupt = { $0.reason = 3 };var rows = 0
        try rejectTask { _ = try model.collect(onChunk:{ _ in rows += 1 }) }
        try expectTask(rows == 0 && model.calls == 2 && model.prefix.count == 131,"Final-callback cancellation accepted")
        names.append("final-callback-cancelled")
    }
    do {
        let model = TaskCommands();model.failAt = 128;var rows = 0
        try rejectTask { _ = try model.collect(onChunk:{ _ in rows += 1 }) }
        try expectTask(rows == 1 && model.calls == 3 && model.prefix.count == 131,"Failed continuation lost successful prefix")
        names.append("continuation-failure")
    }
    do {
        let model = TaskCommands();var rows = 0
        try rejectTask { _ = try model.collect(onChunk:{ row in
            if row.index == 1 { throw NSError(domain:"InjectedSink",code:1) };rows += 1
        }) }
        try expectTask(rows == 1 && model.calls == 3 && model.prefix.count == 259,"Sink failure changed native prefix")
        names.append("sink-failure-after-valid-chunk")
    }
    do {
        let mutations: [(inout NativeSnapshot) -> Void] = [
            { $0.stopToken = 62 },{ $0.reason = 2;$0.stopToken = 61 },
            { $0.ids[0] = 64 },{ $0.ids[0] = 62 },{ $0.tokenTimings.removeLast() },
            { $0.tokenTimings[0] = NativeTokenTiming(index:1,nativeElapsedSeconds:0,callbackUptimeNanoseconds:1) },
            { $0.tokenTimings[1] = NativeTokenTiming(index:1,nativeElapsedSeconds:Double.nan,callbackUptimeNanoseconds:2) },
            { $0.tokenTimings[1] = NativeTokenTiming(index:1,nativeElapsedSeconds:-1,callbackUptimeNanoseconds:2) },
            { $0.tokenTimings[1] = NativeTokenTiming(index:1,nativeElapsedSeconds:0.1,callbackUptimeNanoseconds:0) },
            { $0.status = 1 },{ $0.busy = true },{ $0.hasSession = false }]
        for mutation in mutations { let model = TaskCommands();model.corrupt = mutation;try rejectTask { _ = try model.collect() } }
        let model = TaskCommands();var calls = 0
        try rejectTask { _ = try model.collectNativeInvalidPrefix { calls += 1 } }
        try expectTask(calls == 0,"Wrong prefix emitted")
        names.append("invalid-events-and-prefix")
    }
    do {
        let model = TaskCommands()
        for maximum in [0,257,Int.max] {
            try rejectTask { _ = try model.collect(maximum:maximum) }
        }
        for limit in [0,129] { try rejectTask { _ = try model.collect(limit:limit) } }
        try rejectTask { _ = try collectNativeTaskResponse(prompt:[1,7,13],maximum:256,capacity:258,vocabulary:64,
            stops:[62,63],checkActive:model.active,perform:model.perform) }
        try rejectTask { _ = try collectNativeTaskResponse(prompt:Array(repeating:1,count:307),maximum:1,capacity:562,vocabulary:64,
            stops:[62,63],checkActive:model.active,perform:model.perform) }
        try expectTask(model.calls == 0,"Preflight changed prior session")
        names.append("whole-request-preflight")
    }
    return names
}
private extension TaskCommands {
    func collectNativeInvalidPrefix(onChunk: () -> Void) throws -> NativeTaskResponse {
        try collectNativeTaskResponse(prompt:[1,7,13],maximum:4,capacity:7,vocabulary:64,stops:[62,63],
            checkActive:active,perform:{ command in
                var state = try self.perform(command)
                if state.reason == 1 { state.history += 1 }
                return state
            },onChunk:{ _ in onChunk() })
    }
}
private struct TaskChecks: Encodable {
    let passed: Bool
    let backend: String
    let simulated: [String]
    let oneCall: NativeTaskResponse
    let chunked: NativeTaskResponse
    let hiddenStop: NativeTaskResponse
    let lifecycleCleared: Bool
    let closed: Bool
    let released: Bool
    let physicalDeviceQualified = false
}
@main private struct TaskResponseMain {
    static func main() {
        do {
            guard CommandLine.arguments.count == 2 || (CommandLine.arguments.count == 3 && CommandLine.arguments[2] == "metal") else { throw NSError(domain:"TaskArguments",code:1) }
            let metal = CommandLine.arguments.count == 3
            let simulated = try simulatedTaskChecks()
            var owner: NativeWorker? = try NativeWorker();weak let weakOwner = owner
            let oneCall: NativeTaskResponse;let chunked: NativeTaskResponse;let hiddenStop: NativeTaskResponse
            do {
                let worker = owner!;defer { worker.close() }
                func perform(_ command: NativeCommand) throws -> NativeSnapshot {
                    try expectTask(worker.submit(command),"Actual command rejected")
                    guard let state = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TaskDeadline",code:1) }
                    return state
                }
                var settings = NativeSettings(directory:CommandLine.arguments[1]);settings.metal = metal;settings.capacity = 11;settings.chunk = 3
                let loaded = try perform(.load(settings))
                try expectTask(loaded.status == 0,"Actual load failed")
                func collect(_ limit: Int, _ stops: [UInt32]) throws -> NativeTaskResponse {
                    try collectNativeTaskResponse(prompt:[1,7,13],maximum:8,capacity:11,vocabulary:64,
                        stops:stops,commandLimit:limit,checkActive:{},perform:perform)
                }
                oneCall = try collect(8,[62,63]);chunked = try collect(3,[62,63])
                try expectTask(oneCall.ids == chunked.ids && oneCall.sampled == chunked.sampled
                    && oneCall.history == chunked.history && oneCall.evaluated == chunked.evaluated
                    && oneCall.pending == chunked.pending && oneCall.reason == chunked.reason
                    && oneCall.stopToken == chunked.stopToken,"Chunking changed native result")
                try expectTask(!oneCall.ids.isEmpty,"Tiny oracle emitted no token")
                hiddenStop = try collect(3,[oneCall.ids[0]])
                try expectTask(hiddenStop.ids.isEmpty && hiddenStop.sampled == 1 && hiddenStop.stopToken == oneCall.ids[0]
                    && hiddenStop.history == 4 && hiddenStop.evaluated == 3 && hiddenStop.pending == 1,"Actual hidden stop differs")
                let failed = try perform(.score(target:0))
                try expectTask(failed.status != 0 && failed.stopToken == .max,"Failed command retained hidden stop")
                for command in [NativeCommand.suspend,.replay,.unload] {
                    let state = try perform(command)
                    try expectTask(state.status == 0 && state.stopToken == .max,"Lifecycle retained hidden stop")
                }
                worker.close();try expectTask(worker.waitUntilClosed(seconds:120) && worker.snapshot().workerFinished
                    && worker.snapshot().stopToken == .max,"Actual close failed or retained a hidden stop")
            }
            owner = nil;let deadline = Date(timeIntervalSinceNow:5)
            while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
            try expectTask(weakOwner == nil,"Task worker retained")
            let result = TaskChecks(passed:true,backend:metal ? "metal":"cpu",simulated:simulated,oneCall:oneCall,
                chunked:chunked,hiddenStop:hiddenStop,lifecycleCleared:true,closed:true,released:true)
            print(String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
        } catch { fputs("Task-response check: \(error)\n",stderr);exit(1) }
    }
}

import Foundation
private func expect(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw NSError(domain:"TokletLifecycleChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private struct Result: Encodable {
    let passed: Bool
    let backend: String
    let trace: NativeLifecycleTraceReport
    let cases: [String]
    let closed: Bool
    let released: Bool
    let physicalDeviceQualified = false
    let actualOSLifecycleQualified = false
}
@main struct LifecycleChecks {
    static func main() throws {
        let args = ProcessInfo.processInfo.arguments
        guard args.count == 2 || args.count == 3 && args[2] == "metal" else { throw NSError(domain:"TokletLifecycleChecks",code:2) }
        let metal = args.count == 3
        var settings = NativeSettings(directory:args[1]);settings.metal = metal
        let prompt: [UInt32] = [1,7,13,3]
        func perform(_ worker: NativeWorker, _ command: NativeCommand) throws -> NativeSnapshot {
            try expect(worker.submit(command),"Command rejected")
            guard let state = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletLifecycleChecks",code:3) }
            return state
        }
        func rejects(_ body: () throws -> Void) throws {
            var rejected = false
            do { try body() } catch { rejected = true }
            try expect(rejected,"Invalid trace request was admitted")
        }
        var owner: NativeWorker? = try NativeWorker(captureSavedPrefix:true)
        weak let weakOwner = owner
        let trace = try NativeLifecycleTrace(worker:owner!,maximumPrefix:32)
        do {
            let worker = owner!;defer { worker.close() }
            try trace.begin("load");_ = try perform(worker,.load(settings))
            try trace.record("loaded",origin:.coordinator,applicationState:nil);try trace.end("load")
            try trace.begin("generate")
            let cancelled = try perform(worker,.generate(operation:0,tokens:prompt,maximum:4,stops:[],autoCancelAfter:1))
            try expect(cancelled.reason == 3 && cancelled.savedPrefix == prompt+[47],"Pending cancellation oracle")
            try trace.record("cancelled",origin:.coordinator,applicationState:nil);try trace.end("generate")
            try trace.record("inactive-request",origin:.syntheticCheck,applicationState:1,action:.inactive)
            try trace.record("background-request",origin:.syntheticCheck,applicationState:2,action:.background)
            guard let suspended = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletLifecycleChecks",code:3) }
            try expect(suspended.hasModel && !suspended.hasSession && suspended.savedPrefix == prompt+[47],"Session cleanup prefix")
            try trace.record("session-released",origin:.coordinator,applicationState:2)
            try trace.record("foreground-observed",origin:.syntheticCheck,applicationState:0,action:.active)
            try expect(!worker.snapshot().hasSession,"Foreground silently replayed")
            try trace.begin("replay");let replayed = try perform(worker,.replay)
            try expect(replayed.savedPrefix == prompt+[47] && replayed.evaluated == 5 && replayed.pending == 0,"Exact replay prefix")
            try trace.record("replayed",origin:.coordinator,applicationState:0);try trace.end("replay")
            try trace.begin("continue");let resumed = try perform(worker,.generate(operation:2,tokens:[],maximum:1,stops:[]))
            try expect(resumed.ids == [47] && resumed.savedPrefix == prompt+[47,47],"Resume oracle")
            try trace.record("resumed",origin:.coordinator,applicationState:0);try trace.end("continue")
            do {
                let reference = try NativeWorker(captureSavedPrefix:true);defer { reference.close() }
                _ = try perform(reference,.load(settings))
                let uninterrupted = try perform(reference,.generate(operation:0,tokens:prompt,maximum:2,stops:[]))
                try expect(uninterrupted.savedPrefix == resumed.savedPrefix,"Uninterrupted reference differs")
                reference.close();try expect(reference.waitUntilClosed(seconds:120),"Reference close")
            }
            try trace.record("warning-request",origin:.syntheticCheck,applicationState:0,action:.memoryWarning)
            // A subsequent background request cannot weaken full unload.
            try trace.record("background-after-warning",origin:.syntheticCheck,applicationState:2,action:.background)
            guard let unloaded = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletLifecycleChecks",code:3) }
            try expect(!unloaded.hasModel && !unloaded.hasSession && unloaded.savedPrefix == [],"Warning unload precedence")
            try trace.record("warning-unloaded",origin:.coordinator,applicationState:0)
            let rejected = try perform(worker,.replay)
            try expect(rejected.status == 1 && !rejected.hasSession && rejected.savedPrefix == [],"Warning requires fresh load")
            try trace.record("replay-rejected",origin:.coordinator,applicationState:0)
            _ = try perform(worker,.load(settings));try trace.record("fresh-load",origin:.coordinator,applicationState:0)
            try trace.begin("close");worker.close();try expect(worker.waitUntilClosed(seconds:120),"Close timeout")
            try trace.record("closed",origin:.coordinator,applicationState:0);try trace.end("close");try trace.detach()
        }
        owner = nil
        let deadline = Date(timeIntervalSinceNow:5)
        while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
        try expect(weakOwner == nil,"Trace retained worker owner")
        let complete = trace.report()
        try expect(complete.error == nil && complete.activeCommand == nil && complete.detachedAfterClose,"Incomplete trace close")
        var cases: [String] = []
        do {
            let worker = try NativeWorker(captureSavedPrefix:true);defer { worker.close() }
            try rejects { _ = try NativeLifecycleTrace(worker:worker,maximumEvents:0) };cases.append("event-bounds")
            try rejects { _ = try NativeLifecycleTrace(worker:worker,maximumPrefix:1153) };cases.append("prefix-bounds")
            for (name,action) in [("bad-label",0),("overlap",1),("wrong-end",2),("future-clock",3),("zero-clock",4),("application-state",5),("premature-detach",6)] {
                let bad = try NativeLifecycleTrace(worker:worker,maximumPrefix:32)
                try rejects {
                    switch action {
                    case 0: try bad.begin("bad label")
                    case 1: try bad.begin("one");try bad.begin("two")
                    case 2: try bad.begin("one");try bad.end("two")
                    case 3: try bad.record("future",origin:.syntheticCheck,applicationState:0,observedUptimeNanoseconds:UInt64.max)
                    case 4: try bad.record("zero",origin:.syntheticCheck,applicationState:0,observedUptimeNanoseconds:0)
                    case 5: try bad.record("bad-state",origin:.syntheticCheck,applicationState:3)
                    default: try bad.detach()
                    }
                }
                try expect(bad.report().error != nil && !bad.report().detachedAfterClose,"Failure evidence unavailable")
                try rejects { _ = try bad.snapshot() };cases.append(name)
            }
            let times = try NativeLifecycleTrace(worker:worker)
            let first = try times.record("first",origin:.syntheticCheck,applicationState:nil)
            let older = try times.record("older-observation",origin:.syntheticCheck,applicationState:nil,observedUptimeNanoseconds:1)
            try expect(older.observedUptimeNanoseconds < first.observedUptimeNanoseconds
                && older.recordedUptimeNanoseconds >= first.recordedUptimeNanoseconds,"Observation time rewritten")
            cases.append("separate-observation-clock")
            _ = try perform(worker,.load(settings));_ = try perform(worker,.generate(operation:0,tokens:prompt,maximum:2,stops:[]))
            let tooSmall = try NativeLifecycleTrace(worker:worker,maximumPrefix:4,afterRequest:{ _ = worker.waitUntilIdle(seconds:120) })
            try rejects { try tooSmall.record("warning",origin:.syntheticCheck,applicationState:0,action:.memoryWarning) }
            try expect(!worker.snapshot().hasModel && tooSmall.report().events.isEmpty && tooSmall.report().error != nil,
                       "Oversized pre-action prefix escaped after completed cleanup")
            cases.append("pre-action-bound-after-cleanup")
            _ = try perform(worker,.load(settings));_ = try perform(worker,.generate(operation:0,tokens:prompt,maximum:1,stops:[]))
            let full = try NativeLifecycleTrace(worker:worker,maximumEvents:1,maximumPrefix:32,afterRequest:{ _ = worker.waitUntilIdle(seconds:120) })
            try full.record("one",origin:.syntheticCheck,applicationState:0)
            try rejects { try full.record("background",origin:.syntheticCheck,applicationState:2,action:.background) }
            try expect(worker.snapshot().hasModel && !worker.snapshot().hasSession && worker.snapshot().savedPrefix == prompt+[47],"Trace exhaustion blocked suspension")
            try rejects { try full.record("warning",origin:.syntheticCheck,applicationState:0,action:.memoryWarning) }
            try expect(!worker.snapshot().hasModel && !worker.snapshot().hasSession && worker.snapshot().savedPrefix == [],"Prior trace failure blocked unload")
            try expect(full.report().events.count == 1 && full.report().error != nil,"Failure trace not bounded")
            cases += ["exhausted-trace-background","failed-trace-warning"]
            worker.close();try expect(worker.waitUntilClosed(seconds:120),"Invalid trace worker close")
        }
        let result = Result(passed:true,backend:metal ? "metal":"cpu",trace:complete,cases:cases,closed:true,released:true)
        print(String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
    }
}

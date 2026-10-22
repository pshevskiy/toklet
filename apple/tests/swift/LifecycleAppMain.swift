import Foundation
import CryptoKit
private func expect(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw NSError(domain:"TokletLifecycleAppChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func rejects(_ body: () throws -> Void) throws {
    var failed = false;do { try body() } catch { failed = true }
    try expect(failed,"Expected lifecycle rejection")
}
private struct Result: Encodable {
    let passed: Bool
    let backend: String
    let kind: String
    let cases: [String]
    let trips: [NativeLifecycleRoundTrip]
    let diagnostic: NativeLifecycleResult
    let actualOSLifecycleQualified = false
    let physicalDeviceQualified = false
}
@main struct LifecycleAppChecks {
    static func main() async throws {
        try await Task.detached { try run() }.value
    }
    private static func run() throws {
        let args = ProcessInfo.processInfo.arguments
        guard args.count == 5, ["unit","small"].contains(args[4]) else { throw NSError(domain:"TokletLifecycleAppChecks",code:2) }
        let root = URL(fileURLWithPath:args[1],isDirectory:true)
        let config = try admitLifecycleConfig(root:root,name:args[2],sha256:args[3]);let kind = args[4]
        var cases: [String] = [];var trips: [NativeLifecycleRoundTrip] = []
        if kind == "unit" {
            let base = try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent(args[2]))) as! [String:Any]
            func invalid(_ name: String, mutate: (inout [String:Any]) -> Void) throws {
                var object = base;mutate(&object)
                let data = try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys])
                let digest = SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined();let path = "M7Jobs/"+digest+".json"
                try data.write(to:root.appendingPathComponent(path))
                try rejects { _ = try admitLifecycleConfig(root:root,name:path,sha256:digest) };cases.append(name)
            }
            try invalid("config-version") { $0["version"] = 2 }
            try invalid("config-boolean") { $0["version"] = true }
            try invalid("config-extra") { $0["extra"] = 1 }
            try invalid("config-capacity") { $0["capacity"] = 0 }
            try invalid("config-chunk") { $0["chunk"] = 129 }
            try invalid("config-output") { $0["maximum"] = 129 }
            try invalid("config-reference-space") { $0["capacity"] = config.capacity-1 }
            try invalid("config-vocabulary") { $0["vocabulary"] = 0 }
            try invalid("config-prompt-range") { $0["prompt"] = [UInt32.max] }
            try invalid("config-empty-prompt") { $0["prompt"] = [UInt32]() }
            try invalid("config-budget-overflow") { $0["processBudget"] = UInt64.max }
            try invalid("config-traversal") { $0["modelFolder"] = "../Model" }
            try invalid("config-overlap") { $0["metal"] = true;$0["packedFolder"] = config.modelFolder }
            try invalid("config-inventory") { $0["files"] = [String:String]() }
            try rejects { _ = try admitLifecycleConfig(root:root,name:"other.json",sha256:args[3]) };cases.append("config-canonical-path")
            var huge = Data(repeating:32,count:512*1024+1)
            let digest = SHA256.hash(data:huge).map { String(format:"%02x",$0) }.joined();let largeName = "M7Jobs/"+digest+".json"
            try huge.write(to:root.appendingPathComponent(largeName));huge = Data()
            try rejects { _ = try admitLifecycleConfig(root:root,name:largeName,sha256:digest) };cases.append("config-size")
            for returningInactive in [false,true] {
                let worker = try NativeWorker(captureSavedPrefix:true);defer { worker.close() }
                let trace = try NativeLifecycleTrace(worker:worker,maximumPrefix:32);let signals = NativeLifecycleSignals()
                try signals.attach(worker,trace:trace);try signals.arm()
                signals.observe(.inactive,origin:.syntheticCheck,applicationState:1,observedUptimeNanoseconds:1)
                signals.observe(.background,origin:.syntheticCheck,applicationState:2,observedUptimeNanoseconds:2)
                if returningInactive { signals.observe(.inactive,origin:.syntheticCheck,applicationState:1,observedUptimeNanoseconds:3) }
                signals.observe(.active,origin:.syntheticCheck,applicationState:0,observedUptimeNanoseconds:4)
                let trip = try signals.finish(seconds:0,requireSystem:false)
                try expect(trip.inactive.observedUptimeNanoseconds == 1 && trip.active.observedUptimeNanoseconds == 4
                    && (trip.returnInactive != nil) == returningInactive,"Original phase observation changed")
                trips.append(trip);signals.detach();worker.close();try expect(worker.waitUntilClosed(seconds:120),"Signals close")
                cases.append(returningInactive ? "return-inactive-sequence":"direct-return-sequence")
            }
            for (name,actions) in [("duplicate-inactive",[NativeLifecycleAction.inactive,.inactive]),("background-before-inactive",[.background]),
                                   ("active-before-background",[.active]),("duplicate-background",[.inactive,.background,.background]),
                                   ("duplicate-return-inactive",[.inactive,.background,.inactive,.inactive])] {
                let worker = try NativeWorker();defer { worker.close() };let trace = try NativeLifecycleTrace(worker:worker);let signals = NativeLifecycleSignals()
                try signals.attach(worker,trace:trace);try signals.arm()
                for action in actions { signals.observe(action,origin:.syntheticCheck,applicationState:nil) }
                try rejects { try signals.check() };signals.detach();worker.close();try expect(worker.waitUntilClosed(seconds:120),"Invalid source close");cases.append(name)
            }
            do {
                let worker = try NativeWorker();defer { worker.close() };let trace = try NativeLifecycleTrace(worker:worker);let signals = NativeLifecycleSignals()
                try signals.attach(worker,trace:trace);try signals.arm()
                try rejects { _ = try signals.finish(seconds:0,requireSystem:false) };cases.append("missing-roundtrip")
                for action in [NativeLifecycleAction.inactive,.background,.active] { signals.observe(action,origin:.syntheticCheck,applicationState:nil) }
                try rejects { _ = try signals.finish(seconds:0,requireSystem:true) };cases.append("synthetic-not-system")
                signals.detach();worker.close();try expect(worker.waitUntilClosed(seconds:120),"Provenance source close")
            }
            var live = NativeSnapshot();live.busy = true;live.hasModel = true;live.hasSession = true;live.ids = [47,47]
            let observed = NativeLifecycleEvent(index:2,name:"scene-inactive",origin:.syntheticCheck,observedUptimeNanoseconds:30,
                recordedUptimeNanoseconds:40,applicationState:1,command:"generate",before:NativeLifecycleState(live),state:NativeLifecycleState(live),
                footprint:NativeFootprint(physicalBytes:100,lifetimePeakPhysicalBytes:200,returnedNaturalCount:93))
            let trip = NativeLifecycleRoundTrip(inactive:observed,background:observed,active:observed,returnInactive:nil)
            let good = NativeLifecycleReady(eventIndex:1,submittedUptimeNanoseconds:10,observedUptimeNanoseconds:20,visibleIDs:[47],applicationState:0)
            try requireNativeLifecycleReady(good,trip:trip,maximum:25,requireSystem:true);cases.append("ready-before-interruption")
            for (name,ready) in [
                ("late-ready-index",NativeLifecycleReady(eventIndex:3,submittedUptimeNanoseconds:10,observedUptimeNanoseconds:20,visibleIDs:[47],applicationState:0)),
                ("late-ready-observation",NativeLifecycleReady(eventIndex:1,submittedUptimeNanoseconds:10,observedUptimeNanoseconds:35,visibleIDs:[47],applicationState:0)),
                ("background-ready",NativeLifecycleReady(eventIndex:1,submittedUptimeNanoseconds:10,observedUptimeNanoseconds:20,visibleIDs:[47],applicationState:2)),
                ("unknown-system-foreground",NativeLifecycleReady(eventIndex:1,submittedUptimeNanoseconds:10,observedUptimeNanoseconds:20,visibleIDs:[47],applicationState:nil)),
                ("wrong-ready-prefix",NativeLifecycleReady(eventIndex:1,submittedUptimeNanoseconds:10,observedUptimeNanoseconds:20,visibleIDs:[46],applicationState:0))] {
                try rejects { try requireNativeLifecycleReady(ready,trip:trip,maximum:25,requireSystem:true) };cases.append(name)
            }
            let endpoint = root.appendingPathComponent("Endpoint/Model",isDirectory:true);try FileManager.default.createDirectory(at:endpoint,withIntermediateDirectories:true)
            let file = endpoint.appendingPathComponent("config.json");let old = Data("original".utf8);try old.write(to:file)
            let files = ["Endpoint/Model/config.json":SHA256.hash(data:old).map { String(format:"%02x",$0) }.joined()]
            _ = try nativeArtifactProof(modelFolder:"Endpoint/Model",packedFolder:"",files:files,root:root)
            try Data("mutated".utf8).write(to:file);var memoryChecks: [String] = [];var reproofs = 0
            let endpoints = checkNativeLifecycleEndpoints(admitted:true,memory:{ name in
                memoryChecks.append(name);throw NSError(domain:"SyntheticBudgetFailure",code:1)
            },reprove:{ reproofs += 1;return try nativeArtifactProof(modelFolder:"Endpoint/Model",packedFolder:"",files:files,root:root) })
            try expect(endpoints.errors.count == 3 && endpoints.files.isEmpty && reproofs == 1
                && memoryChecks == ["closed","after-final-artifact-proof"],"Budget failure suppressed endpoint mutation proof")
            cases.append("independent-budget-and-mutation-failures")
            _ = checkNativeLifecycleEndpoints(admitted:false,memory:{ _ in },reprove:{ reproofs += 1;return [:] })
            try expect(reproofs == 1,"Unadmitted artifact endpoint was read");cases.append("no-unadmitted-reproof")
        }
        let signals = NativeLifecycleSignals()
        let diagnostic = runNativeLifecycle(config,root:root,signals:signals,requireSystem:false) { _ in
            if kind == "unit" { throw NSError(domain:"TokletLifecycleAppChecks",code:4,userInfo:[NSLocalizedDescriptionKey:"Forced READY sink failure"]) }
            signals.observe(.inactive,origin:.syntheticCheck,applicationState:1)
            signals.observe(.background,origin:.syntheticCheck,applicationState:2)
            signals.observe(.inactive,origin:.syntheticCheck,applicationState:1)
            signals.observe(.active,origin:.syntheticCheck,applicationState:0)
        }
        if kind == "unit" {
            try expect(!diagnostic.passed && diagnostic.closed && diagnostic.released && diagnostic.filesBefore == config.files && diagnostic.filesAfter == config.files
                && diagnostic.cleanupErrors.isEmpty && diagnostic.endpointErrors.isEmpty
                && diagnostic.trace?.error == nil && diagnostic.trace?.activeCommand == nil
                && diagnostic.trace?.detachedAfterClose == true,
                       "Failed diagnostic did not close/release/reproof/detach")
            try expect(["No demonstrably active generation window","Forced READY sink failure"].contains(diagnostic.error ?? ""),"Unexpected tiny diagnostic failure")
            cases.append("tiny-diagnostic-fails-closed")
        } else {
            try expect(diagnostic.passed && diagnostic.resumedIDs.count == 3 && diagnostic.resumedIDs == diagnostic.referenceIDs
                && diagnostic.freshIDs == Array((diagnostic.rebuiltIDs+diagnostic.referenceIDs).prefix(3)),"Real-model synthetic lifecycle failed")
            cases.append("small-synthetic-roundtrip")
        }
        let result = Result(passed:true,backend:config.metal ? "metal":"cpu",kind:kind,cases:cases,trips:trips,diagnostic:diagnostic)
        print(String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
    }
}

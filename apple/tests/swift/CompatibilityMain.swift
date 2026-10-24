import Foundation
import CToklet

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain:"CompatibilityChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private final class Gate: @unchecked Sendable {
    let reached = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
    private let lock = NSLock()
    private var armed = false
    func arm() { lock.lock();armed = true;lock.unlock() }
    func wait(_ command: NativeCommand) {
        lock.lock();let shouldWait = armed;lock.unlock()
        if case .load = command, shouldWait { reached.signal();_ = proceed.wait(timeout:.now()+30) }
    }
}
@main
struct CompatibilityMain {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NSError(domain:"CompatibilityChecks",code:1) }
        let root = URL(fileURLWithPath:CommandLine.arguments[1])
        let metal = CommandLine.arguments[2] == "metal"
        let budget: UInt64 = 128 * 1024 * 1024
        let context: UInt32 = 32
        let fixture = root.appendingPathComponent("tests/fixtures/mlx-qwen3-v1/tied").path
        let valid = try NativeCompatibility.inspect(directory:fixture,context:context,budget:budget)
        try expect(valid.status == .compatibleUntested && valid.backendAvailableInBuild == metal,"Build availability or graph label")
        try expect(valid.mayAttemptLoad == metal && valid.requestedMemoryLimitBytes == budget,"Exact budget or load boundary")
        try expect(valid.geometry?.hiddenSize == 32 && valid.geometry?.queryWidth == 64,"Independent projection width")
        try expect(!valid.textProfileChecked && !valid.qualificationEvaluated,"Metadata granted text qualification")
        let low = try NativeCompatibility.inspect(directory:fixture,context:context,budget:1)
        try expect(!low.mayAttemptLoad && low.loadAdmission == .exceedsBudget && low.summary.contains("budget"),"Budget display")
        let scalar = try NativeCompatibility.inspect(directory:fixture,backend:.scalarReference,context:context,budget:budget)
        try expect(scalar.mayAttemptLoad && !scalar.runtimeDeviceChecksRequired,"Scalar metadata path")
        // Independent wire mutations exercise decoder policy, not the Rust producer.
        var original: [String:Any] = [
            "schema_version":1,"backend":"metal","status":"compatible_untested","reason":"metadata_accepted","message":"Accepted metadata only",
            "architecture":"test_family","requested_context":context,"requested_memory_limit_bytes":budget,
            "geometry":["hidden_size":32,"query_width":64,"kv_width":32],
            "resources":["source_payload_bytes":1234,"metal_runtime_planning_required":true],
            "tensor_metadata_validated":true,"backend_available_in_build":true,"load_admission":"device_planning_required",
            "payload_validation_required":true,"runtime_device_checks_required":true,"session_admission_required":true,
            "text_profile_checked":false,"qualification_evaluated":false]
        _ = try NativeCompatibility.decode(JSONSerialization.data(withJSONObject:original),backend:.metal,context:context,budget:budget)
        let mutations: [(String,Any)] = [("schema_version",2),("backend","scalar_reference"),("requested_context",33),("requested_memory_limit_bytes",1),
            ("status","future_status"),("status","unknown"),("status","unsupported"),("reason","backend_unavailable"),("reason","load_budget_exceeded"),
            ("backend_available_in_build",false),("load_admission","exceeds_budget"),("load_admission","not_evaluated"),("load_admission","fits_scalar_load_estimate"),
            ("payload_validation_required",false),("runtime_device_checks_required",false),("session_admission_required",false),("text_profile_checked",true),
            ("qualification_evaluated",true),("tensor_metadata_validated",false),("geometry",NSNull()),("resources",NSNull())]
        for (key,value) in mutations {
            var changed = original;changed[key] = value
            var rejected = false
            do { _ = try NativeCompatibility.decode(JSONSerialization.data(withJSONObject:changed),backend:.metal,context:context,budget:budget) } catch { rejected = true }
            try expect(rejected,"Contradictory report accepted: " + key)
        }
        for bytes in [Data(),Data("{".utf8),Data(repeating:0,count:65537)] {
            var rejected = false
            do { _ = try NativeCompatibility.decode(bytes,backend:.metal,context:context,budget:budget) } catch { rejected = true }
            try expect(rejected,"Malformed report accepted")
        }
        original["reason"] = "backend_unavailable";original["backend_available_in_build"] = false
        let unavailable = try NativeCompatibility.decode(JSONSerialization.data(withJSONObject:original),backend:.metal,context:context,budget:budget)
        try expect(!unavailable.mayAttemptLoad && unavailable.summary.contains("unavailable"),"Unavailable backend display")
        original["reason"] = "load_budget_exceeded";original["load_admission"] = "exceeds_budget"
        let unavailableOverBudget = try NativeCompatibility.decode(JSONSerialization.data(withJSONObject:original),backend:.metal,context:context,budget:budget)
        try expect(!unavailableOverBudget.mayAttemptLoad,"Unavailable and over-budget report")
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("toklet-swift-compatibility-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:false)
        defer { try? FileManager.default.removeItem(at:temporary) }
        try Data("{\"model_type\":\"future_graph\"}".utf8).write(to:temporary.appendingPathComponent("config.json"))
        let unknown = try NativeCompatibility.inspect(directory:temporary.path,context:context,budget:budget)
        try expect(unknown.status == .unknown && !unknown.mayAttemptLoad,"Unknown graph")
        let sourceConfig = root.appendingPathComponent("tests/fixtures/mlx-qwen3-v1/tied/config.json")
        var unsupported = try JSONSerialization.jsonObject(with:Data(contentsOf:sourceConfig)) as! [String:Any]
        unsupported["rope_theta"] = 12345
        try JSONSerialization.data(withJSONObject:unsupported).write(to:temporary.appendingPathComponent("config.json"))
        try FileManager.default.copyItem(at:root.appendingPathComponent("tests/fixtures/mlx-qwen3-v1/tied/model.safetensors"),to:temporary.appendingPathComponent("model.safetensors"))
        let rejected = try NativeCompatibility.inspect(directory:temporary.path,context:context,budget:budget)
        try expect(rejected.reason == .unsupportedGeometry && !rejected.mayAttemptLoad,"Unsupported geometry display")
        try Data("{\"model_type\":\"future_graph\"}".utf8).write(to:temporary.appendingPathComponent("config.json"))
        // No GPU is created for rejected metadata or lifecycle-superseded loads.
        for cleanup in ["suspend","unload","close"] {
            let gate = Gate()
            let worker = try NativeWorker(beforeCommandDequeue:{gate.wait($0)})
            defer { gate.proceed.signal();worker.close();_ = worker.waitUntilClosed(seconds:30) }
            var settings = NativeSettings(directory:temporary.path);settings.metal = true;settings.mlx = true
            try expect(worker.submit(.load(settings)),"Unknown load submission")
            guard let initial = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"CompatibilityChecks",code:2) }
            try expect(initial.status == TOKLET_UNSUPPORTED && initial.compatibility?.status == .unknown && !initial.hasModel,"Rejected load summary")
            var invalid = settings;invalid.capacity = 0
            try expect(worker.submit(.load(invalid)),"Invalid load submission")
            guard let cleared = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"CompatibilityChecks",code:2) }
            try expect(cleared.status == TOKLET_INVALID && cleared.compatibility == nil && cleared.compatibilityDirectory.isEmpty,"Invalid attempt republished old report")
            try expect(worker.submit(.load(settings)),"Seed summary submission")
            guard let seeded = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"CompatibilityChecks",code:2) }
            try expect(seeded.compatibility?.status == .unknown,"Missing seed report")
            gate.arm()
            try expect(worker.submit(.load(settings)),"Queued load")
            try expect(gate.reached.wait(timeout:.now()+30) == .success,"Dequeue gate")
            try expect(worker.snapshot().compatibility == nil,"Queued selection retained old report")
            if cleanup == "close" { worker.close() } else { worker.interrupt(unload:cleanup == "unload") }
            gate.proceed.signal()
            if cleanup == "close" { try expect(worker.waitUntilClosed(seconds:30),"Close deadline") }
            else {
                guard let idle = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"CompatibilityChecks",code:2) }
                try expect(idle.compatibility == nil && idle.compatibilityDirectory.isEmpty && !idle.hasModel && !idle.hasSession,"Cleanup republished a stale inspection before close")
                worker.close();try expect(worker.waitUntilClosed(seconds:30),"Close after cleanup")
            }
            let final = worker.snapshot()
            try expect(final.compatibility == nil && final.compatibilityDirectory.isEmpty && !final.hasModel && !final.hasSession,"Stale cleanup report or owner")
        }
        // The immutable real candidate is metadata-compatible, never chat-qualified.
        let real = try NativeCompatibility.inspect(directory:root.appendingPathComponent("models/MLX-Qwen3-0.6B-8bit-11de968").path,context:512,budget:1536*1024*1024)
        try expect(real.resources?.sourcePayloadBytes == 633364480 && !real.qualificationEvaluated && !real.textProfileChecked,"Real metadata scope")
        print("{\"passed\":true,\"compiledMetal\":\(metal),\"wireMutationsRejected\":\(mutations.count),\"queuedCleanupCases\":3,\"physicalDeviceQualified\":false}")
    }
}

import Foundation
import CToklet

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain:"ResourcePlanChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private final class ResourceGate: @unchecked Sendable {
    let reached = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
    private let lock = NSLock(); private var armed = false
    func arm() { lock.lock(); armed = true; lock.unlock() }
    func enter() {
        lock.lock(); let wait = armed; armed = false; lock.unlock()
        if wait { reached.signal(); precondition(proceed.wait(timeout:.now()+30) == .success) }
    }
}
@main
struct ResourcePlanMain {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NSError(domain:"ResourcePlanChecks",code:2) }
        let root = URL(fileURLWithPath:CommandLine.arguments[1]), metal = CommandLine.arguments[2] == "metal"
        let fixture = root.appendingPathComponent("tests/fixtures/mlx-qwen3-v1/tied")
        let bridge = UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES), whole: UInt64 = 128 * 1024 * 1024
        let invalidBudgets: [UInt64] = [bridge-1,bridge,6*1024*1024*1024+1,UInt64.max]
        for budget in invalidBudgets {
            var rejected = false
            do { _ = try NativeResourcePlan.numericalBudget(wholeBudget:budget) } catch { rejected = true }
            try expect(rejected,"Bad whole budget accepted")
        }
        let singleByte = try NativeResourcePlan.numericalBudget(wholeBudget:bridge+1)
        try expect(singleByte == 1,"Budget underflow or hidden debit")
        let budget = try NativeResourcePlan.numericalBudget(wholeBudget:whole)
        var original: [String:Any] = [
            "schema_version":1,"device_name":"Independent wire witness","maximum_buffer_bytes":4096,
            "requested_load_context":32,"requested_capacity":32,"requested_chunk_size":3,"requested_position_offset":0,
            "requested_load_budget_bytes":3000,"requested_session_budget_bytes":3000,
            "projection":"MlxAffine","normalization":"Compensated","readback":"FullLogits","attention":"Reference",
            "source_payload_bytes":1000,"source_admitted_load_bytes":2000,"resident_weights_bytes":900,"resident_admitted_load_bytes":1800,
            "session_memory":["weights_bytes":900,"kv_bytes":100,"scratch_bytes":100,"history_bytes":128,"logits_bytes":512,"metadata_bytes":24,"metadata_allowance_bytes":16,"admitted_bytes":1780],
            "fits_source_load_budget":true,"fits_resident_load_budget":true,"fits_session_budget":true,"fits_requested_budgets":true,
            "numerical_buffers_allocated":false,"payload_validation_required":true,"qualification_evaluated":false]
        func decode(_ value: [String:Any], budget: UInt64 = 3000) throws -> NativeResourcePlan {
            try NativeResourcePlan.decode(JSONSerialization.data(withJSONObject:value),capacity:32,chunk:3,budget:budget)
        }
        let legacy = try decode(original)
        try expect(legacy.positionalResidentBytes == nil && legacy.sessionMemory.positionalBytes == nil,"Legacy zero fields changed")
        var scaled = original, scaledMemory = original["session_memory"] as! [String:Any]
        scaled["positional_resident_bytes"] = UInt64(120)
        scaledMemory["positional_bytes"] = UInt64(120)
        scaledMemory["admitted_bytes"] = UInt64(1900)
        scaled["session_memory"] = scaledMemory
        let acceptedScaled = try decode(scaled)
        try expect(acceptedScaled.positionalResidentBytes == 120 && acceptedScaled.sessionMemory.sum == 1900,"Positional owner missing from sum")
        for (top,nested,admitted,residentLoad) in [
            (UInt64(120),UInt64(121),UInt64(1901),UInt64(1800)),
            (UInt64(121),UInt64(120),UInt64(1900),UInt64(1800)),
            (UInt64(1200),UInt64(1200),UInt64(2980),UInt64(1800)),
            (UInt64.max,UInt64.max,UInt64.max,UInt64.max),
        ] {
            var changed = scaled, memory = scaledMemory
            changed["positional_resident_bytes"] = top
            changed["resident_admitted_load_bytes"] = residentLoad
            memory["positional_bytes"] = nested; memory["admitted_bytes"] = admitted
            changed["session_memory"] = memory
            var rejected = false
            do { _ = try decode(changed) } catch { rejected = true }
            try expect(rejected,"Contradictory/overflowing positional accounting accepted")
        }
        var missing = scaled
        missing["session_memory"] = original["session_memory"]
        var missingRejected = false
        do { _ = try decode(missing) } catch { missingRejected = true }
        try expect(missingRejected,"Omitted nested positional owner accepted")
        let mutations: [(String,Any)] = [("schema_version",2),("device_name",""),("maximum_buffer_bytes",0),
            ("requested_load_context",33),("requested_capacity",33),("requested_chunk_size",4),("requested_position_offset",1),
            ("requested_load_budget_bytes",1),("requested_session_budget_bytes",1),("projection","HalfCompensated"),
            ("normalization","Original"),("readback","GreedyToken"),("attention","FutureAttention"),
            ("numerical_buffers_allocated",true),("payload_validation_required",false),("qualification_evaluated",true),
            ("resident_weights_bytes",901),("source_payload_bytes",2001),("resident_admitted_load_bytes",899),
            ("fits_source_load_budget",false),("fits_resident_load_budget",false),("fits_session_budget",false),("fits_requested_budgets",false)]
        for (key,value) in mutations {
            var changed = original; changed[key] = value; var rejected = false
            do { _ = try decode(changed) } catch { rejected = true }
            try expect(rejected,"Contradictory runtime report accepted: " + key)
        }
        for (key,value) in [("admitted_bytes",UInt64(1781)),("scratch_bytes",UInt64.max)] {
            var changed = original, memory = original["session_memory"] as! [String:Any]
            memory[key] = value; changed["session_memory"] = memory; var rejected = false
            do { _ = try decode(changed) } catch { rejected = true }
            try expect(rejected,"Invalid/overflowing byte sum accepted")
        }
        for data in [Data(),Data("{".utf8),Data(repeating:0,count:65537)] {
            var rejected = false
            do { _ = try NativeResourcePlan.decode(data,capacity:32,chunk:3,budget:3000) } catch { rejected = true }
            try expect(rejected,"Malformed runtime report accepted")
        }
        original["requested_load_budget_bytes"] = 999; original["requested_session_budget_bytes"] = 999
        for key in ["fits_source_load_budget","fits_resident_load_budget","fits_session_budget","fits_requested_budgets"] { original[key] = false }
        let lowWire = try decode(original,budget:999)
        try expect(!lowWire.fitsRequestedBudgets,"Consistent over-budget report rejected")
        if !metal {
            var rejected = false
            do { _ = try NativeResourcePlan.inspect(directory:fixture.path,capacity:32,chunk:3,budget:budget) }
            catch { rejected = (error as NSError).code == Int(TOKLET_UNSUPPORTED) }
            try expect(rejected,"Unavailable runtime backend silently fell back")
        } else {
            let plan = try NativeResourcePlan.inspect(directory:fixture.path,capacity:32,chunk:32,budget:budget)
            try expect(plan.fitsRequestedBudgets && plan.attention == .reference,"Real engine native policy/fit")
            var settings = NativeSettings(directory:fixture.path); settings.metal = true; settings.mlx = true; settings.chunk = 32
            let worker = try NativeWorker()
            defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
            try expect(worker.submit(.load(settings)),"Successful load submit")
            guard let ready = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:3) }
            try expect(ready.status == TOKLET_OK && ready.hasModel && ready.hasSession && ready.runtimePlan?.fitsRequestedBudgets == true,"Missing accepted runtime plan")
            try expect(ready.runtimePlan!.sessionMemory.admittedBytes + bridge == ready.admittedBytes,"Plan differs from actual native factory admission")
            try expect(worker.submit(.suspend),"Suspend")
            guard let suspended = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:4) }
            try expect(suspended.runtimePlan == nil && suspended.compatibility == nil && suspended.hasModel && !suspended.hasSession,"Suspend retained stale report")
            try expect(worker.submit(.unload),"Unload")
            guard let unloaded = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:5) }
            try expect(!unloaded.hasModel && unloaded.runtimePlan == nil,"Unload retained report")
            var low = settings; low.numericalBudget = plan.sessionMemory.admittedBytes + bridge - 1
            try expect(worker.submit(.load(low)),"Over-budget load")
            guard let denied = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:6) }
            try expect(denied.status == TOKLET_LOAD_ERROR && !denied.hasModel && !denied.hasSession && denied.runtimePlan?.fitsSessionBudget == false,"Runtime budget rejection created owners")
            try expect(denied.compatibility?.mayAttemptLoad == true,"Metadata did not reach the runtime budget witness")
            var invalid = settings; invalid.capacity = 0
            try expect(worker.submit(.load(invalid)),"Invalid load")
            guard let rejected = worker.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:7) }
            try expect(rejected.runtimePlan == nil && rejected.compatibility == nil,"Rejected new attempt republished old plan")
            for cleanup in ["suspend","unload","close"] {
                let gate = ResourceGate()
                let queued = try NativeWorker(afterRuntimeInspection:{_ in gate.enter()})
                defer { gate.proceed.signal(); queued.close(); _ = queued.waitUntilClosed(seconds:30) }
                try expect(queued.submit(.load(low)),"Seed denied plan")
                guard let seeded = queued.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:8) }
                try expect(seeded.runtimePlan?.fitsRequestedBudgets == false,"Missing seed plan")
                gate.arm(); try expect(queued.submit(.load(settings)),"Queued load")
                try expect(gate.reached.wait(timeout:.now()+30) == .success,"Runtime gate")
                try expect(queued.snapshot().runtimePlan == nil,"New attempt kept old plan")
                if cleanup == "close" { queued.close() } else { queued.interrupt(unload:cleanup == "unload") }
                gate.proceed.signal()
                if cleanup == "close" {
                    try expect(queued.waitUntilClosed(seconds:30) && queued.snapshot().runtimePlan == nil,"Close kept runtime plan")
                } else {
                    guard let idle = queued.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:9) }
                    // Inspect before Close can conceal a stale post-load publication.
                    try expect(!idle.busy && !idle.hasModel && !idle.hasSession && idle.runtimePlan == nil && idle.compatibility == nil,"Queued cleanup republished runtime plan")
                }
            }
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("toklet-resource-worker-" + UUID().uuidString)
            try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:false)
            defer { try? FileManager.default.removeItem(at:temporary) }
            for name in ["config.json","model.safetensors"] { try FileManager.default.copyItem(at:fixture.appendingPathComponent(name),to:temporary.appendingPathComponent(name)) }
            let changed = try NativeWorker(afterRuntimeInspection:{_ in try? Data("{}".utf8).write(to:temporary.appendingPathComponent("config.json"))})
            defer { changed.close(); _ = changed.waitUntilClosed(seconds:30) }
            var changedSettings = settings; changedSettings.directory = temporary.path
            try expect(changed.submit(.load(changedSettings)),"Changed source load")
            guard let failed = changed.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:10) }
            try expect(failed.status == TOKLET_LOAD_ERROR && !failed.hasModel && !failed.hasSession && failed.runtimePlan?.fitsRequestedBudgets == true,"Actual loader trusted a stale plan")
            try expect(changed.submit(.suspend),"Failure cleanup")
            guard let cleared = changed.waitUntilIdle(seconds:30) else { throw NSError(domain:"ResourcePlanChecks",code:11) }
            try expect(cleared.runtimePlan == nil && cleared.compatibility == nil,"Failure cleanup retained plan")
        }
        let output: [String:Any] = ["passed":true,"compiledMetal":metal,"wireMutationsRejected":25,"queuedCleanupCases":metal ? 3:0,"sourceRevalidationAfterPlan":metal,"physicalDeviceQualified":false]
        print(String(decoding:try JSONSerialization.data(withJSONObject:output,options:[.sortedKeys]),as:UTF8.self))
    }
}

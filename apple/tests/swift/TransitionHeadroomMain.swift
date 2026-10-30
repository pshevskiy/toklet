import Foundation
import CToklet

private func require(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw NativeContextFailure(message) }
}
private func rejects(_ operation: () throws -> Void) throws {
    do { try operation() } catch { return }
    throw NativeContextFailure("Invalid headroom policy accepted")
}
private final class HeadroomReader: @unchecked Sendable {
    private let lock = NSLock()
    private var value: UInt64 = .max
    private var instrument = "Controlled test headroom"
    private var fail = false
    private var block = false
    private var count = 0
    let reached = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
    func set(_ available: UInt64, instrument: String = "Controlled test headroom", fail: Bool = false,
             block: Bool = false) {
        lock.lock(); defer { lock.unlock() }
        value = available; self.instrument = instrument; self.fail = fail; self.block = block
    }
    var calls: Int { lock.lock(); defer { lock.unlock() }; return count }
    func read() throws -> (availableBytes: UInt64, instrument: String) {
        lock.lock()
        count += 1
        let available = value, name = instrument, shouldFail = fail, shouldBlock = block
        block = false
        lock.unlock()
        if shouldBlock {
            reached.signal()
            guard proceed.wait(timeout:.now()+30) == .success else { throw NativeContextFailure("Reader release deadline") }
        }
        if shouldFail { throw NativeContextFailure("Injected headroom query failure") }
        return (available,name)
    }
}

@main
struct TransitionHeadroomMain {
    static func idle(_ worker: NativeWorker) throws -> NativeSnapshot {
        guard let state = worker.waitUntilIdle(seconds:60) else { throw NativeContextFailure("Headroom worker deadline") }
        return state
    }
    static func same(_ before: NativeSnapshot, _ after: NativeSnapshot) throws {
        var lhs = before, rhs = after
        lhs.revision = 0; rhs.revision = 0; lhs.transition = nil; rhs.transition = nil
        try require(String(reflecting:lhs) == String(reflecting:rhs),"Rejected memory check changed old state/configuration")
    }
    static func wanted(_ state: NativeSnapshot, budget: UInt64 = 512*1024*1024,
                       policy: NativeTransitionMemoryPolicy = .friendly) throws -> NativeTransitionRequest {
        guard let context = state.liveContext else { throw NativeContextFailure("No current session") }
        var request = NativeTransitionRequest(id:UUID(),
            selection:NativeTransitionSelection(id:UUID(),modelInstance:context.modelInstance,sessionInstance:context.sessionInstance),
            capacity:16,chunk:3,wholeSessionBudget:128*1024*1024,coexistenceBudget:budget,
            replayReserve:0,textReserve:0,applicationReserve:NativeTransitionHeadroomCheck.applicationReserve)
        request.memoryPolicy = policy
        return request
    }
    static func enqueue(_ worker: NativeWorker, _ request: NativeTransitionRequest) throws {
        try require(worker.selectTransition(request.selection),"Headroom selection refused")
        try require(worker.submit(.replaceWithEmptySession(request)),"Headroom request refused")
    }
    static func emit(_ label: String, _ state: NativeSnapshot, calls: Int) throws {
        var record: [String:Any] = ["event":label,"calls":calls,"status":state.status,
            "has_session":state.hasSession,"history":state.history,"committed":state.evaluated,"pending":state.pending,
            "ids":state.ids,"saved_prefix":state.savedPrefix as Any? ?? NSNull()]
        if let configuration = state.sessionConfiguration {
            record["configuration"] = ["model":configuration.modelInstance,"session":configuration.sessionInstance,
                "capacity":configuration.capacity,"chunk":configuration.chunk,"whole_budget":configuration.wholeSessionBudget,
                "accepted_request":configuration.acceptedRequestID?.uuidString as Any? ?? NSNull()]
        }
        if let result = state.transition {
            record["outcome"] = result.outcome.rawValue; record["committed_swap"] = result.committed
            record["result_status"] = result.status
            record["message"] = result.message
            if let check = result.headroom {
                record["required"] = check.requiredBytes; record["available"] = check.availableBytes
                record["instrument"] = check.instrument; record["fits"] = check.fits
            }
        }
        let data = try JSONSerialization.data(withJSONObject:record,options:[.sortedKeys])
        try require(data.count <= 65536,"Oversized headroom record")
        FileHandle.standardOutput.write(Data("HEADROOM_RECORD ".utf8)+data+Data([10]))
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3, ["cpu","metal"].contains(CommandLine.arguments[2]) else {
            throw NativeContextFailure("Expected root and cpu/metal")
        }
        let root = URL(fileURLWithPath:CommandLine.arguments[1]), metal = CommandLine.arguments[2] == "metal"
        let reader = HeadroomReader()
        let worker = try NativeWorker(captureSavedPrefix:true,readTransitionHeadroom:{ try reader.read() })
        defer { reader.proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        var settings = NativeSettings(directory:root.appendingPathComponent("artifacts/m7-native-tiny-v2").path)
        if metal {
            settings.metal = true; settings.mlx = true
            settings.directory = root.appendingPathComponent("tests/fixtures/mlx-qwen3-v1/tied").path
        }
        try require(worker.submit(.load(settings)),"Headroom fixture load refused")
        var state = try idle(worker)
        try require(state.status == 0 && state.hasSession,"Headroom fixture load failed: \(state.message)")
        let validRequest = try wanted(state)
        var invalid = NativeTransitionRequest(id:validRequest.id,selection:validRequest.selection,
            capacity:validRequest.capacity,chunk:validRequest.chunk,wholeSessionBudget:validRequest.wholeSessionBudget,
            coexistenceBudget:validRequest.coexistenceBudget,replayReserve:0,textReserve:0,
            applicationReserve:NativeTransitionHeadroomCheck.applicationReserve-1)
        invalid.memoryPolicy = .friendly
        try rejects { try invalid.validate() }
        try rejects {
            _ = try NativeTransitionHeadroomCheck.requiredIncrementalBytes(candidateOwned:.max,candidateAllowance:1,bridge:1,json:1)
        }

        // Diagnostic requests retain their original behavior and never query
        // headroom, even when the injected reader would fail.
        reader.set(0,fail:true)
        try enqueue(worker,try wanted(state,policy:.diagnostic))
        state = try idle(worker)
        try emit("diagnostic-no-query",state,calls:reader.calls)
        try require(state.transition?.committed == true && reader.calls == 0,"Diagnostic path queried process headroom")
        guard let plan = state.transition?.plan else { throw NativeContextFailure("Diagnostic plan unavailable") }
        var required: UInt64 = 0
        for part in [plan.candidatePlanned.totalBytes,plan.reserves.candidateMetadataDriverBytes,
                     plan.reserves.candidateBridgeBytes,plan.reserves.jsonSnapshotBytes,UInt64(16*1024*1024)] {
            let next = required.addingReportingOverflow(part)
            try require(!next.overflow,"Independent expected requirement overflow"); required = next.partialValue
        }
        try require(required > 0,"Empty incremental requirement")
        try require(worker.submit(.generate(operation:0,tokens:[1,7,13,3],maximum:2,stops:[])),"Old prefix enqueue")
        state = try idle(worker)
        try require(state.status == 0 && state.history == 6 && state.pending == 1,"Old pending witness failed")
        let before = state

        // Coexistence is a separate admission decision and must reject before
        // consulting a favorable process-headroom observation.
        reader.set(.max)
        try enqueue(worker,try wanted(before,budget:1))
        state = try idle(worker)
        try emit("aggregate-rejects-before-query",state,calls:reader.calls)
        try same(before,state)
        try require(reader.calls == 0 && state.transition?.committed == false,"Failed aggregate queried headroom")

        for available in [UInt64(0),required-1] {
            let calls = reader.calls; reader.set(available)
            try enqueue(worker,try wanted(before))
            state = try idle(worker)
            try emit(available == 0 ? "zero-headroom":"one-byte-under",state,calls:reader.calls)
            try same(before,state)
            try require(reader.calls == calls+1 && state.transition?.headroom?.requiredBytes == required
                && state.transition?.headroom?.availableBytes == available && state.transition?.headroom?.fits == false,
                "Insufficient headroom was not rejected exactly")
        }
        for (name,fail) in [("",false),(String(repeating:"x",count:513),false),("query-failure",true)] {
            let calls = reader.calls; reader.set(.max,instrument:name,fail:fail)
            try enqueue(worker,try wanted(before))
            state = try idle(worker)
            try emit(fail ? "query-error-thrown" : (name.isEmpty ? "query-error-empty":"query-error-too-long"),state,calls:reader.calls)
            try same(before,state)
            try require(reader.calls == calls+1 && state.transition?.committed == false
                && state.transition?.headroom == nil,"Invalid observation admitted a candidate")
        }

        // The query is deliberately paused outside the swap lock. Cancellation
        // and selection changes still invalidate the request after this point.
        for cancel in [true,false] {
            reader.set(.max,block:true)
            try enqueue(worker,try wanted(before))
            try require(reader.reached.wait(timeout:.now()+30) == .success,"Headroom reader barrier deadline")
            if cancel { worker.cancel() }
            else {
                guard let facts = before.liveContext else { throw NativeContextFailure("Lost old identity") }
                _ = worker.selectTransition(NativeTransitionSelection(id:UUID(),modelInstance:facts.modelInstance,sessionInstance:facts.sessionInstance))
            }
            reader.proceed.signal(); state = try idle(worker)
            try emit(cancel ? "cancel-during-query":"selection-during-query",state,calls:reader.calls)
            try same(before,state)
            try require(state.transition == nil,"Stale headroom result published")
        }
        reader.set(required)
        let exact = try wanted(before), calls = reader.calls
        try enqueue(worker,exact)
        state = try idle(worker)
        try emit("exact-headroom-adopts",state,calls:reader.calls)
        try require(reader.calls == calls+1 && state.transition?.committed == true
            && state.transition?.headroom?.requiredBytes == required && state.transition?.headroom?.availableBytes == required
            && state.transition?.headroom?.fits == true && state.sessionConfiguration?.acceptedRequestID == exact.id,
            "Exact incremental headroom did not admit")
        try require(worker.submit(.generate(operation:3,tokens:[1],maximum:1,stops:[])),"Accepted session generation refused")
        state = try idle(worker)
        try require(state.status == 0 && state.ids.count == 1,"Accepted session unusable")
        worker.close(); try require(worker.waitUntilClosed(seconds:30),"Headroom worker close deadline")
        let summary: [String:Any] = ["passed":true,"backend":metal ? "metal":"cpu","records":10,
            "diagnostic_no_query":true,"aggregate_before_query":true,"exact_and_under":true,
            "query_errors":3,"stale_query_barriers":2,"closed":true]
        let data = try JSONSerialization.data(withJSONObject:summary,options:[.sortedKeys])
        FileHandle.standardOutput.write(Data("HEADROOM_CHECKS ".utf8)+data+Data([10]))
    }
}

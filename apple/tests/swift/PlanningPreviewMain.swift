import Foundation
import CToklet

private func check(_ ok: @autoclosure () -> Bool, _ message: String) throws {
    if !ok() { throw NativeContextFailure(message) }
}
private func idle(_ worker: NativeWorker) throws -> NativeSnapshot {
    guard let value = worker.waitUntilIdle(seconds:60) else { throw NativeContextFailure("Preview worker deadline") }
    return value
}
private func sameActiveState(_ before: NativeSnapshot, _ after: NativeSnapshot) throws {
    // Compare every snapshot field, including metrics and owner identities, not
    // a whitelist which silently misses a newly added field. Only these two
    // explicitly independent fields may change at a completed preview boundary.
    var lhs = before, rhs = after
    lhs.revision = 0; rhs.revision = 0; lhs.preview = nil; rhs.preview = nil
    try check(String(reflecting:lhs) == String(reflecting:rhs),"Preview changed active snapshot")
}
private final class PreviewGate: @unchecked Sendable {
    let reached = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
    private let lock = NSLock()
    private var limitsCount = 0, publicationCount = 0
    private var last: NativePreviewResult.Outcome?
    private var armed = false
    func arm() { lock.lock(); armed = true; lock.unlock() }
    func pauseIfArmed() {
        lock.lock(); let wait = armed; armed = false; lock.unlock()
        if wait { pause() }
    }
    func limits() { lock.lock(); limitsCount += 1; lock.unlock() }
    func publication(_ result: NativePreviewResult) { lock.lock(); publicationCount += 1; last = result.outcome; lock.unlock() }
    func counts() -> (Int,Int,NativePreviewResult.Outcome?) {
        lock.lock(); defer { lock.unlock() }; return (limitsCount,publicationCount,last)
    }
    func pause() { reached.signal(); precondition(proceed.wait(timeout:.now()+30) == .success) }
    func wait() throws { try check(reached.wait(timeout:.now()+30) == .success,"Preview gate deadline") }
}
private func request(_ selection: NativePreviewSelection, upper: UInt32 = 32, chunk: UInt32 = 3,
                     offset: UInt32 = 0, ceiling: UInt32? = nil, load: UInt64 = 128*1024*1024,
                     session: UInt64 = 128*1024*1024) -> NativePreviewRequest {
    NativePreviewRequest(id:UUID(),selection:selection,upper:upper,chunk:chunk,offset:offset,
                         contextCeiling:ceiling,loadBudget:load,sessionBudget:session)
}
private func reject(_ body: () throws -> Void) throws {
    do { try body() } catch { return }
    throw NativeContextFailure("Invalid derived request accepted")
}

@main
struct PlanningPreviewMain {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected root and cpu/metal") }
        let root = URL(fileURLWithPath:CommandLine.arguments[1]), metal = CommandLine.arguments[2] == "metal"
        let fixture = root.appendingPathComponent("tests/fixtures/mlx-qwen3-v1/tied").path
        let selection = NativePreviewSelection(id:UUID(),directory:fixture,profile:.ordinary)
        let bounded = request(selection,upper:32,offset:5,ceiling:20)
        let derived = try bounded.capacityRequest(planningUpper:32)
        try check(derived.context == 20 && derived.upper == 15 && derived.offset == 5 && derived.chunk == 3,"Incorrect absolute/relative derivation")
        try check(tryRequest(request(selection,upper:7,offset:5),32)?.upper == 7,"Caller upper was widened")
        var explicit = bounded; explicit.normalization = .original; explicit.readback = .greedyToken; explicit.attention = .online
        let explicitDerived = try explicit.capacityRequest(planningUpper:32)
        try check(explicitDerived.normalization == .original && explicitDerived.readback == .greedyToken
                  && explicitDerived.attention == .online,"Derived modes changed")
        for bad in [request(selection,offset:32),request(selection,chunk:4,offset:29),
                    request(selection,offset:.max),request(selection,upper:0),request(selection,ceiling:0),
                    request(selection,load:0),request(selection,session:6*1024*1024*1024+1)] {
            try reject { _ = try bad.capacityRequest(planningUpper:32) }
        }
        try reject { _ = try bounded.capacityRequest(planningUpper:0) }

        let loadedGate = PreviewGate()
        let worker = try NativeWorker(captureSavedPrefix:true,afterPreviewLimits:{ _,_ in loadedGate.pauseIfArmed() })
        defer { loadedGate.proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        worker.selectPreview(selection)
        let initial = worker.snapshot(), first = request(selection)
        try check(worker.submit(.preview(first)),"Unloaded preview enqueue")
        var state = try idle(worker)
        try check(state.preview?.request == first,"Preview identity missing")
        try check(state.preview?.outcome == (metal ? .planned : .metadataOnly),"Backend preview outcome")
        try check(state.preview?.limitation == (metal ? nil : .backendUnavailable),"Backend limitation")
        try sameActiveState(initial,state)
        let saved = state.preview
        try check(!worker.selectPreview(NativePreviewSelection(id:UUID(),directory:String(repeating:"x",count:16385),profile:.ordinary)),
                  "Oversized selection retained")
        try check(!worker.submit(.preview(request(selection,upper:0))),"Malformed preview accepted")
        try check(worker.snapshot().preview?.request == saved?.request,"Rejected preview destroyed result")
        worker.selectPreview(NativePreviewSelection(id:UUID(),directory:fixture,profile:.ordinary))
        try check(worker.snapshot().preview == nil && !worker.submit(.preview(first)),"Same-path reselection kept old request")
        worker.selectPreview(selection)
        let replacement = request(selection,session:0)
        try check(worker.submit(.preview(replacement)),"Replacement rejected")
        state = try idle(worker)
        try check(state.preview?.request == replacement,"Replacement identity")
        if metal { try check(state.preview?.capacity?.capacity.selected == nil,"Zero budget selected capacity") }
        let audited = NativePreviewSelection(id:UUID(),directory:fixture,profile:.smolAudited)
        worker.selectPreview(audited)
        try check(worker.submit(.preview(request(audited))),"Audited metadata enqueue")
        state = try idle(worker)
        try check(state.preview?.limitation == .auditedProfileHasNoCapacityAPI && state.preview?.capacity == nil,
                  "Audited profile reached ordinary capacity API")
        let missing = NativePreviewSelection(id:UUID(),directory:fixture+"/missing",profile:.ordinary)
        worker.selectPreview(missing)
        try check(worker.submit(.preview(request(missing))),"Unsupported metadata enqueue")
        state = try idle(worker)
        try check(state.preview?.limitation == .metadataUnsupported && state.preview?.capacity == nil,"Unsupported metadata authorized planning")

        // Both queued and in-flight work are invalidated deterministically.
        var barriers = 0
        for stage in ["dequeue","limits","publication"] {
            for action in ["selection","cancel","suspend","unload","close"] {
                try stale(selection,stage:stage,action:action,bad:false,metal:metal)
                barriers += 1
            }
        }
        if metal {
            for action in ["selection","cancel","suspend","unload","close"] {
                try stale(selection,stage:"publication",action:action,bad:true,metal:true)
                barriers += 1
            }
        }

        if metal { try changedMetadata(fixture) }

        // Actual owners/generation evidence survive preview; the preview never
        // invokes Engine.perform or its common completion state replacement.
        var settings = NativeSettings(directory:root.appendingPathComponent("artifacts/m7-native-tiny-v2").path)
        settings.metal = metal
        if metal { settings.directory = fixture; settings.mlx = true }
        try check(worker.submit(.load(settings)),"Tiny load enqueue")
        state = try idle(worker)
        try check(state.status == 0 && state.hasSession && state.preview == nil,"Tiny load or preview invalidation (status \(state.status): \(state.message))")
        try check((state.runtimePlan != nil) == metal,"Actual MLX load plan witness missing")
        try check(worker.submit(.generate(operation:0,tokens:[1,7,13,3],maximum:2,stops:[])),"Generation enqueue")
        state = try idle(worker)
        try check(state.status == 0 && state.ids.count == 2 && state.tokenTimings.count == 2 && state.modelContext != nil,
                  "Missing actual generation witness")
        worker.selectPreview(selection)
        let generated = worker.snapshot()
        try check(worker.submit(.preview(request(selection))),"Loaded preview enqueue")
        state = try idle(worker);try sameActiveState(generated,state)
        try check(state.modelContext == generated.modelContext && state.liveContext == generated.liveContext,"Changed native owners")
        let beforeCancelledPreview = state
        loadedGate.arm()
        try check(worker.submit(.preview(request(selection))),"Loaded cancelled preview enqueue")
        try loadedGate.wait();worker.cancel();loadedGate.proceed.signal()
        state = try idle(worker);try sameActiveState(beforeCancelledPreview,state)
        try check(state.preview == nil,"Cancelled loaded preview published")
        barriers += 1
        try check(worker.submit(.generate(operation:0,tokens:[1,7,13,3],maximum:0,stops:[])),"Score prefix enqueue")
        try check(tryIdleOK(worker),"Score prefix failed")
        try check(worker.submit(.score(target:47)),"Score enqueue")
        state = try idle(worker);try check(state.status == 0 && state.score != nil,"Missing actual score witness")
        let scored = state
        try check(worker.submit(.preview(request(selection,offset:32))),"Empty-range preview enqueue")
        state = try idle(worker);try sameActiveState(scored,state)
        try check(state.preview?.outcome == (metal ? .failed : .metadataOnly),"Error/unsupported result missing")
        try check(worker.submit(.chat(id:UUID(),conversation:Data("[]".utf8),maximum:1)),"Rejected chat enqueue")
        state = try idle(worker);try check(state.status != 0 && state.chatResult?.rejected == true,"Missing rejected chat witness")
        let rejectedChat = state
        try check(worker.submit(.preview(request(selection))),"Preview after command failure")
        state = try idle(worker);try sameActiveState(rejectedChat,state)
        try check(state.status == rejectedChat.status && state.message == rejectedChat.message,"Preview replaced active error")
        try check(worker.submit(.reset),"Ordinary command enqueue")
        try check(worker.snapshot().preview == nil,"Ordinary command retained preview")
        _ = try idle(worker)
        worker.close();try check(worker.waitUntilClosed(seconds:30),"Worker close deadline")
        print("{\"passed\":true,\"compiledMetal\":\(metal),\"barrier_cases\":\(barriers),\"tiny_generation_witness\":true,\"preview_loads_model\":false,\"app_qualified\":false}")
    }

    private static func tryRequest(_ value: NativePreviewRequest, _ upper: UInt32) -> NativeCapacityPlan.Request? {
        try? value.capacityRequest(planningUpper:upper)
    }
    private static func tryIdleOK(_ worker: NativeWorker) -> Bool { (try? idle(worker).status) == 0 }

    private static func changedMetadata(_ fixture: String) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("toklet-preview-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        defer { try? FileManager.default.removeItem(at:directory) }
        for name in ["config.json","model.safetensors"] {
            try FileManager.default.copyItem(at:URL(fileURLWithPath:fixture).appendingPathComponent(name),to:directory.appendingPathComponent(name))
        }
        let gate = PreviewGate()
        let worker = try NativeWorker(afterPreviewLimits:{ _,_ in gate.pause() })
        defer { gate.proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let selection = NativePreviewSelection(id:UUID(),directory:directory.path,profile:.ordinary)
        worker.selectPreview(selection)
        try check(worker.submit(.preview(request(selection))),"Mutable metadata preview")
        try gate.wait()
        let config = directory.appendingPathComponent("config.json")
        var value = try JSONSerialization.jsonObject(with:Data(contentsOf:config)) as! [String:Any]
        value["max_position_embeddings"] = 8
        try JSONSerialization.data(withJSONObject:value).write(to:config)
        gate.proceed.signal()
        let state = try idle(worker)
        try check(state.preview?.limits?.factoryPositionUpper == 32 && state.preview?.outcome == .failed
                  && state.preview?.capacity == nil,"Two metadata calls were treated as one authenticated artifact")
        var settings = NativeSettings(directory:directory.path);settings.metal = true;settings.mlx = true
        try check(worker.submit(.load(settings)),"Changed metadata load attempt")
        let denied = try idle(worker)
        try check(denied.status != 0 && !denied.hasModel && !denied.hasSession && denied.preview == nil,
                  "Load reused preview without source revalidation")
    }

    private static func stale(_ selection: NativePreviewSelection, stage: String, action: String, bad: Bool, metal: Bool) throws {
        let gate = PreviewGate()
        let worker = try NativeWorker(beforeCommandDequeue:{ command in
            if case .preview = command, stage == "dequeue" { gate.pause() }
        },afterPreviewLimits:{ _,_ in
            gate.limits(); if stage == "limits" { gate.pause() }
        },beforePreviewPublication:{ result in
            gate.publication(result); if stage == "publication" { gate.pause() }
        })
        defer { gate.proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        worker.selectPreview(selection)
        let original = request(selection,offset:bad ? 32:0)
        try check(worker.submit(.preview(original)),"Stale case enqueue")
        try gate.wait()
        try check(!worker.submit(.preview(request(selection))) && !worker.submit(.reset),"Busy preview accepted replacement")
        switch action {
        case "selection": worker.selectPreview(NativePreviewSelection(id:UUID(),directory:selection.directory,profile:selection.profile))
        case "cancel": worker.cancel()
        case "suspend": worker.interrupt(unload:false)
        case "unload": worker.interrupt(unload:true)
        case "close": worker.close()
        default: preconditionFailure("Unknown test action")
        }
        try check(worker.snapshot().preview == nil,"Stale result not cleared immediately")
        gate.proceed.signal()
        if action == "close" { try check(worker.waitUntilClosed(seconds:30),"Stale close deadline") }
        else { _ = try idle(worker) }
        try check(worker.snapshot().preview == nil,"Stale success/error published")
        let counts = gate.counts()
        if stage == "dequeue" { try check(counts.0 == 0 && counts.1 == 0,"Superseded queued preview inspected") }
        if stage == "limits" { try check(counts.0 == 1 && counts.1 == 0,"Superseded limits reached capacity/publication") }
        if stage == "publication" {
            try check(counts.1 == 1 && counts.2 == (bad ? .failed : (metal ? .planned : .metadataOnly)),"Expected terminal gate missing")
        }
    }
}

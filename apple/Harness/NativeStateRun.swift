import Foundation
import CryptoKit

private func stateRunError(_ message: String) -> NSError {
    NSError(domain:"TokletStateRun",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}
struct NativeStateRow: Codable, Sendable {
    let name: String
    let state: NativeLifecycleState
    let scalarScore: NativeDistributionScore?
    let callbackCount: Int
    let admittedBytes: UInt64
    let completedUptimeNanoseconds: UInt64
    let footprint: NativeFootprint
    let environment: NativeEnvironment
}
struct NativeStateResult: Encodable, Sendable {
    let backend: String
    let prefillWitnessRequired: Bool
    var passed = false
    var error: String?
    var cleanupErrors: [String] = []
    var endpointErrors: [String] = []
    var filesBefore: [String:String] = [:]
    var filesAfter: [String:String] = [:]
    var checkpoints: [NativeMemoryCheckpoint] = []
    var rows: [NativeStateRow] = []
    var trace: NativeLifecycleTraceReport?
    var prefillWitness = false
    var closed = false
    var released = false
    let physicalDeviceQualified = false
    let m7Complete = false
}
// Cold diagnostic: bounded copies of exact history, one model and one session.
// A pending input tail after partial prefill can exceed the one sampled-token
// tail of ordinary decode. Every comparison uses its exact original prompt.
func runNativeState(_ config: NativeLifecycleConfig, root: URL, signals: NativeLifecycleSignals,
                    requirePrefillWitness: Bool = true) -> NativeStateResult {
    precondition(!Thread.isMainThread)
    var result = NativeStateResult(backend:nativeJobBackendName(metal:config.metal,mlx:config.mlx),prefillWitnessRequired:requirePrefillWitness)
    var owner: NativeWorker?;weak var weakOwner: NativeWorker?
    var trace: NativeLifecycleTrace?;var admitted = false
    func require(_ value: Bool, _ message: String) throws {
        if !value { throw stateRunError(message) }
    }
    func foreground() throws {
        try signals.check()
        let phase = NativeEnvironment.read().applicationState
        try require(phase == nil || phase == 0,"State diagnostic requires foreground")
        try require(trace?.report().events.isEmpty != false,"Unexpected lifecycle event during state diagnostic")
    }
    func memory(_ phase: String) throws {
        let value = try NativeFootprint.read();result.checkpoints.append(NativeMemoryCheckpoint(phase:phase,footprint:value))
        try require(value.physicalBytes <= config.processBudget && value.lifetimePeakPhysicalBytes <= config.processBudget,"State process trial budget exceeded")
    }
    func record(_ name: String, _ state: NativeSnapshot) throws {
        try require(result.rows.count < 64,"State row bound exceeded")
        let value = try NativeFootprint.read()
        result.rows.append(NativeStateRow(name:name,state:NativeLifecycleState(state),scalarScore:state.score,
            callbackCount:state.tokenTimings.count,admittedBytes:state.admittedBytes,
            completedUptimeNanoseconds:DispatchTime.now().uptimeNanoseconds,footprint:value,environment:NativeEnvironment.read()))
        try require(value.physicalBytes <= config.processBudget && value.lifetimePeakPhysicalBytes <= config.processBudget,"State row process cap exceeded")
        if state.hasModel { try require(state.admittedBytes > 0 && state.admittedBytes <= config.numericalBudget,"State native admission differs") }
    }
    func finish(_ name: String, status: UInt32 = 0) throws -> NativeSnapshot {
        guard let worker = owner, let state = worker.waitUntilIdle(seconds:300) else { throw stateRunError("State command deadline: " + name) }
        try foreground();try record(name,state)
        try require(!state.closed && state.status == status,"State command status differs: " + name)
        return state
    }
    func perform(_ command: NativeCommand, _ name: String, status: UInt32 = 0) throws -> NativeSnapshot {
        try foreground();try require(owner?.submit(command) == true,"State submission rejected: " + name)
        return try finish(name,status:status)
    }
    func prefix(_ state: NativeSnapshot, _ tokens: [UInt32], pending: UInt32, ids: [UInt32]) throws {
        try require(state.hasModel && state.hasSession && state.savedPrefix == tokens && state.savedHistory == tokens.count
            && state.history == UInt32(tokens.count) && state.evaluated + pending == state.history && state.pending == pending
            && state.ids == ids && state.tokenTimings.count == ids.count,"Exact state prefix/counts differ")
    }
    func scalar(_ name: String, target: UInt32) throws -> NativeDistributionScore {
        let state = try perform(.score(target:target),name)
        guard let score = state.score else { throw stateRunError("Missing scalar score") }
        try require(state.ids.isEmpty && state.tokenTimings.isEmpty && state.pending == 0 && score.evaluated == state.evaluated
            && score.vocabulary == config.vocabulary && score.target == target,"Scalar score state differs")
        return score
    }
    do {
        try config.validate();try require(config.maximum >= 3 && config.vocabulary >= 2,"State reference configuration differs")
        if requirePrefillWitness { try require(config.prompt.count == 512 && config.chunk == 128 && config.capacity == 1152,"Physical prefill witness requires the bounded 512-token workload") }
        try foreground();try memory("before-artifact-proof")
        result.filesBefore = try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
        admitted = true;try memory("after-artifact-proof")
        let configName = config.modelFolder + "/config.json"
        let handle = try FileHandle(forReadingFrom:try localURL(root,configName));defer { try? handle.close() }
        let bytes = try handle.read(upToCount:512 * 1024+1) ?? Data()
        struct Vocabulary: Decodable { let vocab_size: UInt32 }
        try require(bytes.count <= 512 * 1024 && SHA256.hash(data:bytes).map({ String(format:"%02x",$0) }).joined() == config.files[configName],"State model config bytes differ")
        try require(try JSONDecoder().decode(Vocabulary.self,from:bytes).vocab_size == config.vocabulary,"State vocabulary differs")
        owner = try NativeWorker(captureSavedPrefix:true);weakOwner = owner
        trace = try NativeLifecycleTrace(worker:owner!,maximumPrefix:Int(config.capacity));try signals.attach(owner!,trace:trace!)
        var settings = NativeSettings(directory:try localURL(root,config.modelFolder).path)
        settings.packedDirectory = config.packedFolder.isEmpty ? "":try localURL(root,config.packedFolder).path
        settings.metal = config.metal;settings.mlx = config.mlx == true;settings.capacity = config.capacity;settings.chunk = config.chunk;settings.numericalBudget = config.numericalBudget
        let loaded = try perform(.load(settings),"loaded");try prefix(loaded,[],pending:0,ids:[]);try memory("loaded")
        let primer = Array(config.prompt.prefix(16));let target = primer.last!
        let initial = try perform(.generate(operation:0,tokens:primer,maximum:0,stops:[]),"primer")
        try prefix(initial,primer,pending:0,ids:[])
        let firstScore = try scalar("primer-score",target:target)
        let cached = try perform(.generate(operation:0,tokens:primer,maximum:0,stops:[]),"fully-cached")
        try prefix(cached,primer,pending:0,ids:[])
        try require(try scalar("cached-score",target:target) == firstScore,"Fully cached scalar score differs")
        let pending = try perform(.generate(operation:2,tokens:[],maximum:3,stops:[]),"pending-primer")
        try require(pending.reason == 1 && pending.ids.count == 3,"Pending primer output differs")
        let pendingPrefix = primer+pending.ids;try prefix(pending,pendingPrefix,pending:1,ids:pending.ids)
        let append = Array(config.prompt.prefix(2));let appendedPrefix = pendingPrefix+append
        let appended = try perform(.generate(operation:1,tokens:append,maximum:2,stops:[]),"append-pending")
        try require(appended.reason == 1 && appended.ids.count == 2,"Append output differs")
        try prefix(appended,appendedPrefix+appended.ids,pending:1,ids:appended.ids)
        _ = try perform(.reset,"append-reference-reset")
        let appendReference = try perform(.generate(operation:0,tokens:appendedPrefix,maximum:2,stops:[]),"append-reference")
        try require(appendReference.reason == 1 && appendReference.ids == appended.ids,"Append exact-prefix fresh reference differs")
        try prefix(appendReference,appendedPrefix+appended.ids,pending:1,ids:appended.ids)
        var changed = primer;changed[0] = changed[0] == 0 ? 1:0
        let changedState = try perform(.generate(operation:0,tokens:changed,maximum:3,stops:[]),"changed-prefix")
        try require(changedState.reason == 1 && changedState.ids.count == 3,"Changed prefix output differs")
        try prefix(changedState,changed+changedState.ids,pending:1,ids:changedState.ids)
        _ = try perform(.reset,"changed-reference-reset")
        let changedReference = try perform(.generate(operation:0,tokens:changed,maximum:3,stops:[]),"changed-reference")
        try require(changedReference.reason == 1 && changedReference.ids == changedState.ids,"Changed-prefix fresh reference differs")
        try prefix(changedReference,changed+changedState.ids,pending:1,ids:changedState.ids)
        if requirePrefillWitness {
            _ = try perform(.reset,"prefill-reset");try foreground()
            try require(owner!.submit(.generate(operation:0,tokens:config.prompt,maximum:0,stops:[])),"Prefill submit rejected")
            // Trigger timing is not evidence of progress. The completed report
            // below must prove actual committed partial prefill and no emissions.
            Thread.sleep(forTimeInterval:0.05)
            let active = owner!.snapshot();try record("prefill-cancel-trigger",active)
            try require(active.busy && active.ids.isEmpty && active.tokenTimings.isEmpty,"No active prefill cancellation window")
            owner!.cancel();let cancelled = try finish("prefill-cancelled")
            try require(cancelled.reason == 3 && cancelled.evaluated > 0 && cancelled.evaluated < UInt32(config.prompt.count),"No committed partial-prefill witness")
            try prefix(cancelled,config.prompt,pending:UInt32(config.prompt.count)-cancelled.evaluated,ids:[])
            result.prefillWitness = true
            let committed = try perform(.generate(operation:2,tokens:[],maximum:0,stops:[]),"prefill-completed")
            try prefix(committed,config.prompt,pending:0,ids:[])
            let resumed = try perform(.generate(operation:2,tokens:[],maximum:3,stops:[]),"prefill-resumed")
            try require(resumed.reason == 1 && resumed.ids.count == 3,"Prefill resumed output differs")
            try prefix(resumed,config.prompt+resumed.ids,pending:1,ids:resumed.ids)
            _ = try perform(.reset,"prefill-reference-reset")
            let reference = try perform(.generate(operation:0,tokens:config.prompt,maximum:3,stops:[]),"prefill-reference")
            try require(reference.reason == 1 && reference.ids == resumed.ids,"Partial-prefill fresh reference differs")
            try prefix(reference,config.prompt+resumed.ids,pending:1,ids:resumed.ids)
        }
        _ = try perform(.reset,"capacity-reset")
        var full: [UInt32] = [];full.reserveCapacity(Int(config.capacity))
        for index in 0..<Int(config.capacity) { full.append(config.prompt[index % config.prompt.count]) }
        let atCapacity = try perform(.generate(operation:0,tokens:full,maximum:0,stops:[]),"capacity-committed")
        try prefix(atCapacity,full,pending:0,ids:[])
        // The admitted capacity is small and checked before constructing cap+1.
        let over = full+[target]
        let rejected: [(String,NativeCommand)] = [
            ("overflow-continue",.generate(operation:2,tokens:[],maximum:1,stops:[])),
            ("overflow-append",.generate(operation:1,tokens:[target],maximum:0,stops:[])),
            ("overflow-prompt",.generate(operation:0,tokens:over,maximum:0,stops:[]))]
        for (name,command) in rejected {
            let before = try scalar(name+"-before-score",target:target)
            let state = try perform(command,name,status:1)
            try prefix(state,full,pending:0,ids:[])
            try require(state.reason == 0,"Rejected capacity operation must report no new stop reason")
            let after = try scalar(name+"-after-score",target:target)
            try require(before == after,"Rejected capacity operation changed scalar score")
        }
        _ = try perform(.reset,"usable-reset")
        let fresh = try perform(.generate(operation:0,tokens:changed,maximum:3,stops:[]),"usable-fresh-reference")
        try require(fresh.reason == 1 && fresh.ids == changedReference.ids,"Post-rejection fresh reference differs")
        try prefix(fresh,changed+fresh.ids,pending:1,ids:fresh.ids)
    } catch { result.error = String(error.localizedDescription.prefix(512)) }
    if let worker = owner {
        worker.close();let done = worker.waitUntilClosed(seconds:300);let state = worker.snapshot()
        result.closed = done && state.closed && state.workerFinished && state.status == 0 && !state.hasModel && !state.hasSession
            && state.savedPrefix == nil && state.savedHistory == 0 && state.history == 0 && state.evaluated == 0 && state.pending == 0 && state.reason == 0
            && state.admittedBytes == 0 && state.ids.isEmpty && state.tokenTimings.isEmpty && state.message.isEmpty
        do { try record("closed",state) } catch { result.cleanupErrors.append(String(error.localizedDescription.prefix(512))) }
        if !result.closed { result.cleanupErrors.append("State close acknowledgement differs") }
    }
    signals.detach()
    if result.closed, let trace { do { try trace.detach() } catch { result.cleanupErrors.append(String(error.localizedDescription.prefix(512))) } }
    owner = nil
    let deadline = Date(timeIntervalSinceNow:5)
    while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
    result.released = weakOwner == nil
    if !result.released { result.cleanupErrors.append("State worker retained after close") }
    result.trace = trace?.report()
    let endpoints = checkNativeLifecycleEndpoints(admitted:admitted,memory:memory,reprove:{
        try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
    })
    result.filesAfter = endpoints.files;result.endpointErrors = endpoints.errors
    result.passed = result.error == nil && result.cleanupErrors.isEmpty && result.endpointErrors.isEmpty && result.closed && result.released
        && result.filesBefore == config.files && result.filesAfter == config.files && result.trace?.events.isEmpty == true
        && result.trace?.error == nil && result.trace?.activeCommand == nil && result.trace?.detachedAfterClose == true
        && (!requirePrefillWitness || result.prefillWitness)
    return result
}

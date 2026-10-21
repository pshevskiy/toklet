import Foundation
import CryptoKit
private func lifecycleRunError(_ message: String) -> NSError {
    NSError(domain:"TokletLifecycleRun",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}
struct NativeLifecycleConfig: Codable, Sendable {
    let version: Int
    let modelFolder: String
    let packedFolder: String
    let files: [String:String]
    let metal: Bool
    let capacity: UInt32
    let chunk: UInt32
    let numericalBudget: UInt64
    let processBudget: UInt64
    let prompt: [UInt32]
    let maximum: UInt32
    let vocabulary: UInt32
    func validate() throws {
        guard version == 1, (1...1152).contains(capacity), chunk > 0, chunk <= min(128,capacity),
              (2...128).contains(maximum), (1...1024).contains(prompt.count), vocabulary > 0,
              prompt.allSatisfy({ $0 < vocabulary }), prompt.count + Int(maximum) + 3 <= Int(capacity),
              numericalBudget > 4096, numericalBudget < processBudget, processBudget <= 4 * 1024 * 1024 * 1024 else {
            throw lifecycleRunError("Invalid explicit lifecycle config")
        }
        try validateNativeArtifactFields(modelFolder:modelFolder,packedFolder:packedFolder,files:files,metal:metal)
    }
}
func admitLifecycleConfig(root: URL, name: String, sha256: String) throws -> NativeLifecycleConfig {
    guard sha256.count == 64, sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
          name == "M7Jobs/" + sha256 + ".json" else { throw lifecycleRunError("Canonical lifecycle job required") }
    let url = try localURL(root,name)
    let properties = try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey])
    guard properties.isRegularFile == true, properties.isSymbolicLink != true else { throw lifecycleRunError("Regular lifecycle config required") }
    let handle = try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    let data = try handle.read(upToCount:512 * 1024+1) ?? Data()
    guard data.count <= 512 * 1024, SHA256.hash(data:data).map({ String(format:"%02x",$0) }).joined() == sha256 else {
        throw lifecycleRunError("Lifecycle config consumed bytes differ")
    }
    let fields: Set<String> = ["version","modelFolder","packedFolder","files","metal","capacity","chunk","numericalBudget","processBudget","prompt","maximum","vocabulary"]
    guard let object = try JSONSerialization.jsonObject(with:data) as? [String:Any], Set(object.keys) == fields else {
        throw lifecycleRunError("Exact lifecycle fields required")
    }
    let config = try JSONDecoder().decode(NativeLifecycleConfig.self,from:data);try config.validate();return config
}
struct NativeLifecycleReady: Codable, Sendable {
    let eventIndex: Int
    let submittedUptimeNanoseconds: UInt64
    let observedUptimeNanoseconds: UInt64
    let visibleIDs: [UInt32]
    let applicationState: Int?
}
func requireNativeLifecycleReady(_ ready: NativeLifecycleReady, trip: NativeLifecycleRoundTrip,
                                 maximum: UInt32, requireSystem: Bool) throws {
    guard (2...128).contains(maximum), ready.eventIndex >= 0, ready.eventIndex < trip.inactive.index,
          ready.submittedUptimeNanoseconds > 0, ready.submittedUptimeNanoseconds <= ready.observedUptimeNanoseconds,
          ready.observedUptimeNanoseconds < trip.inactive.observedUptimeNanoseconds,
          ready.applicationState == 0 || !requireSystem && ready.applicationState == nil,
          !ready.visibleIDs.isEmpty, ready.visibleIDs.count < Int(maximum),
          trip.inactive.before.busy, trip.inactive.command == "generate",
          !trip.inactive.before.ids.isEmpty, trip.inactive.before.ids.count < Int(maximum),
          Array(trip.inactive.before.ids.prefix(ready.visibleIDs.count)) == ready.visibleIDs else {
        throw lifecycleRunError("READY must precede active foreground interruption")
    }
}
struct NativeLifecycleResult: Encodable, Sendable {
    let backend: String
    var passed = false
    var error: String?
    var cleanupErrors: [String] = []
    var endpointErrors: [String] = []
    var filesBefore: [String:String] = [:]
    var filesAfter: [String:String] = [:]
    var checkpoints: [NativeMemoryCheckpoint] = []
    var environmentBefore: NativeEnvironment?
    var environmentAfter: NativeEnvironment?
    var ready: NativeLifecycleReady?
    var loadedAdmittedBytes: UInt64 = 0
    var roundTrip: NativeLifecycleRoundTrip?
    var interruptedPrefix: [UInt32] = []
    var resumedIDs: [UInt32] = []
    var referenceIDs: [UInt32] = []
    var rebuiltIDs: [UInt32] = []
    var freshIDs: [UInt32] = []
    var trace: NativeLifecycleTraceReport?
    var closed = false
    var released = false
    let actualOSLifecycleQualified = false
    let physicalDeviceQualified = false
    let m7Complete = false
}
// Endpoint checks are independent: a process-cap failure must not suppress
// artifact reproof, and rejected observations remain in the caller's evidence.
func checkNativeLifecycleEndpoints(admitted: Bool, memory: (String) throws -> Void,
                                   reprove: () throws -> [String:String]) -> (files: [String:String], errors: [String]) {
    var files: [String:String] = [:];var errors: [String] = []
    do { try memory("closed") } catch { errors.append("Closed memory: " + String(error.localizedDescription.prefix(512))) }
    if admitted {
        do { files = try reprove() } catch { errors.append("Artifact endpoint: " + String(error.localizedDescription.prefix(512))) }
    }
    do { try memory("after-final-artifact-proof") } catch { errors.append("Final memory: " + String(error.localizedDescription.prefix(512))) }
    return (files,errors)
}
// The only automatic phase under test is cancellation/session cleanup. Replay
// is a separate coordinator command after an observed foreground return. There
// is one worker/model. The uninterrupted reference repeats original prefill
// and the same number of single-token decode steps before continuing; it never
// needs a redundant live model copy.
func runNativeLifecycle(_ config: NativeLifecycleConfig, root: URL, signals: NativeLifecycleSignals,
                        requireSystem: Bool = true,
                        onReady: (NativeLifecycleReady) throws -> Void) -> NativeLifecycleResult {
    precondition(!Thread.isMainThread)
    var result = NativeLifecycleResult(backend:config.metal ? "metal":"cpu")
    var owner: NativeWorker?;weak var weakOwner: NativeWorker?
    var trace: NativeLifecycleTrace?
    var artifactsAdmitted = false
    func foreground() throws {
        try signals.check()
        let state = NativeEnvironment.read().applicationState
        guard state == nil || state == 0 else { throw lifecycleRunError("Lifecycle coordinator requires foreground") }
    }
    func memory(_ name: String) throws {
        let value = try NativeFootprint.read()
        result.checkpoints.append(NativeMemoryCheckpoint(phase:name,footprint:value))
        guard value.physicalBytes <= config.processBudget, value.lifetimePeakPhysicalBytes <= config.processBudget else {
            throw lifecycleRunError("Lifecycle whole-process trial budget exceeded")
        }
    }
    @discardableResult
    func record(_ name: String) throws -> NativeLifecycleEvent {
        guard let trace else { throw lifecycleRunError("Lifecycle trace unavailable") }
        let event = try trace.record(name,origin:.coordinator,applicationState:NativeEnvironment.read().applicationState)
        guard event.footprint.physicalBytes <= config.processBudget, event.footprint.lifetimePeakPhysicalBytes <= config.processBudget else {
            throw lifecycleRunError("Lifecycle event trial budget exceeded")
        }
        return event
    }
    func perform(_ command: NativeCommand, _ name: String, status: UInt32 = 0) throws -> NativeSnapshot {
        try foreground();guard let worker = owner, let trace else { throw lifecycleRunError("Lifecycle owner unavailable") }
        try trace.begin(name)
        guard worker.submit(command), let state = worker.waitUntilIdle(seconds:180), !state.closed, state.status == status else {
            throw lifecycleRunError("Lifecycle command deadline/status differs: " + name)
        }
        try foreground();try record(name);try trace.end(name)
        if state.hasModel {
            guard state.admittedBytes > 0, state.admittedBytes <= config.numericalBudget else { throw lifecycleRunError("Lifecycle native admission differs") }
        }
        return state
    }
    do {
        try config.validate();try foreground();result.environmentBefore = NativeEnvironment.read()
        try memory("before-artifact-proof")
        result.filesBefore = try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
        artifactsAdmitted = true;try memory("after-artifact-proof")
        let configName = config.modelFolder + "/config.json"
        let handle = try FileHandle(forReadingFrom:try localURL(root,configName));defer { try? handle.close() }
        let bytes = try handle.read(upToCount:512 * 1024+1) ?? Data()
        struct Vocabulary: Decodable { let vocab_size: UInt32 }
        guard bytes.count <= 512 * 1024,
              SHA256.hash(data:bytes).map({ String(format:"%02x",$0) }).joined() == config.files[configName],
              try JSONDecoder().decode(Vocabulary.self,from:bytes).vocab_size == config.vocabulary else {
            throw lifecycleRunError("Lifecycle model vocabulary differs")
        }
        try foreground();owner = try NativeWorker(captureSavedPrefix:true);weakOwner = owner
        trace = try NativeLifecycleTrace(worker:owner!,maximumPrefix:Int(config.capacity));try signals.attach(owner!,trace:trace!)
        var settings = NativeSettings(directory:try localURL(root,config.modelFolder).path)
        settings.packedDirectory = config.packedFolder.isEmpty ? "":try localURL(root,config.packedFolder).path
        settings.metal = config.metal;settings.capacity = config.capacity;settings.chunk = config.chunk;settings.numericalBudget = config.numericalBudget
        let loaded = try perform(.load(settings),"loaded")
        guard loaded.hasModel, loaded.hasSession, loaded.savedPrefix == [] else { throw lifecycleRunError("Lifecycle load owners differ") }
        result.loadedAdmittedBytes = loaded.admittedBytes
        try memory("loaded");try signals.arm();try trace!.begin("generate")
        let submitted = DispatchTime.now().uptimeNanoseconds
        guard owner!.submit(.generate(operation:0,tokens:config.prompt,maximum:config.maximum,stops:[])) else { throw lifecycleRunError("Lifecycle generation rejected") }
        while true {
            try signals.check()
            let current = DispatchTime.now().uptimeNanoseconds
            guard current >= submitted, current-submitted < 180_000_000_000 else { throw lifecycleRunError("Lifecycle active generation deadline") }
            let state = owner!.snapshot()
            if result.ready == nil, state.busy, !state.ids.isEmpty, state.ids.count < Int(config.maximum) {
                try foreground();let event = try record("generation-active")
                if event.before.busy, !event.before.ids.isEmpty, event.before.ids.count < Int(config.maximum),
                   event.applicationState == 0 || !requireSystem && event.applicationState == nil {
                    let ready = NativeLifecycleReady(eventIndex:event.index,submittedUptimeNanoseconds:submitted,
                        observedUptimeNanoseconds:event.recordedUptimeNanoseconds,visibleIDs:event.before.ids,applicationState:event.applicationState)
                    result.ready = ready;try onReady(ready)
                }
            }
            if !state.busy { break }
            Thread.sleep(forTimeInterval:0.002)
        }
        guard result.ready != nil else { throw lifecycleRunError("No demonstrably active generation window") }
        let trip = try signals.finish(seconds:180,requireSystem:requireSystem)
        try requireNativeLifecycleReady(result.ready!,trip:trip,maximum:config.maximum,requireSystem:requireSystem)
        guard let suspended = owner!.waitUntilIdle(seconds:180), suspended.status == 0,
              suspended.hasModel, !suspended.hasSession, suspended.history == 0, suspended.evaluated == 0, suspended.pending == 0,
              !suspended.ids.isEmpty, suspended.ids.count < Int(config.maximum),
              suspended.savedPrefix == config.prompt+suspended.ids else { throw lifecycleRunError("Active interruption/saved prefix differs") }
        result.roundTrip = trip;result.interruptedPrefix = suspended.savedPrefix!
        try record("suspended");try trace!.end("generate");try foreground()
        let replayed = try perform(.replay,"replayed")
        guard replayed.savedPrefix == result.interruptedPrefix, replayed.history == UInt32(result.interruptedPrefix.count),
              replayed.evaluated == replayed.history, replayed.pending == 0 else { throw lifecycleRunError("Explicit lifecycle replay differs") }
        let resumed = try perform(.generate(operation:2,tokens:[],maximum:3,stops:[]),"resumed")
        guard resumed.reason == 1, resumed.ids.count == 3,
              resumed.savedPrefix == result.interruptedPrefix+resumed.ids else { throw lifecycleRunError("Lifecycle continuation differs") }
        result.resumedIDs = resumed.ids
        let reset = try perform(.reset,"reference-reset")
        guard reset.savedPrefix == [], reset.history == 0, reset.evaluated == 0, reset.pending == 0 else { throw lifecycleRunError("Reference reset differs") }
        let sampledCount = result.interruptedPrefix.count-config.prompt.count
        let rebuilt = try perform(.generate(operation:0,tokens:config.prompt,maximum:UInt32(sampledCount),stops:[]),"reference-prefix")
        guard rebuilt.reason == 1, rebuilt.ids.count == sampledCount,
              rebuilt.savedPrefix == result.interruptedPrefix else { throw lifecycleRunError("Original prefill/decode prefix reconstruction differs") }
        result.rebuiltIDs = rebuilt.ids
        let reference = try perform(.generate(operation:2,tokens:[],maximum:3,stops:[]),"reference")
        guard reference.reason == 1, reference.ids == result.resumedIDs,
              reference.savedPrefix == result.interruptedPrefix+result.resumedIDs else { throw lifecycleRunError("Uninterrupted exact-prefix reference differs") }
        result.referenceIDs = reference.ids
        signals.observe(.memoryWarning,origin:.syntheticCheck,applicationState:NativeEnvironment.read().applicationState)
        guard let unloaded = owner!.waitUntilIdle(seconds:180), unloaded.status == 0, !unloaded.hasModel, !unloaded.hasSession,
              unloaded.savedPrefix == [], unloaded.savedHistory == 0 else { throw lifecycleRunError("Synthetic warning failed to unload") }
        try record("warning-unloaded")
        let rejected = try perform(.replay,"replay-rejected",status:1)
        guard !rejected.hasModel, !rejected.hasSession, rejected.savedPrefix == [] else { throw lifecycleRunError("Warning failed to require fresh load") }
        let fresh = try perform(.load(settings),"fresh-load")
        guard fresh.hasModel, fresh.hasSession, fresh.savedPrefix == [] else { throw lifecycleRunError("Fresh lifecycle load differs") }
        let regenerated = try perform(.generate(operation:0,tokens:config.prompt,maximum:3,stops:[]),"fresh-prompt")
        let known = Array((result.rebuiltIDs+result.referenceIDs).prefix(3))
        guard regenerated.reason == 1, regenerated.ids.count == 3, regenerated.ids == known else { throw lifecycleRunError("Fresh prompt oracle differs") }
        result.freshIDs = regenerated.ids
        let final = try perform(.unload,"unloaded")
        guard !final.hasModel, !final.hasSession, final.savedPrefix == [], final.admittedBytes == 0 else { throw lifecycleRunError("Lifecycle final unload differs") }
    } catch { result.error = String(error.localizedDescription.prefix(512)) }
    if let worker = owner {
        worker.close();let finished = worker.waitUntilClosed(seconds:180);let state = worker.snapshot()
        result.closed = finished && state.closed && state.workerFinished && state.status == 0
            && !state.hasModel && !state.hasSession && state.savedPrefix == nil && state.savedHistory == 0
            && state.history == 0 && state.evaluated == 0 && state.pending == 0 && state.reason == 0
        if result.closed {
            do { try record("closed") } catch { result.cleanupErrors.append(String(error.localizedDescription.prefix(512))) }
        } else { result.cleanupErrors.append("Lifecycle close acknowledgement/status differs") }
    }
    signals.detach()
    if result.closed, let trace {
        do { try trace.detach() } catch { result.cleanupErrors.append(String(error.localizedDescription.prefix(512))) }
    }
    owner = nil
    let releaseDeadline = Date(timeIntervalSinceNow:5)
    while weakOwner != nil && Date() < releaseDeadline { Thread.sleep(forTimeInterval:0.002) }
    result.released = weakOwner == nil
    if !result.released { result.cleanupErrors.append("Lifecycle worker owner retained after close") }
    result.trace = trace?.report()
    result.environmentAfter = NativeEnvironment.read()
    let endpoints = checkNativeLifecycleEndpoints(admitted:artifactsAdmitted,memory:memory,reprove:{
        try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
    })
    result.filesAfter = endpoints.files;result.endpointErrors.append(contentsOf:endpoints.errors)
    if let trace = result.trace, trace.events.contains(where:{ $0.footprint.lifetimePeakPhysicalBytes > config.processBudget || $0.footprint.physicalBytes > config.processBudget }) {
        result.endpointErrors.append("Lifecycle observed event process cap exceeded")
    }
    result.passed = result.error == nil && result.cleanupErrors.isEmpty && result.endpointErrors.isEmpty && result.closed && result.released
        && result.roundTrip != nil && result.ready != nil && result.trace?.error == nil && result.trace?.activeCommand == nil
        && result.trace?.detachedAfterClose == true && result.filesBefore == config.files && result.filesAfter == config.files
    return result
}

import Foundation
import CoreFoundation
import Darwin
import CryptoKit
#if canImport(UIKit)
import UIKit
#endif

private func measurementError(_ message: String) -> NSError {
    NSError(domain:"TokletMeasurement",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}

// Treat the Foundation bridge as checked metadata: no Boolean, negative,
// fractional or truncating conversion may become an optimistic space report.
func nativeFilesystemFreeBytes(_ value: Any?) throws -> UInt64 {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID(),
          let bytes = UInt64(number.stringValue), bytes > 0,
          number.compare(NSNumber(value:bytes)) == .orderedSame else {
        throw measurementError("Filesystem free-byte observation unavailable")
    }
    return bytes
}

struct NativeFilesystemAvailability: Encodable, Sendable {
    let freeBytes: UInt64
    let readStartedUptimeNanoseconds: UInt64
    let readCompletedUptimeNanoseconds: UInt64
    let instrument = "Foundation.FileManager.attributesOfFileSystem.systemFreeSize"
    let scope = "trusted-root-filesystem"

    static func read(_ root: URL) throws -> NativeFilesystemAvailability {
        let start = DispatchTime.now().uptimeNanoseconds
        let attributes = try FileManager.default.attributesOfFileSystem(forPath:root.path)
        let free = try nativeFilesystemFreeBytes(attributes[.systemFreeSize])
        return NativeFilesystemAvailability(freeBytes:free,readStartedUptimeNanoseconds:start,
            readCompletedUptimeNanoseconds:DispatchTime.now().uptimeNanoseconds)
    }
}

struct NativeFootprint: Codable, Sendable {
    let physicalBytes: UInt64
    let lifetimePeakPhysicalBytes: UInt64
    let returnedNaturalCount: UInt32

    static func read() throws -> NativeFootprint {
        var info = task_vm_info_data_t()
        let capacity = MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        var count = mach_msg_type_number_t(capacity)
        // The initialized C record owns capacity integer_t slots. task_info
        // writes at most the supplied count for this task, during this rebound
        // lifetime only. No Swift access aliases that storage during the call.
        let status = withUnsafeMutablePointer(to:&info) { pointer in
            pointer.withMemoryRebound(to:integer_t.self,capacity:capacity) {
                task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
            }
        }
        // SDK revision macros containing sizeof are not imported into Swift.
        // Derive byte coverage from this SDK's actual C layout instead of a
        // hardcoded revision/count. Both physical fields precede peakEnd.
        guard let offset = MemoryLayout<task_vm_info_data_t>.offset(of:\.ledger_phys_footprint_peak) else {
            throw measurementError("Unavailable task memory layout")
        }
        let peakEnd = offset + MemoryLayout<Int64>.size
        guard status == KERN_SUCCESS, count <= capacity,
              Int(count) * MemoryLayout<integer_t>.size >= peakEnd,
              info.ledger_phys_footprint_peak >= 0 else {
            throw measurementError("TASK_VM_INFO physical peak unavailable: status \(status), count \(count)")
        }
        return NativeFootprint(physicalBytes:info.phys_footprint,
                               lifetimePeakPhysicalBytes:UInt64(info.ledger_phys_footprint_peak),
                               returnedNaturalCount:count)
    }
}

struct NativeEnvironment: Codable, Sendable {
    let thermalState: Int
    let lowPowerMode: Bool
    let batteryState: Int?
    let batteryLevel: Double?
    let applicationState: Int?

    static func read() -> NativeEnvironment {
        #if canImport(UIKit)
        // Diagnostic runner is off the main thread; UIKit reads stay on it.
        return DispatchQueue.main.sync {
            let device = UIDevice.current
            return NativeEnvironment(thermalState:ProcessInfo.processInfo.thermalState.rawValue,
                lowPowerMode:ProcessInfo.processInfo.isLowPowerModeEnabled,
                batteryState:device.batteryState.rawValue,
                batteryLevel:device.batteryLevel >= 0 ? Double(device.batteryLevel) : nil,
                applicationState:UIApplication.shared.applicationState.rawValue)
        }
        #else
        return NativeEnvironment(thermalState:ProcessInfo.processInfo.thermalState.rawValue,
            lowPowerMode:ProcessInfo.processInfo.isLowPowerModeEnabled,
            batteryState:nil,batteryLevel:nil,applicationState:nil)
        #endif
    }
}

struct NativeMeasurementConfig: Codable, Sendable {
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
    let expectedPrefix: [UInt32]
    let warmups: Int
    let repetitions: Int
    let minimumNativeSeconds: Double

    func validate() throws {
        guard version == 1, (1...32768).contains(capacity), chunk > 0, chunk <= capacity,
              maximum > 0, maximum <= 128, !prompt.isEmpty,
              maximum <= capacity, prompt.count <= Int(capacity - maximum),
              expectedPrefix.count <= Int(maximum), warmups == 3, repetitions == 10,
              minimumNativeSeconds.isFinite,
              minimumNativeSeconds == 0 || (600...3600).contains(minimumNativeSeconds),
              numericalBudget > 4096, numericalBudget < processBudget,
              processBudget <= 4 * 1024 * 1024 * 1024,
              !files.isEmpty, files.count <= 256,
              files[modelFolder + "/config.json"] != nil,
              packedFolder.isEmpty || metal,
              packedFolder.isEmpty || (packedFolder != modelFolder
                && !packedFolder.hasPrefix(modelFolder + "/") && !modelFolder.hasPrefix(packedFolder + "/")) else {
            throw measurementError("Invalid explicit measurement policy")
        }
        try validateRelative(modelFolder)
        if !packedFolder.isEmpty { try validateRelative(packedFolder) }
        for (name,hash) in files {
            try validateRelative(name)
            guard name.hasPrefix(modelFolder + "/") || (!packedFolder.isEmpty && name.hasPrefix(packedFolder + "/")),
                  hash.count == 64, hash.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw measurementError("Invalid admitted artifact inventory")
            }
        }
    }
}

private func validateRelative(_ name: String) throws {
    guard !name.isEmpty, name.utf8.count <= 16384, !name.hasPrefix("/"),
          name.split(separator:"/",omittingEmptySubsequences:false).allSatisfy({ component in
              !component.isEmpty && component != "." && component != ".." && component.utf8.allSatisfy {
                  (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45,46,95].contains($0)
              }
          }) else { throw measurementError("Invalid relative artifact path") }
}

private func localURL(_ root: URL, _ name: String) throws -> URL {
    try validateRelative(name)
    var url = root
    for component in name.split(separator:"/") {
        url.appendPathComponent(String(component))
        let properties = try url.resourceValues(forKeys:[.isSymbolicLinkKey])
        guard properties.isSymbolicLink != true else { throw measurementError("Artifact symlink rejected") }
    }
    return url
}

private func streamedHash(_ url: URL, maximum: UInt64) throws -> String {
    let properties = try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey])
    guard properties.isRegularFile == true, properties.isSymbolicLink != true else {
        throw measurementError("Artifact must be a regular file")
    }
    let handle = try FileHandle(forReadingFrom:url)
    defer { try? handle.close() }
    var hash = SHA256(); var count: UInt64 = 0
    // FileHandle's bridged Data can otherwise remain autoreleased for the
    // entire diagnostic queue block. Only EOF escapes each chunk's pool.
    while try autoreleasepool(invoking: { () throws -> Bool in
        guard let data = try handle.read(upToCount:256 * 1024), !data.isEmpty else { return false }
        count += UInt64(data.count)
        guard count <= maximum else { throw measurementError("Artifact exceeds file limit") }
        hash.update(data:data)
        return true
    }) {}
    return hash.finalize().map { String(format:"%02x",$0) }.joined()
}

func admitMeasurementConfig(root: URL, name: String, sha256: String) throws -> NativeMeasurementConfig {
    let url = try localURL(root,name)
    let hash = try streamedHash(url,maximum:512 * 1024)
    guard hash == sha256 else { throw measurementError("Measurement config SHA differs") }
    // Recheck after reading too: do not decode bytes different from admission.
    let handle = try FileHandle(forReadingFrom:url)
    defer { try? handle.close() }
    let data = try handle.read(upToCount:512 * 1024 + 1) ?? Data()
    guard data.count <= 512 * 1024,
          SHA256.hash(data:data).map({ String(format:"%02x",$0) }).joined() == sha256 else {
        throw measurementError("Measurement config changed during read")
    }
    let fields: Set<String> = ["version","modelFolder","packedFolder","files","metal","capacity","chunk",
        "numericalBudget","processBudget","prompt","maximum","expectedPrefix","warmups","repetitions","minimumNativeSeconds"]
    guard let object = try JSONSerialization.jsonObject(with:data) as? [String:Any], Set(object.keys) == fields else {
        throw measurementError("Exact measurement config fields required")
    }
    let config = try JSONDecoder().decode(NativeMeasurementConfig.self,from:data)
    try config.validate()
    return config
}

private func artifactProof(_ config: NativeMeasurementConfig, _ root: URL) throws -> [String:String] {
    // Foundation enumeration may return /private/var while Documents uses /var.
    // Canonicalize only the trusted root here; reject child links before
    // canonicalizing an enumerated entry so forbidden links cannot disappear.
    let root = root.standardizedFileURL.resolvingSymlinksInPath()
    var names = Set<String>()
    let rootComponents = root.pathComponents
    for folder in [config.modelFolder,config.packedFolder] where !folder.isEmpty {
        let directory = try localURL(root,folder)
        let properties = try directory.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
        var enumerationError: Error?
        guard properties.isDirectory == true, properties.isSymbolicLink != true,
              let iterator = FileManager.default.enumerator(at:directory,
                  includingPropertiesForKeys:[.isDirectoryKey,.isRegularFileKey,.isSymbolicLinkKey],
                  errorHandler:{ _,error in enumerationError = error; return false }) else {
            throw measurementError("Artifact directory unavailable")
        }
        var count = 0
        for case let url as URL in iterator {
            count += 1
            guard count <= 512 else {
                throw measurementError("Artifact inventory exceeds bounds")
            }
            let p = try url.resourceValues(forKeys:[.isDirectoryKey,.isRegularFileKey,.isSymbolicLinkKey])
            guard p.isSymbolicLink != true else { throw measurementError("Artifact symlink rejected") }
            let canonical = url.standardizedFileURL.resolvingSymlinksInPath().pathComponents
            guard Array(canonical.prefix(rootComponents.count)) == rootComponents else {
                throw measurementError("Artifact inventory leaves trusted root")
            }
            if p.isDirectory == true { continue }
            guard p.isRegularFile == true else { throw measurementError("Nonregular artifact rejected") }
            names.insert(canonical.dropFirst(rootComponents.count).joined(separator:"/"))
        }
        if let error = enumerationError { throw error }
    }
    guard names == Set(config.files.keys) else { throw measurementError("Exact artifact inventory differs") }
    var observed: [String:String] = [:]
    for name in names.sorted() {
        let hash = try streamedHash(try localURL(root,name),maximum:4 * 1024 * 1024 * 1024)
        guard config.files[name] == hash else { throw measurementError("Artifact SHA differs: \(name)") }
        observed[name] = hash
    }
    return observed
}

struct NativeMeasurementRequest: Codable, Sendable {
    let index: Int
    let role: String
    let submittedUptimeNanoseconds: UInt64
    let completedUptimeNanoseconds: UInt64
    let tokens: [UInt32]
    let events: [NativeTokenTiming]
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
    let reason: UInt32
    let status: UInt32
    let admittedBytes: UInt64
    let footprint: NativeFootprint
    let environment: NativeEnvironment
}
struct NativeMemoryCheckpoint: Codable, Sendable {
    let phase: String
    let footprint: NativeFootprint
}
struct NativeMeasurementResult: Encodable, Sendable {
    let backend: String
    var passed = false
    var error: String?
    var cleanupErrors: [String] = []
    var endpointErrors: [String] = []
    var filesBefore: [String:String] = [:]
    var filesAfter: [String:String] = [:]
    var checkpoints: [NativeMemoryCheckpoint] = []
    var filesystemBefore: NativeFilesystemAvailability?
    var interruptionCount: UInt64 = 0
    var loadAndSessionSeconds: Double = 0
    var warmupsCompleted = 0
    var measuredCompleted = 0
    var sumNativeLastEventSeconds: Double = 0
    var closed = false
    var released = false
    let physicalDeviceQualified = false
    let m7Complete = false
}

// Notification callbacks may arrive on the main thread while native work owns
// its dedicated thread. The lock protects the weak worker and interruption count;
// native cancellation/cleanup is requested only after releasing this lock.
private final class MeasurementInterruptions: @unchecked Sendable {
    private let lock = NSLock()
    private weak var worker: NativeWorker?
    private var count: UInt64 = 0
    private var observers: [NSObjectProtocol] = []
    init() {
        #if canImport(UIKit)
        for (name,action) in [(UIApplication.willResignActiveNotification,0),
                              (UIApplication.didEnterBackgroundNotification,1),
                              (UIApplication.didReceiveMemoryWarningNotification,2)] {
            observers.append(NotificationCenter.default.addObserver(forName:name,object:nil,queue:nil) { [weak self] _ in
                self?.interrupt(action)
            })
        }
        #endif
    }
    func attach(_ owner: NativeWorker?) {
        lock.lock(); worker = owner; lock.unlock()
    }
    func observed() -> UInt64 {
        lock.lock(); defer { lock.unlock() }; return count
    }
    private func interrupt(_ action: Int) {
        lock.lock(); count += 1; let owner = worker; lock.unlock()
        if action == 0 { owner?.cancel() }
        else { owner?.interrupt(unload:action == 2) }
    }
    func stop() {
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
        observers.removeAll(); attach(nil)
    }
}

// This bounded raw-ID measurement has no tokenizer, hidden stops or lifecycle
// qualification. The explicit numerical cap admits native allocations before
// loading; whole-process peak is an observed post-operation rejection policy.
// App.init may run before UIKit reports active. This startup-only gate runs off
// the main thread, before any artifact/load/request timing or interruption
// observer. Once measurement begins, all existing foreground checks stay strict.
func awaitNativeMeasurementForeground(timeoutNanoseconds: UInt64 = 30_000_000_000,
    now: () -> UInt64 = { DispatchTime.now().uptimeNanoseconds },
    isActive: () -> Bool = {
        let state = NativeEnvironment.read().applicationState
        return state == nil || state == 0
    },
    pause: () -> Void = { Thread.sleep(forTimeInterval:0.02) }) throws {
    precondition(!Thread.isMainThread)
    let start = now()
    while true {
        let current = now()
        guard current >= start, current - start < timeoutNanoseconds else {
            throw measurementError("Foreground startup deadline exceeded")
        }
        let active = isActive()
        // A synchronous main-queue state read may itself be delayed. Enforce
        // the checked deadline after it too; the host owns the process watchdog.
        let observed = now()
        guard observed >= current, observed - start < timeoutNanoseconds else {
            throw measurementError("Foreground startup deadline exceeded")
        }
        if active { return }
        pause()
    }
}

func runNativeMeasurement(_ config: NativeMeasurementConfig, root: URL,
                          onSample: (NativeMeasurementRequest) throws -> Void) -> NativeMeasurementResult {
    precondition(!Thread.isMainThread)
    var result = NativeMeasurementResult(backend:config.metal ? "metal" : "cpu")
    let interruptions = MeasurementInterruptions()
    defer { interruptions.stop() }
    var owner: NativeWorker?
    weak var weakOwner: NativeWorker?
    func memory(_ phase: String) throws {
        let footprint = try NativeFootprint.read()
        result.checkpoints.append(NativeMemoryCheckpoint(phase:phase,footprint:footprint))
        guard footprint.physicalBytes <= config.processBudget,
              footprint.lifetimePeakPhysicalBytes <= config.processBudget else {
            throw measurementError("Whole-process trial budget exceeded: \(phase)")
        }
    }
    func perform(_ command: NativeCommand) throws -> NativeSnapshot {
        guard let worker = owner, worker.submit(command),
              let state = worker.waitUntilIdle(seconds:1200) else { throw measurementError("Worker completion deadline/rejection") }
        guard state.status == 0 else { throw measurementError("Native status \(state.status): \(state.message)") }
        return state
    }
    func requireForeground() throws {
        let state = NativeEnvironment.read().applicationState
        guard (state == nil || state == 0), interruptions.observed() == 0 else {
            throw measurementError("Foreground measurement interrupted")
        }
    }
    do {
        try config.validate()
        try requireForeground()
        result.filesystemBefore = try NativeFilesystemAvailability.read(root)
        try memory("before-artifact-proof")
        result.filesBefore = try artifactProof(config,root)
        try memory("after-artifact-proof")
        try requireForeground()
        owner = try NativeWorker(); weakOwner = owner; interruptions.attach(owner)
        var settings = NativeSettings(directory:try localURL(root,config.modelFolder).path)
        settings.packedDirectory = config.packedFolder.isEmpty ? "" : try localURL(root,config.packedFolder).path
        settings.metal = config.metal; settings.capacity = config.capacity; settings.chunk = config.chunk
        settings.numericalBudget = config.numericalBudget
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let loaded = try perform(.load(settings))
        result.loadAndSessionSeconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9
        guard loaded.hasModel, loaded.hasSession else { throw measurementError("Native load owners missing") }
        try memory("loaded")
        var index = 0
        while index < config.warmups + config.repetitions || result.sumNativeLastEventSeconds < config.minimumNativeSeconds {
            guard index < 2048 else { throw measurementError("Sustained sequence exceeds request bound") }
            let reset = try perform(.reset)
            guard reset.history == 0, reset.evaluated == 0, reset.pending == 0 else { throw measurementError("Reset prefix differs") }
            let environment = NativeEnvironment.read()
            // UIApplication.State.active is zero. Background behavior is a
            // separate protocol; never count a suspended run as foreground.
            guard (environment.applicationState == nil || environment.applicationState == 0), interruptions.observed() == 0 else {
                throw measurementError("Foreground measurement interrupted")
            }
            let submitted = DispatchTime.now().uptimeNanoseconds
            let state = try perform(.generate(operation:0,tokens:config.prompt,maximum:config.maximum,stops:[]))
            let completed = DispatchTime.now().uptimeNanoseconds
            let footprint = try NativeFootprint.read()
            let after = NativeEnvironment.read()
            let role = index < config.warmups ? "warmup" : "measured"
            try onSample(NativeMeasurementRequest(index:index,role:role,
                submittedUptimeNanoseconds:submitted,completedUptimeNanoseconds:completed,tokens:state.ids,
                events:state.tokenTimings,history:state.history,evaluated:state.evaluated,pending:state.pending,
                reason:state.reason,status:state.status,admittedBytes:state.admittedBytes,footprint:footprint,environment:after))
            guard state.ids.count == Int(config.maximum), state.tokenTimings.count == state.ids.count,
                  state.history == UInt32(config.prompt.count) + config.maximum,
                  state.evaluated + 1 == state.history, state.pending == 1, state.reason == 1,
                  Array(state.ids.prefix(config.expectedPrefix.count)) == config.expectedPrefix,
                  footprint.physicalBytes <= config.processBudget, footprint.lifetimePeakPhysicalBytes <= config.processBudget,
                  (after.applicationState == nil || after.applicationState == 0), interruptions.observed() == 0 else {
                throw measurementError("Request oracle/state/whole-process budget/foreground differs")
            }
            for (offset,event) in state.tokenTimings.enumerated() {
                guard event.index == UInt32(offset), event.nativeElapsedSeconds.isFinite,
                      event.nativeElapsedSeconds >= 0,
                      event.callbackUptimeNanoseconds >= submitted, event.callbackUptimeNanoseconds <= completed else {
                    throw measurementError("Native timing/window differs")
                }
            }
            if role == "warmup" { result.warmupsCompleted += 1 }
            else {
                result.measuredCompleted += 1
                result.sumNativeLastEventSeconds += state.tokenTimings.last!.nativeElapsedSeconds
            }
            index += 1
        }
        _ = try perform(.unload)
        try memory("unloaded")
    } catch { result.error = String(describing:error) }
    if let worker = owner {
        worker.close(); result.closed = worker.waitUntilClosed(seconds:1200)
        if !result.closed { result.cleanupErrors.append("Worker close deadline") }
        else if worker.snapshot().status != 0 { result.cleanupErrors.append("Worker close status differs") }
    }
    interruptions.attach(nil)
    owner = nil
    let end = Date(timeIntervalSinceNow:5)
    while weakOwner != nil && Date() < end { Thread.sleep(forTimeInterval:0.002) }
    result.released = weakOwner == nil
    if !result.released { result.cleanupErrors.append("Worker owner still retained") }
    do { try memory("closed") } catch { result.cleanupErrors.append(String(describing:error)) }
    // Recheck admitted artifacts after cleanup on both success and failure.
    // Endpoint errors never replace a request/output/cleanup failure.
    if !result.filesBefore.isEmpty {
        do { result.filesAfter = try artifactProof(config,root) }
        catch { result.endpointErrors.append(String(describing:error)) }
        do { try memory("after-final-artifact-proof") }
        catch { result.cleanupErrors.append(String(describing:error)) }
    }
    result.interruptionCount = interruptions.observed()
    result.passed = result.interruptionCount == 0 && result.error == nil && result.cleanupErrors.isEmpty && result.endpointErrors.isEmpty && result.closed && result.released
        && result.filesBefore == config.files && result.filesAfter == config.files
        && result.warmupsCompleted == config.warmups && result.measuredCompleted >= config.repetitions
        && result.sumNativeLastEventSeconds >= config.minimumNativeSeconds
    return result
}

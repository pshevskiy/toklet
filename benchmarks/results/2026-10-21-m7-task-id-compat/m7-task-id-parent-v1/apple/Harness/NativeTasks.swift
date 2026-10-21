import Foundation
import CryptoKit

private func tasksError(_ message: String) -> NSError {
    NSError(domain:"TokletTasks",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}
struct NativeTasksConfig: Codable, Sendable {
    let version: Int
    let modelFolder: String
    let packedFolder: String
    let files: [String:String]
    let metal: Bool
    let capacity: UInt32
    let chunk: UInt32
    let numericalBudget: UInt64
    let processBudget: UInt64
    let tasksFile: String
    let tasksSHA256: String
    let caseCount: Int

    func validate() throws {
        guard version == 1, (1...100).contains(caseCount), (1...562).contains(capacity),
              chunk > 0, chunk <= 128, chunk <= capacity,
              numericalBudget > 4096, numericalBudget < processBudget,
              processBudget <= 4 * 1024 * 1024 * 1024,
              tasksSHA256.count == 64,
              tasksSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw tasksError("Invalid bounded task collection policy")
        }
        try validateNativeArtifactFields(modelFolder:modelFolder,packedFolder:packedFolder,files:files,metal:metal)
        try validateRelative(tasksFile)
        guard tasksFile != modelFolder, !tasksFile.hasPrefix(modelFolder + "/"),
              packedFolder.isEmpty || (tasksFile != packedFolder && !tasksFile.hasPrefix(packedFolder + "/")) else {
            throw tasksError("Tasks must be outside model inventories")
        }
    }
}
struct NativeTaskCase: Codable, Sendable {
    let id: String
    let promptTokens: [UInt32]
    let maximum: Int
}
struct NativeTaskSet: Codable, Sendable {
    let version: Int
    let vocabulary: UInt32
    let stops: [UInt32]
    let cases: [NativeTaskCase]

    func validate(_ config: NativeTasksConfig) throws {
        guard version == 1, vocabulary > 0, cases.count == config.caseCount,
              !stops.isEmpty, stops.count <= 2, Set(stops).count == stops.count,
              stops.allSatisfy({ $0 < vocabulary }), Set(cases.map(\.id)).count == cases.count else {
            throw tasksError("Task set identity/stop policy differs")
        }
        for item in cases {
            guard !item.id.isEmpty, item.id.utf8.count <= 80,
                  item.id.utf8.allSatisfy({ (48...57).contains($0) || (65...90).contains($0)
                    || (97...122).contains($0) || [45,46,95].contains($0) }),
                  (1...306).contains(item.promptTokens.count), (1...256).contains(item.maximum),
                  item.promptTokens.allSatisfy({ $0 < vocabulary }),
                  UInt32(item.promptTokens.count + item.maximum) <= config.capacity else {
                throw tasksError("Task prompt, output budget or identifier differs")
            }
        }
    }
}
private func admittedTaskJSON(root: URL, name: String, sha256: String) throws -> Data {
    let url = try localURL(root,name)
    guard try streamedHash(url,maximum:512 * 1024) == sha256 else { throw tasksError("Task artifact SHA differs") }
    let handle = try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    let data = try handle.read(upToCount:512 * 1024+1) ?? Data()
    guard data.count <= 512 * 1024,
          SHA256.hash(data:data).map({ String(format:"%02x",$0) }).joined() == sha256 else {
        throw tasksError("Task artifact changed during bounded read")
    }
    return data
}
func admitTasksConfig(root: URL, name: String, sha256: String) throws -> NativeTasksConfig {
    let data = try admittedTaskJSON(root:root,name:name,sha256:sha256)
    let fields: Set<String> = ["version","modelFolder","packedFolder","files","metal","capacity","chunk",
        "numericalBudget","processBudget","tasksFile","tasksSHA256","caseCount"]
    guard let object = try JSONSerialization.jsonObject(with:data) as? [String:Any], Set(object.keys) == fields else {
        throw tasksError("Exact task config fields required")
    }
    let config = try JSONDecoder().decode(NativeTasksConfig.self,from:data)
    try config.validate();return config
}
func admitTaskSet(_ config: NativeTasksConfig, root: URL) throws -> NativeTaskSet {
    try config.validate()
    let data = try admittedTaskJSON(root:root,name:config.tasksFile,sha256:config.tasksSHA256)
    guard let object = try JSONSerialization.jsonObject(with:data) as? [String:Any],
          Set(object.keys) == Set(["version","vocabulary","stops","cases"]),
          let cases = object["cases"] as? [[String:Any]], cases.count == config.caseCount,
          cases.allSatisfy({ Set($0.keys) == Set(["id","promptTokens","maximum"]) }) else {
        throw tasksError("Exact task set/case fields required")
    }
    let tasks = try JSONDecoder().decode(NativeTaskSet.self,from:data)
    try tasks.validate(config);return tasks
}
struct NativeTaskChunkSample: Codable, Sendable {
    let caseIndex: Int
    let caseID: String
    let chunk: NativeTaskChunk
    let submittedUptimeNanoseconds: UInt64
    let completedUptimeNanoseconds: UInt64
    let admittedBytes: UInt64
    let footprint: NativeFootprint
    let environment: NativeEnvironment
}
struct NativeTaskCaseSample: Codable, Sendable {
    let caseIndex: Int
    let caseID: String
    let response: NativeTaskResponse
    let resetHistory: UInt32
    let resetEvaluated: UInt32
    let resetPending: UInt32
    let admittedBytes: UInt64
    let footprint: NativeFootprint
    let environment: NativeEnvironment
}
struct NativeTasksResult: Encodable, Sendable {
    let backend: String
    var passed = false
    var error: String?
    var cleanupErrors: [String] = []
    var endpointErrors: [String] = []
    var filesBefore: [String:String] = [:]
    var filesAfter: [String:String] = [:]
    var tasksBeforeSHA256: String?
    var tasksAfterSHA256: String?
    var checkpoints: [NativeMemoryCheckpoint] = []
    var interruptionCount: UInt64 = 0
    var loadAndSessionSeconds: Double = 0
    var loadedAdmittedBytes: UInt64 = 0
    var chunksCompleted = 0
    var casesCompleted = 0
    var closed = false
    var released = false
    let fullQualityQualified = false
    let physicalDeviceQualified = false
    let m7Complete = false
}

// A bounded raw-response collector. No tokenizer, grading or task correctness is
// inferred here. Own-process peaks are observed rejection gates, not OS hard caps.
func runNativeTasks(_ config: NativeTasksConfig, root: URL,
    onChunk: (NativeTaskChunkSample) throws -> Void,
    onCase: (NativeTaskCaseSample) throws -> Void) -> NativeTasksResult {
    precondition(!Thread.isMainThread)
    var result = NativeTasksResult(backend:config.metal ? "metal":"cpu")
    let interruptions = NativeDiagnosticInterruptions();defer { interruptions.stop() }
    var owner: NativeWorker?;weak var weakOwner: NativeWorker?
    var submitted: UInt64 = 0;var completed: UInt64 = 0;var admitted: UInt64 = 0
    func footprint() throws -> NativeFootprint {
        let value = try NativeFootprint.read()
        guard value.physicalBytes <= config.processBudget, value.lifetimePeakPhysicalBytes <= config.processBudget else {
            throw tasksError("Whole-process task trial budget exceeded")
        }
        return value
    }
    func memory(_ phase: String) throws {
        result.checkpoints.append(NativeMemoryCheckpoint(phase:phase,footprint:try footprint()))
    }
    func foreground() throws {
        let environment = NativeEnvironment.read()
        guard (environment.applicationState == nil || environment.applicationState == 0), interruptions.observed() == 0 else {
            throw tasksError("Foreground task collection interrupted")
        }
    }
    func perform(_ command: NativeCommand) throws -> NativeSnapshot {
        try foreground()
        submitted = DispatchTime.now().uptimeNanoseconds
        guard let worker = owner, worker.submit(command), let state = worker.waitUntilIdle(seconds:1200) else {
            throw tasksError("Task worker deadline/rejection")
        }
        completed = DispatchTime.now().uptimeNanoseconds
        try foreground()
        guard state.status == 0 else { throw tasksError("Task native status differs: \(state.status)") }
        if case .unload = command {
            guard !state.hasModel, !state.hasSession, state.admittedBytes == 0 else {
                throw tasksError("Task unload owners/admission differ")
            }
        } else {
            guard state.admittedBytes > 0, state.admittedBytes <= config.numericalBudget else {
                throw tasksError("Task native admission differs")
            }
        }
        admitted = state.admittedBytes;return state
    }
    do {
        try config.validate();try foreground();try memory("before-artifact-proof")
        result.filesBefore = try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
        let tasks = try admitTaskSet(config,root:root);result.tasksBeforeSHA256 = config.tasksSHA256
        try memory("after-artifact-proof");try foreground()
        owner = try NativeWorker();weakOwner = owner;interruptions.attach(owner)
        var settings = NativeSettings(directory:try localURL(root,config.modelFolder).path)
        settings.packedDirectory = config.packedFolder.isEmpty ? "":try localURL(root,config.packedFolder).path
        settings.metal = config.metal;settings.capacity = config.capacity;settings.chunk = config.chunk
        settings.numericalBudget = config.numericalBudget
        let start = DispatchTime.now().uptimeNanoseconds
        let loaded = try perform(.load(settings))
        result.loadAndSessionSeconds = Double(DispatchTime.now().uptimeNanoseconds-start)/1e9
        guard loaded.hasModel, loaded.hasSession else { throw tasksError("Missing task native owners") }
        result.loadedAdmittedBytes = loaded.admittedBytes;try memory("loaded")
        for (index,item) in tasks.cases.enumerated() {
            let response = try collectNativeTaskResponse(prompt:item.promptTokens,maximum:item.maximum,
                capacity:config.capacity,vocabulary:tasks.vocabulary,stops:tasks.stops,checkActive:foreground,perform:perform,onChunk:{ chunk in
                    try foreground();let observed = try footprint();let environment = NativeEnvironment.read()
                    try foreground()
                    try onChunk(NativeTaskChunkSample(caseIndex:index,caseID:item.id,chunk:chunk,
                        submittedUptimeNanoseconds:submitted,completedUptimeNanoseconds:completed,
                        admittedBytes:admitted,footprint:observed,environment:environment))
                    result.chunksCompleted += 1
                })
            // Reset acknowledgement is part of each completed case, including
            // the last case. Keep the response's pre-reset prefix in its record.
            let reset = try perform(.reset)
            guard reset.history == 0, reset.evaluated == 0, reset.pending == 0, reset.savedHistory == 0,
                  reset.reason == 0, reset.stopToken == .max, reset.ids.isEmpty, reset.tokenTimings.isEmpty else {
                throw tasksError("Final case reset differs")
            }
            try foreground();let observed = try footprint();let environment = NativeEnvironment.read()
            try foreground()
            try onCase(NativeTaskCaseSample(caseIndex:index,caseID:item.id,response:response,
                resetHistory:reset.history,resetEvaluated:reset.evaluated,resetPending:reset.pending,
                admittedBytes:reset.admittedBytes,footprint:observed,environment:environment))
            result.casesCompleted += 1;try memory("case-\(index)")
        }
        _ = try perform(.unload);try memory("unloaded")
    } catch { result.error = String(describing:error) }
    if let worker = owner {
        worker.close();result.closed = worker.waitUntilClosed(seconds:1200)
        if !result.closed { result.cleanupErrors.append("Task worker close deadline") }
        else if worker.snapshot().status != 0 { result.cleanupErrors.append("Task worker close status differs") }
    }
    interruptions.attach(nil);owner = nil
    let deadline = Date(timeIntervalSinceNow:5)
    while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
    result.released = weakOwner == nil
    if !result.released { result.cleanupErrors.append("Task owner retained") }
    do { try memory("closed") } catch { result.cleanupErrors.append(String(describing:error)) }
    if !result.filesBefore.isEmpty {
        do {
            result.filesAfter = try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
            _ = try admitTaskSet(config,root:root);result.tasksAfterSHA256 = config.tasksSHA256
        } catch { result.endpointErrors.append(String(describing:error)) }
        do { try memory("after-final-artifact-proof") } catch { result.cleanupErrors.append(String(describing:error)) }
    }
    result.interruptionCount = interruptions.observed()
    result.passed = result.error == nil && result.cleanupErrors.isEmpty && result.endpointErrors.isEmpty
        && result.interruptionCount == 0 && result.closed && result.released
        && result.filesBefore == config.files && result.filesAfter == config.files
        && result.tasksBeforeSHA256 == config.tasksSHA256 && result.tasksAfterSHA256 == config.tasksSHA256
        && result.casesCompleted == config.caseCount
    return result
}

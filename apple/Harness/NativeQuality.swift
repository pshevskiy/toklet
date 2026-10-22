import Foundation
import CryptoKit

private func qualityError(_ message: String) -> NSError {
    NSError(domain:"TokletQuality",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}
struct NativeQualityConfig: Codable, Sendable {
    let version: Int
    let modelFolder: String
    let packedFolder: String
    let files: [String:String]
    let metal: Bool
    let capacity: UInt32
    let chunk: UInt32
    let numericalBudget: UInt64
    let processBudget: UInt64
    let tokensFile: String
    let tokensSHA256: String
    let targets: Int
    let blockSize: Int
    var mlx: Bool? = nil

    func validate() throws {
        try validateNativeJobBackend(version:version,metal:metal,packedFolder:packedFolder,mlx:mlx)
        guard (1...10000).contains(targets), (1...1024).contains(blockSize),
              capacity == UInt32(blockSize+1), chunk == 1,
              numericalBudget > 4096, numericalBudget < processBudget,
              processBudget <= 4 * 1024 * 1024 * 1024,
              tokensSHA256.count == 64,
              tokensSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw qualityError("Invalid explicit teacher-forcing policy")
        }
        try validateNativeArtifactFields(modelFolder:modelFolder,packedFolder:packedFolder,files:files,metal:metal)
        try validateRelative(tokensFile)
        guard tokensFile != modelFolder, !tokensFile.hasPrefix(modelFolder + "/"),
              packedFolder.isEmpty || (tokensFile != packedFolder && !tokensFile.hasPrefix(packedFolder + "/")) else {
            throw qualityError("Corpus must be outside model inventories")
        }
    }
}
func admitQualityConfig(root: URL, name: String, sha256: String) throws -> NativeQualityConfig {
    let url = try localURL(root,name)
    guard try streamedHash(url,maximum:512 * 1024) == sha256 else { throw qualityError("Quality config SHA differs") }
    let handle = try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    let data = try handle.read(upToCount:512 * 1024+1) ?? Data()
    guard data.count <= 512 * 1024,
          SHA256.hash(data:data).map({ String(format:"%02x",$0) }).joined() == sha256 else {
        throw qualityError("Quality config changed during bounded read")
    }
    let fields: Set<String> = ["version","modelFolder","packedFolder","files","metal","capacity","chunk",
        "numericalBudget","processBudget","tokensFile","tokensSHA256","targets","blockSize"]
    guard try nativeJobFieldsMatch(data,fields:fields) else {
        throw qualityError("Exact quality config fields required")
    }
    let config = try JSONDecoder().decode(NativeQualityConfig.self,from:data)
    try config.validate();return config
}
func admitQualityTokens(_ config: NativeQualityConfig, root: URL) throws -> [UInt32] {
    try config.validate()
    let expectedBytes = (config.targets+1)*4
    let url = try localURL(root,config.tokensFile)
    guard try streamedHash(url,maximum:UInt64(expectedBytes)) == config.tokensSHA256 else {
        throw qualityError("Corpus SHA differs")
    }
    let handle = try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    let data = try handle.read(upToCount:expectedBytes+1) ?? Data()
    guard data.count == expectedBytes,
          SHA256.hash(data:data).map({ String(format:"%02x",$0) }).joined() == config.tokensSHA256 else {
        throw qualityError("Exact corpus bytes changed during bounded read")
    }
    var tokens: [UInt32] = [];tokens.reserveCapacity(config.targets+1)
    // Byte loads do not assume Data's backing alignment. The validated length
    // is exactly4*(targets+1), so each group contains four initialized bytes.
    for index in stride(from:0,to:data.count,by:4) {
        tokens.append(UInt32(data[index]) | (UInt32(data[index+1])<<8)
            | (UInt32(data[index+2])<<16) | (UInt32(data[index+3])<<24))
    }
    return tokens
}
struct NativeQualitySample: Codable, Sendable {
    let target: NativeTeacherForcedScore
    let footprint: NativeFootprint
    let environment: NativeEnvironment
}
struct NativeQualityResult: Encodable, Sendable {
    let backend: String
    var passed = false
    var error: String?
    var cleanupErrors: [String] = []
    var endpointErrors: [String] = []
    var filesBefore: [String:String] = [:]
    var filesAfter: [String:String] = [:]
    var tokensBeforeSHA256: String?
    var tokensAfterSHA256: String?
    var checkpoints: [NativeMemoryCheckpoint] = []
    var interruptionCount: UInt64 = 0
    var loadAndSessionSeconds: Double = 0
    var samplesCompleted = 0
    var summary: NativeTeacherForcedSummary?
    var closed = false
    var released = false
    let fullQualityQualified = false
    let physicalDeviceQualified = false
    let m7Complete = false
}

// Own-process peak is an observed post-operation rejection policy, not an OS
// allocation hard cap. Incomplete/jetsam/interrupted streams cannot pass. Loss
// output is diagnostic and excludes tokenization, generation/tasks and grading.
func runNativeQuality(_ config: NativeQualityConfig, root: URL,
                      onSample: (NativeQualitySample) throws -> Void) -> NativeQualityResult {
    precondition(!Thread.isMainThread)
    var result = NativeQualityResult(backend:nativeJobBackendName(metal:config.metal,mlx:config.mlx))
    let interruptions = NativeDiagnosticInterruptions();defer { interruptions.stop() }
    var owner: NativeWorker?;weak var weakOwner: NativeWorker?
    var expectedBlocks = 0
    func footprint() throws -> NativeFootprint {
        let value = try NativeFootprint.read()
        guard value.physicalBytes <= config.processBudget, value.lifetimePeakPhysicalBytes <= config.processBudget else {
            throw qualityError("Whole-process quality trial budget exceeded")
        }
        return value
    }
    func memory(_ phase: String) throws {
        result.checkpoints.append(NativeMemoryCheckpoint(phase:phase,footprint:try footprint()))
    }
    func foreground() throws {
        let environment = NativeEnvironment.read()
        guard (environment.applicationState == nil || environment.applicationState == 0), interruptions.observed() == 0 else {
            throw qualityError("Foreground quality collection interrupted")
        }
    }
    func perform(_ command: NativeCommand) throws -> NativeSnapshot {
        try foreground()
        guard let worker = owner, worker.submit(command), let state = worker.waitUntilIdle(seconds:1200) else {
            throw qualityError("Quality worker deadline/rejection")
        }
        try foreground()
        guard state.status == 0 else { throw qualityError("Quality native status differs: \(state.status)") }
        if case .unload = command {
            guard !state.hasModel, !state.hasSession, state.admittedBytes == 0 else {
                throw qualityError("Quality unload owners/admission differ")
            }
        } else {
            guard state.admittedBytes > 0, state.admittedBytes <= config.numericalBudget else {
                throw qualityError("Quality native admission differs")
            }
        }
        return state
    }
    do {
        try config.validate();expectedBlocks = (config.targets+config.blockSize-1)/config.blockSize
        try foreground();try memory("before-artifact-proof")
        result.filesBefore = try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
        let tokens = try admitQualityTokens(config,root:root);result.tokensBeforeSHA256 = config.tokensSHA256
        try memory("after-artifact-proof");try foreground()
        owner = try NativeWorker();weakOwner = owner;interruptions.attach(owner)
        var settings = NativeSettings(directory:try localURL(root,config.modelFolder).path)
        settings.packedDirectory = config.packedFolder.isEmpty ? "":try localURL(root,config.packedFolder).path
        settings.metal = config.metal;settings.mlx = config.mlx == true;settings.capacity = config.capacity;settings.chunk = config.chunk
        settings.numericalBudget = config.numericalBudget
        let start = DispatchTime.now().uptimeNanoseconds
        let loaded = try perform(.load(settings))
        result.loadAndSessionSeconds = Double(DispatchTime.now().uptimeNanoseconds-start)/1e9
        guard loaded.hasModel, loaded.hasSession else { throw qualityError("Missing quality native owners") }
        try memory("loaded")
        let summary = try collectNativeTeacherForcedScores(tokens:tokens,blockSize:config.blockSize,perform:perform,onScore:{ row in
            try foreground()
            let observed = try footprint();let environment = NativeEnvironment.read()
            guard (environment.applicationState == nil || environment.applicationState == 0), interruptions.observed() == 0 else {
                throw qualityError("Quality sample interrupted")
            }
            try onSample(NativeQualitySample(target:row,footprint:observed,environment:environment))
            result.samplesCompleted += 1
        },onBlock:{ block in
            try foreground();try memory("block-\(block.index)")
        })
        result.summary = summary
        _ = try perform(.unload);try memory("unloaded")
    } catch { result.error = String(describing:error) }
    if let worker = owner {
        worker.close();result.closed = worker.waitUntilClosed(seconds:1200)
        if !result.closed { result.cleanupErrors.append("Quality worker close deadline") }
        else if worker.snapshot().status != 0 { result.cleanupErrors.append("Quality worker close status differs") }
    }
    interruptions.attach(nil);owner = nil
    let deadline = Date(timeIntervalSinceNow:5)
    while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
    result.released = weakOwner == nil
    if !result.released { result.cleanupErrors.append("Quality owner retained") }
    do { try memory("closed") } catch { result.cleanupErrors.append(String(describing:error)) }
    if !result.filesBefore.isEmpty {
        do {
            result.filesAfter = try nativeArtifactProof(modelFolder:config.modelFolder,packedFolder:config.packedFolder,files:config.files,root:root)
            _ = try admitQualityTokens(config,root:root);result.tokensAfterSHA256 = config.tokensSHA256
        } catch { result.endpointErrors.append(String(describing:error)) }
        do { try memory("after-final-artifact-proof") } catch { result.cleanupErrors.append(String(describing:error)) }
    }
    result.interruptionCount = interruptions.observed()
    result.passed = result.error == nil && result.cleanupErrors.isEmpty && result.endpointErrors.isEmpty
        && result.interruptionCount == 0 && result.closed && result.released
        && result.filesBefore == config.files && result.filesAfter == config.files
        && result.tokensBeforeSHA256 == config.tokensSHA256 && result.tokensAfterSHA256 == config.tokensSHA256
        && result.samplesCompleted == config.targets && result.summary?.targets == config.targets
        && result.summary?.blocks == expectedBlocks && result.summary?.blockSize == config.blockSize
    return result
}

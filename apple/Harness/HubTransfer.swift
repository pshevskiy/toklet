import CryptoKit
import Foundation
import Darwin

// Cleanup failures stay distinct from ordinary cancellation so every owning
// task group can join siblings and retain their failures before returning.
private struct HubCleanupFailure: LocalizedError {
    let causes: [String]
    var errorDescription: String? { "Download cleanup failed: " + causes.joined(separator:"; ") }
}

// Keep diagnostics independent of NSError userInfo, which may contain signed
// redirect URLs. Retain the code for transient-only retry classification.
private struct HubNetworkFailure: LocalizedError {
    let code: URLError.Code
    let detail: String
    let partialRangeBytes: Int64
    var errorDescription: String? { detail }
}

// One ephemeral URLSession and serial delegate queue per transfer. Delegate-owned
// fields are initialized before resume(), then accessed only on that queue.
// The cancellation lock protects only the cross-thread session/cancelled handoff.
// Completion invalidates the session and closes the sink before resuming its
// continuation, so callers may clean up staging files after the await returns.
final class HubTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var session: URLSession?
    private var cancelled = false
    private var continuation: CheckedContinuation<Void,Error>?
    private var handle: FileHandle?
    private let closeHandle: @Sendable (FileHandle) throws -> Void
    private let destination: URL
    private let range: Range<Int64>?
    private let total: Int64?
    private let maximum: Int64
    private let progress: @Sendable (Int64) -> Void
    private let attempt: Int
    private let started = ContinuousClock.now
    private var received: Int64 = 0
    private var accepted = false
    private var failure: Error?
    private var redirects = 0
    private var lastSpaceCheck: Int64 = 0
    private var lastProgress: Int64 = 0

    init(destination: URL,maximum: Int64,progress: @escaping @Sendable (Int64) -> Void,range: Range<Int64>? = nil,total: Int64? = nil,attempt: Int = 1,closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() }) {
        self.attempt = attempt; self.destination = destination; self.maximum = maximum; self.progress = progress; self.range = range; self.total = total; self.closeHandle = closeHandle
    }
    static func allowed(_ url: URL) -> Bool {
        guard url.scheme == "https", url.user == nil, url.password == nil,
              url.port == nil || url.port == 443, let host = url.host?.lowercased() else { return false }
        return host == "huggingface.co" || host.hasSuffix(".huggingface.co") || host == "hf.co" || host.hasSuffix(".hf.co")
    }
    static func fetch(_ url: URL,_ destination: URL,_ maximum: Int64,
                      _ progress: @escaping @Sendable (Int64) -> Void,checkpointRoot: URL? = nil) async throws {
        if maximum >= 1024 * 1024 * 1024 {
            let official = HubProfile.gemmaBF16Shard
            let exact = maximum == official.bytes && url == HubAPI.fileURL(HubProfile.gemmaBF16,official)
            if exact, let checkpointRoot, HubProfile.catalog.contains(.gemmaBF16) {
                try await checkpointedGemma(url,destination,root:checkpointRoot,shard:official,progress:progress)
                return
            }
            let chunkBytes: Int64 = (exact ? 4 : 16) * 1024 * 1024
            try await ranged(url,destination,maximum,progress,chunkBytes:chunkBytes,resumesPartialRanges:exact)
        } else {
            let metadata = url.host == "huggingface.co" && url.path.hasPrefix("/api/models") && maximum <= HubAPI.metadataLimit
            try await whole(url,destination,maximum,progress,resourceTimeout:metadata ? .seconds(20) : .seconds(7200))
        }
    }
    private static var gemmaCheckpointKey: String {
        HubProfile.gemmaBF16.revision + "-" + HubProfile.gemmaBF16Shard.sha256
    }

    private static func checkpointedGemma(_ url: URL,_ destination: URL,root: URL,shard: HubFile,
                                         progress: @escaping @Sendable (Int64) -> Void) async throws {
        let maximum = Duration.seconds(7200)
        let expires = ContinuousClock.now.advanced(by:maximum)
        try await withDeadline(maximum) {
            let checkpoint = try HubShardCheckpoint(root:root,identity:gemmaCheckpointKey,
                maximum:shard.bytes,completeSHA:shard.sha256)
            let offset = try checkpoint.seed(destination)
            if offset != 0 { progress(offset) }
            if offset == shard.bytes { return }
            // Leave bounded time for joined cleanup and the checkpoint or final SHA.
            let networkTime = ContinuousClock.now.duration(to:expires) - .seconds(90)
            guard networkTime > .zero else { throw HubFailure(deadlineMessage) }
            do {
                try await ranged(url,destination,shard.bytes,progress,chunkBytes:4 * 1024 * 1024,
                    resourceTimeout:networkTime,resumesPartialRanges:true,startingAt:offset)
            } catch {
                if !Task.isCancelled && (transient(error) || (error as? HubFailure)?.message == deadlineMessage) {
                    let original = error
                    do { try checkpoint.save(destination,after:offset) }
                    catch { throw cleanupFailure(original,"download checkpoint: " + error.localizedDescription) }
                }
                throw error
            }
            do { try checkpoint.save(destination,after:offset,completed:true) }
            catch {
                if !Task.isCancelled { try checkpoint.clear() }
                throw error
            }
        }
    }

    // A complete shard survives an ancillary-file failure. Once the snapshot
    // is published, retire its duplicate checkpoint if no other owner holds it.
    static func retireGemmaCheckpoint(root: URL) {
        let key = gemmaCheckpointKey
        let folder = root.appendingPathComponent(".resume-" + key,isDirectory:true)
        guard FileManager.default.fileExists(atPath:folder.path) else { return }
        let shard = HubProfile.gemmaBF16Shard
        guard let checkpoint = try? HubShardCheckpoint(root:root,identity:key,
            maximum:shard.bytes,completeSHA:shard.sha256) else { return }
        try? checkpoint.clear()
    }

    // One exact, revision-pinned official shard has a separate large range
    // ceiling. Its published contents still require the complete profile SHA.
    static func admitsRangeTotal(_ total: Int64, from url: URL) -> Bool {
        if (1...HubProfile.maximumFileBytes).contains(total) { return true }
        let profile = HubProfile.gemmaBF16, shard = HubProfile.gemmaBF16Shard
        return HubProfile.catalog.contains(profile) && total == shard.bytes &&
            url.absoluteString == HubAPI.fileURL(profile,shard).absoluteString
    }
    static func run(_ url: URL,_ destination: URL,_ maximum: Int64,
                    _ progress: @escaping @Sendable (Int64) -> Void,configuration: URLSessionConfiguration,
                    range: Range<Int64>? = nil,total: Int64? = nil,attempt: Int = 1,
                    closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() }) async throws {
        guard allowed(url), maximum > 0, maximum <= HubProfile.maximumFileBytes, (1...3).contains(attempt) else { throw HubFailure("Invalid transfer request.") }
        if let range {
            guard let total, admitsRangeTotal(total,from:url), range.lowerBound >= 0, range.upperBound <= total,
                  range.upperBound - range.lowerBound == maximum else { throw HubFailure("Invalid range request.") }
        } else if total != nil { throw HubFailure("Missing range request.") }
        let transfer = HubTransfer(destination:destination,maximum:maximum,progress:progress,range:range,total:total,attempt:attempt,closeHandle:closeHandle)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                transfer.start(url,configuration:configuration,continuation:continuation)
            }
        } onCancel: { transfer.cancel() }
    }
    private func start(_ url: URL,configuration: URLSessionConfiguration,continuation: CheckedContinuation<Void,Error>) {
        lock.lock()
        if cancelled { lock.unlock(); continuation.resume(throwing:CancellationError()); return }
        self.continuation = continuation
        do {
            // No overwrite or symlink following: this fresh path belongs to the
            // caller's unique staging directory and must not already exist.
            let descriptor = Darwin.open(destination.path,O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,0o600)
            guard descriptor >= 0 else { throw HubFailure("Download destination already exists or cannot be created.") }
            handle = FileHandle(fileDescriptor:descriptor,closeOnDealloc:true)
            configuration.urlCache = nil; configuration.httpCookieStorage = nil
            configuration.httpShouldSetCookies = false; configuration.urlCredentialStorage = nil
            configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
            configuration.timeoutIntervalForRequest = 60; configuration.timeoutIntervalForResource = range == nil ? 7200 : 300
            let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1
            let session = URLSession(configuration:configuration,delegate:self,delegateQueue:queue)
            self.session = session
            var request = URLRequest(url:url)
            request.setValue("identity",forHTTPHeaderField:"Accept-Encoding")
            request.setValue("Toklet/0.1",forHTTPHeaderField:"User-Agent")
            if let range { request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)",forHTTPHeaderField:"Range") }
            let task = session.dataTask(with:request)
            lock.unlock(); task.resume()
        } catch {
            var startupError = error
            self.continuation = nil
            do { if let handle { try closeHandle(handle) } }
            catch { startupError = Self.cleanupFailure(startupError,"sink close: " + error.localizedDescription) }
            handle = nil
            lock.unlock(); continuation.resume(throwing:startupError)
        }
    }
    private func cancel() {
        lock.lock(); cancelled = true; let session = session; lock.unlock()
        session?.invalidateAndCancel()
    }
    func urlSession(_ session: URLSession,task: URLSessionTask,willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        redirects += 1
        guard redirects <= 8, let url = request.url, Self.allowed(url) else {
            failure = HubFailure("Download redirected to an unsupported address.")
            completionHandler(nil); task.cancel(); return
        }
        var next = request
        next.setValue("identity",forHTTPHeaderField:"Accept-Encoding")
        if let range { next.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)",forHTTPHeaderField:"Range") }
        completionHandler(next)
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse, let url = response.url, Self.allowed(url), http.statusCode == (range == nil ? 200 : 206) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            failure = HubFailure(code == 401 || code == 403 ? "This repository requires Hugging Face access. Only public downloads are supported." : "Hugging Face request failed (HTTP \(code)). Retry when the service is available.")
            completionHandler(.cancel); return
        }
        if let range, let total {
            guard http.value(forHTTPHeaderField:"Content-Range") == "bytes \(range.lowerBound)-\(range.upperBound - 1)/\(total)",
                  http.value(forHTTPHeaderField:"Content-Length") == String(maximum),
                  response.expectedContentLength == maximum else {
                failure = HubFailure("Download range identity or length changed.")
                completionHandler(.cancel); return
            }
        }
        guard response.expectedContentLength <= maximum,
              http.value(forHTTPHeaderField:"Content-Encoding").map({ $0 == "identity" }) ?? true else {
            failure = HubFailure("Download response exceeds its size limit or uses unsupported encoding.")
            completionHandler(.cancel); return
        }
        accepted = true; completionHandler(.allow)
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive data: Data) {
        guard failure == nil else { return }
        do {
            guard accepted, Int64(data.count) <= maximum - received else { throw HubFailure("Download exceeded its declared file size.") }
            if received == 0 || received - lastSpaceCheck >= 1024 * 1024 {
                guard try ModelHub.diskSpace(destination.deletingLastPathComponent()) >= ModelHub.reserve + Int64(data.count) else { throw HubFailure("Storage became full during download. Free space and retry.") }
                lastSpaceCheck = received
            }
            try handle?.write(contentsOf:data)
            received += Int64(data.count)
            if received - lastProgress >= 1024 * 1024 { progress(received); lastProgress = received }
        } catch { failure = error; dataTask.cancel() }
    }
    func urlSession(_ session: URLSession,task: URLSessionTask,didCompleteWithError error: Error?) {
        var result = failure ?? error
        if let network = result as? URLError {
            let span = range.map { "bytes \($0.lowerBound)-\($0.upperBound - 1)/\(total!)" } ?? "whole file"
            result = HubNetworkFailure(code:network.code,detail:"Network transfer failed (NSURLErrorDomain \(network.code.rawValue); attempt \(attempt); \(span); received \(received)/\(maximum) bytes; elapsed \(started.duration(to:.now))).",partialRangeBytes:range != nil && accepted && received < maximum ? received : 0)
        }
        if result == nil && !accepted { result = HubFailure("Missing download response.") }
        if result == nil && range != nil && received != maximum { result = HubFailure("Download range was incomplete.") }
        do { if let handle { try closeHandle(handle) } }
        catch { result = Self.cleanupFailure(result,"sink close: " + error.localizedDescription) }
        handle = nil
        lock.lock(); let wasCancelled = cancelled; self.session = nil; lock.unlock()
        session.finishTasksAndInvalidate()
        let continuation = continuation; self.continuation = nil
        if let cleanup = result as? HubCleanupFailure {
            let failure = wasCancelled ? Self.combineCleanup(HubFailure("Download cancelled."),cleanup) : cleanup
            continuation?.resume(throwing:failure)
        } else if wasCancelled { continuation?.resume(throwing:CancellationError()) }
        else if let result { continuation?.resume(throwing:result) }
        else { progress(received); continuation?.resume() }
    }
}


extension HubTransfer {
    // Model discovery may tolerate a failed metadata page, but a failed close,
    // join or temporary-file removal must survive cancellation and every sibling.
    static func metadataCleanupFailure(_ primary: Error?,_ cause: String) -> any Error {
        cleanupFailure(primary,cause)
    }
    static func mergeCleanupFailure(_ primary: Error?,_ candidate: Error) -> (any Error)? {
        guard let cleanup = candidate as? HubCleanupFailure else { return nil }
        return combineCleanup(primary,cleanup)
    }
    private static func cleanupFailure(_ primary: Error?,_ cause: String) -> HubCleanupFailure {
        let previous = (primary as? HubCleanupFailure)?.causes ?? primary.map { ["original: " + $0.localizedDescription] } ?? []
        return HubCleanupFailure(causes:previous + [cause])
    }
    private static func combineCleanup(_ primary: Error?,_ cleanup: HubCleanupFailure) -> HubCleanupFailure {
        let previous = (primary as? HubCleanupFailure)?.causes ?? primary.map { ["original: " + $0.localizedDescription] } ?? []
        return HubCleanupFailure(causes:previous + cleanup.causes)
    }
    // Both range batches and the deadline race must observe later sibling
    // cleanup errors after cancellation. Ordinary sibling cancellation is expected.
    private static func finishGroup<T: Sendable>(_ group: inout ThrowingTaskGroup<T,any Error>,primary: Error?) async throws {
        group.cancelAll()
        var result = primary
        while true {
            do { if try await group.next() == nil { break } }
            catch let cleanup as HubCleanupFailure { result = combineCleanup(result,cleanup) }
            catch { /* The first operation/deadline error remains authoritative. */ }
        }
        if let result { throw result }
    }

    // Each child owns one configuration copy. It is never read or mutated by
    // another child; the wrapper crosses only this coarse structured-task boundary.
    private struct SessionOptions: @unchecked Sendable { let value: URLSessionConfiguration }

    // Large immutable files use four bounded streams. A connection loss
    // allows at most three attempts for the current chunk. The exact official Gemma
    // shard uses 4 MiB chunks and retains accepted prefixes within this call.
    // ModelHub still authenticates the complete file before publication.
    static func ranged(_ url: URL,_ destination: URL,_ total: Int64,
                       _ progress: @escaping @Sendable (Int64) -> Void,
                       chunkBytes: Int64 = 16 * 1024 * 1024,parallelism: Int = 4,
                       configuration: URLSessionConfiguration = .ephemeral,
                       closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() },
                       resourceTimeout: Duration = .seconds(7200),resumesPartialRanges: Bool = false,startingAt: Int64 = 0) async throws {
        let options = SessionOptions(value:configuration.copy() as! URLSessionConfiguration)
        try await withDeadline(resourceTimeout) {
            try await assembleRanges(url,destination,total,progress,chunkBytes:chunkBytes,parallelism:parallelism,
                                     configuration:options.value,closeHandle:closeHandle,resumesPartialRanges:resumesPartialRanges,startingAt:startingAt)
        }
    }

    // Metadata and smaller files need the same transient-only attempt bound as
    // ranges. A retry starts a new exclusive sink only after the previous
    // continuation has joined URLSession completion and closed its owned sink.
    static func whole(_ url: URL,_ destination: URL,_ maximum: Int64,
                      _ progress: @escaping @Sendable (Int64) -> Void,
                      configuration: URLSessionConfiguration = .ephemeral,
                      closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() },
                      resourceTimeout: Duration = .seconds(7200)) async throws {
        let options = SessionOptions(value:configuration.copy() as! URLSessionConfiguration)
        try await withDeadline(resourceTimeout) {
            for attempt in 0..<3 {
                try Task.checkCancellation()
                do {
                    try await run(url,destination,maximum,progress,configuration:options.value.copy() as! URLSessionConfiguration,attempt:attempt + 1,closeHandle:closeHandle)
                    return
                } catch {
                    if error is HubCleanupFailure { throw error }
                    try Task.checkCancellation()
                    guard attempt < 2, transient(error) else { throw error }
                    // A transient network failure comes from the session, after this
                    // call created the sink. Startup/collision errors never retry
                    // or remove an existing destination.
                    let original = error
                    do { try FileManager.default.removeItem(at:destination) }
                    catch { throw cleanupFailure(original,"retry sink removal: " + error.localizedDescription) }
                    try await Task.sleep(for:.milliseconds(250 * (attempt + 1)))
                }
            }
        }
    }

    // Cold typed diagnostics without NSError userInfo or signed redirect URLs.
    // Cleanup and HTTP/file failures deliberately do not become transport errors.
    static func networkFailureCode(_ error: Error) -> URLError.Code? {
        (error as? HubNetworkFailure)?.code ?? (error as? URLError)?.code
    }
    private static func transient(_ error: Error) -> Bool {
        let code = networkFailureCode(error)
        guard let code else { return false }
        return [.timedOut,.networkConnectionLost,.cannotConnectToHost,.cannotFindHost,.dnsLookupFailed,.notConnectedToInternet].contains(code)
    }
    private static let deadlineMessage = "Transfer exceeded its resource deadline."
    private static func withDeadline(_ duration: Duration,_ operation: @escaping @Sendable () async throws -> Void) async throws {
        guard duration > .zero, duration <= .seconds(7200) else { throw HubFailure("Invalid transfer deadline.") }
        try await withThrowingTaskGroup(of:Void.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(for:duration)
                throw HubFailure(deadlineMessage)
            }
            var primary: Error?
            do { _ = try await group.next() } catch { primary = error }
            try await finishGroup(&group,primary:primary)
        }
    }

    private static func assembleRanges(_ url: URL,_ destination: URL,_ total: Int64,
                       _ progress: @escaping @Sendable (Int64) -> Void,
                       chunkBytes: Int64 = 16 * 1024 * 1024,parallelism: Int = 4,
                       configuration: URLSessionConfiguration = .ephemeral,
                       closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() },resumesPartialRanges: Bool,startingAt: Int64) async throws {
        guard allowed(url), admitsRangeTotal(total,from:url), (0..<total).contains(startingAt),
              chunkBytes > 0, chunkBytes <= 16 * 1024 * 1024, (1...4).contains(parallelism) else {
            throw HubFailure("Invalid ranged transfer request.")
        }
        try Task.checkCancellation()
        let flags = startingAt == 0 ? O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW : O_WRONLY | O_NOFOLLOW | O_NONBLOCK
        let descriptor = Darwin.open(destination.path,flags,0o600)
        guard descriptor >= 0 else { throw HubFailure("Download destination already exists or cannot be created.") }
        let sink = FileHandle(fileDescriptor:descriptor,closeOnDealloc:true)
        defer { try? sink.close() }
        if startingAt != 0 {
            var info = stat()
            guard fstat(descriptor,&info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size == startingAt else {
                throw HubFailure("Download prefix extent differs.")
            }
            try sink.seek(toOffset:UInt64(startingAt))
        }
        let directory = destination.deletingLastPathComponent().appendingPathComponent(".ranges-" + UUID().uuidString,isDirectory:true)
        guard Darwin.mkdir(directory.path,0o700) == 0 else {
            let primary = HubFailure("Could not create exclusive range directory.")
            do { try sink.close() } catch { throw cleanupFailure(primary,"assembly sink close: " + error.localizedDescription) }
            throw primary
        }
        // The unique directory is owned by this call. Structured task groups join
        // every child (including cancelled transfers) before cleanup reaches here.
        var primary: Error?
        do {
            var committed = startingAt
            while committed < total {
                try Task.checkCancellation()
                let start = committed
                let pieces = try await withThrowingTaskGroup(of:[(Int64,URL,Int64)].self) { group in
                    for index in 0..<parallelism {
                        let lower = start + Int64(index) * chunkBytes
                        if lower >= total { break }
                        let upper = min(total,lower + chunkBytes)
                        let options = SessionOptions(value:configuration.copy() as! URLSessionConfiguration)
                        group.addTask {
                            var cursor = lower
                            var pieces: [(Int64,URL,Int64)] = []
                            for attempt in 0..<3 {
                                try Task.checkCancellation()
                                let path = directory.appendingPathComponent("part-\(lower)-\(attempt)")
                                do {
                                    try await run(url,path,upper - cursor,{ _ in },configuration:options.value.copy() as! URLSessionConfiguration,
                                                  range:cursor..<upper,total:total,attempt:attempt + 1,closeHandle:closeHandle)
                                    pieces.append((cursor,path,upper - cursor))
                                    return pieces
                                } catch {
                                    let canRetry = attempt < 2 && transient(error)
                                    let received = (error as? HubNetworkFailure)?.partialRangeBytes ?? 0
                                    if resumesPartialRanges && canRetry && received > 0 {
                                        // The accepted 206 declared this exact interval. The sink
                                        // has closed. Keep only its bounded prefix and request the rest.
                                        pieces.append((cursor,path,received)); cursor += received
                                    } else {
                                        let original = error
                                        do { if FileManager.default.fileExists(atPath:path.path) { try FileManager.default.removeItem(at:path) } }
                                        catch { throw cleanupFailure(original,"chunk removal: " + error.localizedDescription) }
                                    }
                                    if error is HubCleanupFailure { throw error }
                                    try Task.checkCancellation()
                                    guard canRetry else { throw error }
                                    try await Task.sleep(for:.milliseconds(250 * (attempt + 1)))
                                }
                            }
                            throw HubFailure("Range attempts exhausted.")
                        }
                    }
                    var pieces: [(Int64,URL,Int64)] = []
                    do { for try await chunk in group { pieces.append(contentsOf:chunk) } }
                    catch { try await finishGroup(&group,primary:error) }
                    return pieces.sorted { $0.0 < $1.0 }
                }
                for (lower,path,bytes) in pieces {
                    try Task.checkCancellation()
                    guard lower == committed else { throw HubFailure("Noncontiguous range assembly.") }
                    guard try ModelHub.diskSpace(destination.deletingLastPathComponent()) >= ModelHub.reserve + bytes else {
                        throw HubFailure("Storage became full during range assembly. Free space and retry.")
                    }
                    let input = try FileHandle(forReadingFrom:path)
                    var pieceError: Error?
                    do {
                        while let data = try input.read(upToCount:64 * 1024), !data.isEmpty {
                            try Task.checkCancellation()
                            guard Int64(data.count) <= lower + bytes - committed else { throw HubFailure("Range assembly exceeded piece limit.") }
                            try sink.write(contentsOf:data); committed += Int64(data.count)
                        }
                        guard committed == lower + bytes else { throw HubFailure("Range assembly piece was truncated.") }
                    } catch { pieceError = error }
                    do { try input.close() } catch { pieceError = cleanupFailure(pieceError,"assembly input close: " + error.localizedDescription) }
                    if let pieceError { throw pieceError }
                    do { try FileManager.default.removeItem(at:path) }
                    catch { throw cleanupFailure(nil,"assembled chunk removal: " + error.localizedDescription) }
                    progress(committed)
                }
            }
            try Task.checkCancellation()
        } catch { primary = error }
        do { try sink.close() } catch { primary = cleanupFailure(primary,"assembly sink close: " + error.localizedDescription) }
        do { try FileManager.default.removeItem(at:directory) }
        catch { primary = cleanupFailure(primary,"range staging removal: " + error.localizedDescription) }
        if let primary { throw primary }
    }
}

/// Unpublished bytes for one exact shard. The pointer moves only after the new
/// prefix is complete and locally hashed. The caller still verifies full SHA.
final class HubShardCheckpoint {
    private struct Prefix: Codable {
        let generation: UUID
        let bytes: Int64
        let digest: String
        var name: String { "prefix-\(generation.uuidString).part" }
    }

    private let folder: URL
    private let directory: Int32
    private let lock: Int32
    private let maximum: Int64
    private let completeSHA: String?

    init(root: URL,identity: String,maximum: Int64,completeSHA: String? = nil) throws {
        guard maximum > 0, identity.utf8.count <= 128,
              !identity.isEmpty && identity.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) || $0 == 45 }) else {
            throw HubFailure("Invalid download checkpoint identity.")
        }
        let folder = root.appendingPathComponent(".resume-" + identity,isDirectory:true)
        guard mkdir(folder.path,0o700) == 0 || errno == EEXIST else {
            throw HubFailure("Could not create the private download checkpoint.")
        }
        let directory = Darwin.open(folder.path,O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard directory >= 0 else { throw HubFailure("Could not open the private download checkpoint.") }
        var info = stat()
        guard fstat(directory,&info) == 0, info.st_uid == geteuid(), info.st_mode & 0o077 == 0 else {
            Darwin.close(directory)
            throw HubFailure("Download checkpoint permissions differ.")
        }
        let lock = openat(directory,"lease",O_CREAT | O_RDWR | O_NOFOLLOW | O_NONBLOCK,0o600)
        guard lock >= 0 else {
            Darwin.close(directory)
            throw HubFailure("Could not open the download checkpoint lease.")
        }
        guard fstat(lock,&info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == geteuid(), info.st_mode & 0o077 == 0, flock(lock,LOCK_EX | LOCK_NB) == 0 else {
            Darwin.close(lock); Darwin.close(directory)
            throw HubFailure("Another download owns this checkpoint.")
        }
        self.folder = folder; self.directory = directory; self.lock = lock
        self.maximum = maximum; self.completeSHA = completeSHA
    }

    deinit { Darwin.close(lock); Darwin.close(directory) }

    private var pointer: URL { folder.appendingPathComponent("prefix.json") }

    /// Copy only a consistent acknowledged checkpoint into fresh staging.
    func seed(_ destination: URL) throws -> Int64 {
        guard FileManager.default.fileExists(atPath:pointer.path) else {
            try prune(keeping:nil)
            return 0
        }
        let prefix: Prefix
        do {
            let data = try ModelHub.withRegular(pointer,maximum:1024,directory:directory) { handle,size in
                try handle.read(upToCount:Int(size)) ?? Data()
            }
            prefix = try JSONDecoder().decode(Prefix.self,from:data)
            guard (1...maximum).contains(prefix.bytes), prefix.bytes < maximum || prefix.digest == completeSHA,
                  try digest(folder.appendingPathComponent(prefix.name),expected:prefix.bytes,inDirectory:directory) == prefix.digest else {
                throw HubFailure("Saved download prefix failed its local check.")
            }
        } catch {
            try Task.checkCancellation()
            try clear()
            return 0
        }
        try prune(keeping:prefix.name)
        try duplicate(folder.appendingPathComponent(prefix.name),destination,bytes:prefix.bytes,inDirectory:directory)
        return prefix.bytes
    }

    /// Called only after the owned transfer joins and closes its output.
    func save(_ source: URL,after previous: Int64,completed: Bool = false) throws {
        guard FileManager.default.fileExists(atPath:source.path) else { return }
        let bytes = try ModelHub.withRegular(source,maximum:maximum) { _,size in size }
        guard bytes > previous, bytes < maximum || completed else { return }
        if completed && bytes != maximum { throw HubFailure("Completed download shard extent differs.") }
        try Task.checkCancellation()
        let generation = UUID()
        let next = folder.appendingPathComponent("prefix-\(generation.uuidString).part")
        var pointed = false
        defer { if !pointed { try? FileManager.default.removeItem(at:next) } }
        try duplicate(source,next,bytes:bytes)
        let localDigest = try digest(next,expected:bytes,inDirectory:directory)
        if completed && localDigest != completeSHA { throw HubFailure("Downloaded file failed its SHA-256 check.") }
        let data = try JSONEncoder().encode(Prefix(generation:generation,bytes:bytes,digest:localDigest))
        try Task.checkCancellation()
        try data.write(to:pointer,options:.atomic)
        pointed = true
        try prune(keeping:next.lastPathComponent)
    }

    func clear() throws {
        if FileManager.default.fileExists(atPath:pointer.path) { try FileManager.default.removeItem(at:pointer) }
        try prune(keeping:nil)
    }

    private func prune(keeping: String?) throws {
        for name in try FileManager.default.contentsOfDirectory(atPath:folder.path) where name != keeping {
            guard name.hasPrefix("prefix-"), name.hasSuffix(".part"),
                  UUID(uuidString:String(name.dropFirst(7).dropLast(5))) != nil else { continue }
            try FileManager.default.removeItem(at:folder.appendingPathComponent(name))
        }
    }

    private func digest(_ source: URL,expected: Int64,inDirectory: Int32) throws -> String {
        try ModelHub.withRegular(source,maximum:maximum,directory:inDirectory) { handle,bytes in
            guard bytes == expected else { throw HubFailure("Download prefix extent differs.") }
            var hasher = SHA256()
            var count: Int64 = 0
            while let data = try autoreleasepool(invoking:{ try handle.read(upToCount:256 * 1024) }), !data.isEmpty {
                try Task.checkCancellation()
                count += Int64(data.count)
                guard count <= expected else { throw HubFailure("Download prefix exceeded its size.") }
                hasher.update(data:data)
            }
            guard count == expected else { throw HubFailure("Download prefix was truncated.") }
            return hasher.finalize().map { String(format:"%02x",$0) }.joined()
        }
    }

    private func duplicate(_ source: URL,_ destination: URL,bytes: Int64,inDirectory: Int32? = nil) throws {
        try ModelHub.withRegular(source,maximum:maximum,directory:inDirectory) { input,size in
            guard size == bytes else { throw HubFailure("Download prefix extent differs.") }
            if fclonefileat(input.fileDescriptor,AT_FDCWD,destination.path,UInt32(CLONE_NOFOLLOW_ANY)) == 0 { return }
            let descriptor = Darwin.open(destination.path,O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,0o600)
            guard descriptor >= 0 else { throw HubFailure("Could not create the download prefix copy.") }
            let output = FileHandle(fileDescriptor:descriptor,closeOnDealloc:true)
            defer { try? output.close() }
            var count: Int64 = 0
            while let data = try autoreleasepool(invoking:{ try input.read(upToCount:256 * 1024) }), !data.isEmpty {
                try Task.checkCancellation(); count += Int64(data.count)
                guard count <= bytes else { throw HubFailure("Download prefix copy exceeded its size.") }
                try output.write(contentsOf:data)
            }
            guard count == bytes else { throw HubFailure("Download prefix copy was truncated.") }
            try output.close()
        }
    }
}

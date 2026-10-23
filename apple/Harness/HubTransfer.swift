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
                      _ progress: @escaping @Sendable (Int64) -> Void) async throws {
        if maximum >= 1024 * 1024 * 1024 {
            try await ranged(url,destination,maximum,progress)
        } else {
            try await whole(url,destination,maximum,progress)
        }
    }
    static func run(_ url: URL,_ destination: URL,_ maximum: Int64,
                    _ progress: @escaping @Sendable (Int64) -> Void,configuration: URLSessionConfiguration,
                    range: Range<Int64>? = nil,total: Int64? = nil,attempt: Int = 1,
                    closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() }) async throws {
        guard allowed(url), maximum > 0, maximum <= HubProfile.maximumFileBytes, (1...3).contains(attempt) else { throw HubFailure("Invalid transfer request.") }
        if let range {
            guard let total, total > 0, total <= HubProfile.maximumFileBytes, range.lowerBound >= 0, range.upperBound <= total,
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
            result = HubNetworkFailure(code:network.code,detail:"Network transfer failed (NSURLErrorDomain \(network.code.rawValue); attempt \(attempt); \(span); received \(received)/\(maximum) bytes; elapsed \(started.duration(to:.now))).")
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

    // Large immutable files use four bounded 16 MiB streams. A connection loss
    // retries only that chunk, at most three attempts. Final full-file SHA remains
    // ModelHub's publication gate; successful ranges alone never publish a model.
    static func ranged(_ url: URL,_ destination: URL,_ total: Int64,
                       _ progress: @escaping @Sendable (Int64) -> Void,
                       chunkBytes: Int64 = 16 * 1024 * 1024,parallelism: Int = 4,
                       configuration: URLSessionConfiguration = .ephemeral,
                       closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() },
                       resourceTimeout: Duration = .seconds(7200)) async throws {
        let options = SessionOptions(value:configuration.copy() as! URLSessionConfiguration)
        try await withDeadline(resourceTimeout) {
            try await assembleRanges(url,destination,total,progress,chunkBytes:chunkBytes,parallelism:parallelism,
                                     configuration:options.value,closeHandle:closeHandle)
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

    private static func transient(_ error: Error) -> Bool {
        let code = (error as? HubNetworkFailure)?.code ?? (error as? URLError)?.code
        guard let code else { return false }
        return [.timedOut,.networkConnectionLost,.cannotConnectToHost,.cannotFindHost,.dnsLookupFailed,.notConnectedToInternet].contains(code)
    }
    private static func withDeadline(_ duration: Duration,_ operation: @escaping @Sendable () async throws -> Void) async throws {
        guard duration > .zero, duration <= .seconds(7200) else { throw HubFailure("Invalid transfer deadline.") }
        try await withThrowingTaskGroup(of:Void.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(for:duration)
                throw HubFailure("Download exceeded the two-hour resource deadline.")
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
                       closeHandle: @escaping @Sendable (FileHandle) throws -> Void = { try $0.close() }) async throws {
        guard allowed(url), total > 0, total <= HubProfile.maximumFileBytes,
              chunkBytes > 0, chunkBytes <= 16 * 1024 * 1024, (1...4).contains(parallelism) else {
            throw HubFailure("Invalid ranged transfer request.")
        }
        try Task.checkCancellation()
        let descriptor = Darwin.open(destination.path,O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,0o600)
        guard descriptor >= 0 else { throw HubFailure("Download destination already exists or cannot be created.") }
        let sink = FileHandle(fileDescriptor:descriptor,closeOnDealloc:true)
        defer { try? sink.close() }
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
            var committed: Int64 = 0
            while committed < total {
                try Task.checkCancellation()
                let start = committed
                let pieces = try await withThrowingTaskGroup(of:(Int64,URL).self) { group in
                    for index in 0..<parallelism {
                        let lower = start + Int64(index) * chunkBytes
                        if lower >= total { break }
                        let upper = min(total,lower + chunkBytes)
                        let options = SessionOptions(value:configuration.copy() as! URLSessionConfiguration)
                        group.addTask {
                            for attempt in 0..<3 {
                                try Task.checkCancellation()
                                let path = directory.appendingPathComponent("part-\(lower)-\(attempt)")
                                do {
                                    try await run(url,path,upper - lower,{ _ in },configuration:options.value.copy() as! URLSessionConfiguration,
                                                  range:lower..<upper,total:total,attempt:attempt + 1,closeHandle:closeHandle)
                                    return (lower,path)
                                } catch {
                                    var cause = error
                                    do { if FileManager.default.fileExists(atPath:path.path) { try FileManager.default.removeItem(at:path) } }
                                    catch { cause = cleanupFailure(cause,"chunk removal: " + error.localizedDescription) }
                                    if cause is HubCleanupFailure { throw cause }
                                    try Task.checkCancellation()
                                    guard attempt < 2, transient(cause) else { throw cause }
                                    try await Task.sleep(for:.milliseconds(250 * (attempt + 1)))
                                }
                            }
                            throw HubFailure("Range attempts exhausted.")
                        }
                    }
                    var pieces: [(Int64,URL)] = []
                    do { for try await piece in group { pieces.append(piece) } }
                    catch { try await finishGroup(&group,primary:error) }
                    return pieces.sorted { $0.0 < $1.0 }
                }
                for (lower,path) in pieces {
                    try Task.checkCancellation()
                    guard lower == committed else { throw HubFailure("Noncontiguous range assembly.") }
                    guard try ModelHub.diskSpace(destination.deletingLastPathComponent()) >= ModelHub.reserve + min(chunkBytes,total - lower) else {
                        throw HubFailure("Storage became full during range assembly. Free space and retry.")
                    }
                    let input = try FileHandle(forReadingFrom:path)
                    var pieceError: Error?
                    do {
                        while let data = try input.read(upToCount:64 * 1024), !data.isEmpty {
                            try Task.checkCancellation()
                            guard Int64(data.count) <= total - committed else { throw HubFailure("Range assembly exceeded file limit.") }
                            try sink.write(contentsOf:data); committed += Int64(data.count)
                        }
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

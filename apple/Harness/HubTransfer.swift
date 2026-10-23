import Foundation
import Darwin

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
    private let destination: URL
    private let maximum: Int64
    private let progress: @Sendable (Int64) -> Void
    private var received: Int64 = 0
    private var accepted = false
    private var failure: Error?
    private var redirects = 0
    private var lastSpaceCheck: Int64 = 0
    private var lastProgress: Int64 = 0

    init(destination: URL,maximum: Int64,progress: @escaping @Sendable (Int64) -> Void) {
        self.destination = destination; self.maximum = maximum; self.progress = progress
    }
    static func allowed(_ url: URL) -> Bool {
        guard url.scheme == "https", url.user == nil, url.password == nil,
              url.port == nil || url.port == 443, let host = url.host?.lowercased() else { return false }
        return host == "huggingface.co" || host.hasSuffix(".huggingface.co") || host == "hf.co" || host.hasSuffix(".hf.co")
    }
    static func fetch(_ url: URL,_ destination: URL,_ maximum: Int64,
                      _ progress: @escaping @Sendable (Int64) -> Void) async throws {
        try await run(url,destination,maximum,progress,configuration:.ephemeral)
    }
    static func run(_ url: URL,_ destination: URL,_ maximum: Int64,
                    _ progress: @escaping @Sendable (Int64) -> Void,configuration: URLSessionConfiguration) async throws {
        guard allowed(url), maximum > 0, maximum <= 1024 * 1024 * 1024 else { throw HubFailure("Invalid transfer request.") }
        let transfer = HubTransfer(destination:destination,maximum:maximum,progress:progress)
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
            configuration.timeoutIntervalForRequest = 60; configuration.timeoutIntervalForResource = 7200
            let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1
            let session = URLSession(configuration:configuration,delegate:self,delegateQueue:queue)
            self.session = session
            var request = URLRequest(url:url)
            request.setValue("identity",forHTTPHeaderField:"Accept-Encoding")
            request.setValue("Toklet/0.1",forHTTPHeaderField:"User-Agent")
            let task = session.dataTask(with:request)
            lock.unlock(); task.resume()
        } catch {
            self.continuation = nil; try? handle?.close(); handle = nil
            lock.unlock(); continuation.resume(throwing:error)
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
        completionHandler(request)
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse, let url = response.url, Self.allowed(url), http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            failure = HubFailure(code == 401 || code == 403 ? "This repository requires Hugging Face access. Only public downloads are supported." : "Hugging Face request failed (HTTP \(code)). Retry when the service is available.")
            completionHandler(.cancel); return
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
        if result == nil && !accepted { result = HubFailure("Missing download response.") }
        do { try handle?.close() } catch { if result == nil { result = error } }
        handle = nil
        lock.lock(); let wasCancelled = cancelled; self.session = nil; lock.unlock()
        session.finishTasksAndInvalidate()
        let continuation = continuation; self.continuation = nil
        if wasCancelled { continuation?.resume(throwing:CancellationError()) }
        else if let result { continuation?.resume(throwing:result) }
        else { progress(received); continuation?.resume() }
    }
}

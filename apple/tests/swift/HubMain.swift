import Foundation
import CryptoKit
import Darwin

private func require(_ condition: @autoclosure () throws -> Bool,_ message: String) throws {
    if try !condition() { throw HubFailure("Test failed: " + message) }
}
private func rejects(_ label: String,_ operation: () async throws -> Void) async throws {
    do { try await operation() } catch { return }
    throw HubFailure("Expected rejection: " + label)
}
private actor Started {
    private var ready = false
    private var waiter: CheckedContinuation<Void,Never>?
    func signal() { ready = true; waiter?.resume(); waiter = nil }
    func wait() async { if !ready { await withCheckedContinuation { waiter = $0 } } }
}

// Synchronous URLProtocol responses exercise the actual URLSession delegate,
// headers and bounded file sink without network access or an HTTP server.
private final class HubProtocol: URLProtocol, @unchecked Sendable {
    final class Plan: @unchecked Sendable {
        let lock = NSLock()
        private var value: (Int,[String:String],Data) = (200,[:],Data())
        func set(_ code: Int,_ headers: [String:String],_ data: Data) { lock.lock(); value = (code,headers,data); lock.unlock() }
        func get() -> (Int,[String:String],Data) { lock.lock(); defer { lock.unlock() }; return value }
    }
    static let plan = Plan()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (code,headers,data) = Self.plan.get()
        let response = HTTPURLResponse(url:request.url!,statusCode:code,httpVersion:"HTTP/1.1",headerFields:headers)!
        client?.urlProtocol(self,didReceive:response,cacheStoragePolicy:.notAllowed)
        client?.urlProtocol(self,didLoad:data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

private final class SlowHubProtocol: URLProtocol, @unchecked Sendable {
    final class State: @unchecked Sendable {
        private let lock = NSLock(); private var stopped = false
        func stop() { lock.lock(); stopped = true; lock.unlock() }
        func isStopped() -> Bool { lock.lock(); defer { lock.unlock() }; return stopped }
    }
    static let state = State()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let response = HTTPURLResponse(url:request.url!,statusCode:200,httpVersion:"HTTP/1.1",headerFields:["Content-Length":"2097152"])!
        client?.urlProtocol(self,didReceive:response,cacheStoragePolicy:.notAllowed)
        client?.urlProtocol(self,didLoad:Data(repeating:1,count:1024 * 1024))
        DispatchQueue.global().asyncAfter(deadline:.now() + 0.25) { [self] in
            guard !Self.state.isStopped() else { return }
            client?.urlProtocol(self,didLoad:Data(repeating:2,count:1024 * 1024))
            client?.urlProtocolDidFinishLoading(self)
        }
    }
    override func stopLoading() { Self.state.stop() }
}

// Tiny deterministic range server; no network or multi-GB allocation.
private final class RangeHubProtocol: URLProtocol, @unchecked Sendable {
    final class Plan: @unchecked Sendable {
        let lock = NSLock()
        private var mode = "good"
        private var payload = Data()
        private var attempts: [Int:Int] = [:]
        private var invalidHeader = false
        private var signal: (@Sendable () -> Void)?
        private var redirected: URLRequest?
        func set(_ mode: String,_ payload: Data,signal: (@Sendable () -> Void)? = nil) {
            lock.lock(); defer { lock.unlock() }
            self.mode = mode; self.payload = payload; attempts = [:]; invalidHeader = false; self.signal = signal
        }
        func response(_ request: URLRequest) -> (Int,[String:String],Data,Bool,Bool) {
            lock.lock(); defer { lock.unlock() }
            let fields = (request.value(forHTTPHeaderField:"Range") ?? "").replacingOccurrences(of:"bytes=",with:"").split(separator:"-")
            guard fields.count == 2, let lower = Int(fields[0]), let upper = Int(fields[1]), lower >= 0, upper >= lower, upper < payload.count else {
                invalidHeader = true; return (400,[:],Data(),false,false)
            }
            if request.value(forHTTPHeaderField:"Accept-Encoding") != "identity" { invalidHeader = true }
            attempts[lower,default:0] += 1
            let attempt = attempts[lower]!
            var headers = ["Content-Length":String(upper - lower + 1),"Content-Range":"bytes \(lower)-\(upper)/\(payload.count)"]
            var body = payload.subdata(in:lower..<(upper + 1)); var code = 206
            if mode == "wrong-start" { headers["Content-Range"] = "bytes \(lower + 1)-\(upper)/\(payload.count)" }
            if mode == "wrong-total" { headers["Content-Range"] = "bytes \(lower)-\(upper)/\(payload.count + 1)" }
            if mode == "conflict" { headers["Content-Range"]! += ", bytes \(lower)-\(upper)/\(payload.count)" }
            if mode == "encoding" { headers["Content-Encoding"] = "gzip" }
            if mode == "wrong-length" { headers["Content-Length"] = String(body.count + 1) }
            if mode == "duplicate-length" { headers["Content-Length"] = "\(body.count), \(body.count)" }
            if mode == "conflicting-length" { headers["Content-Length"] = "\(body.count), \(body.count + 1)" }
            if mode == "truncated" { body = body.prefix(max(0,body.count - 1)) }
            if mode == "overlong" { body.append(0) }
            if mode == "ignored" { code = 200 }
            if mode == "denied" { code = 403 }
            return (code,headers,body,mode == "always-loss" || (mode == "one-loss" && lower == 0 && attempt == 1),mode == "slow")
        }
        func started() { lock.lock(); let signal = signal; lock.unlock(); signal?() }
        func valid() -> Bool { lock.lock(); defer { lock.unlock() }; return !invalidHeader }
        func counts() -> [Int:Int] { lock.lock(); defer { lock.unlock() }; return attempts }
        func redirect(_ request: URLRequest?) { lock.lock(); redirected = request; lock.unlock() }
        func redirectValue() -> URLRequest? { lock.lock(); defer { lock.unlock() }; return redirected }
    }
    static let plan = Plan()
    private let lock = NSLock()
    private var stopped = false
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (code,headers,data,fail,slow) = Self.plan.response(request)
        let response = HTTPURLResponse(url:request.url!,statusCode:code,httpVersion:"HTTP/1.1",headerFields:headers)!
        client?.urlProtocol(self,didReceive:response,cacheStoragePolicy:.notAllowed)
        if fail {
            client?.urlProtocol(self,didLoad:data.prefix(2))
            client?.urlProtocol(self,didFailWithError:URLError(.networkConnectionLost)); return
        }
        if slow {
            client?.urlProtocol(self,didLoad:data.prefix(2)); Self.plan.started()
            DispatchQueue.global().asyncAfter(deadline:.now() + 0.25) { [self] in
                lock.lock(); let active = !stopped; lock.unlock()
                guard active else { return }
                client?.urlProtocol(self,didLoad:data.dropFirst(2)); client?.urlProtocolDidFinishLoading(self)
            }
        } else {
            client?.urlProtocol(self,didLoad:data); client?.urlProtocolDidFinishLoading(self)
        }
    }
    override func stopLoading() { lock.lock(); stopped = true; lock.unlock() }
}

@main
struct HubMain {
    static func main() async throws {
        if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--network" {
            let root = URL(fileURLWithPath:CommandLine.arguments[2]); let hub = ModelHub(root:root)
            let rows = try await hub.search("SmolLM2-135M")
            try require(rows.contains { $0.id == HubProfile.smol.repository },"real search includes qualified model")
            let folder = try await hub.download(.smol,admission:admission,progress:{ _,_ in })
            let verified = try await hub.verified(.smol)
            try require(verified == folder,"network snapshot verification")
            print("TOKLET_HUB_NETWORK_PASS repository=\(HubProfile.smol.repository) revision=\(HubProfile.smol.revision) bytes=\(HubProfile.smol.totalBytes)")
            return
        }
        try await checks()
        print("TOKLET_HUB_CHECKS_PASS")
    }
    static let admission = HubAdmission(unifiedGPU:true,physicalMemory:6 * 1024 * 1024 * 1024,availableMemory:1024 * 1024 * 1024)
    static func checks() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletHubCheck-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:root) }
        let bytes = Data("fixture model 🧪".utf8)
        let file = HubFile(name:"model.safetensors",bytes:Int64(bytes.count),sha256:SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
        let profile = HubProfile(repository:"fixture/model",revision:String(repeating:"a",count:40),files:[file])
        let inventory = try JSONSerialization.data(withJSONObject:["sha":profile.revision,"siblings":[["rfilename":file.name,"size":file.bytes,"lfs":["size":file.bytes,"sha256":file.sha256]]]])
        let good: HubFetch = { url,destination,_,progress in
            try (url.path.contains("/api/") ? inventory : bytes).write(to:destination,options:.withoutOverwriting)
            progress(url.path.contains("/api/") ? Int64(inventory.count) : Int64(bytes.count))
        }
        let hub = ModelHub(root:root.appendingPathComponent("models"),fetch:good,freeBytes:{ _ in 10 * 1024 * 1024 * 1024 })
        let folder = try await hub.download(profile,admission:admission,progress:{ _,_ in })
        let verified = try await hub.verified(profile)
        try require(verified == folder,"published and verified")
        let repeated = try await hub.download(profile,admission:admission,progress:{ _,_ in })
        try require(repeated == folder,"idempotent existing snapshot")
        let receipt = try Data(contentsOf:folder.appendingPathComponent("snapshot.json"))
        try Data("corrupt".utf8).write(to:folder.appendingPathComponent(file.name))
        try await rejects("corrupted existing snapshot") { _ = try await hub.download(profile,admission:admission,progress:{ _,_ in }) }
        try require(try Data(contentsOf:folder.appendingPathComponent("snapshot.json")) == receipt,"corrupt snapshot preserved")
        try bytes.write(to:folder.appendingPathComponent(file.name))
        try FileManager.default.removeItem(at:folder.appendingPathComponent(file.name))
        try FileManager.default.createSymbolicLink(at:folder.appendingPathComponent(file.name),withDestinationURL:root.appendingPathComponent("outside"))
        try await rejects("symlink file") { _ = try await hub.verified(profile) }
        try FileManager.default.removeItem(at:folder.appendingPathComponent(file.name))
        try bytes.write(to:folder.appendingPathComponent(file.name))
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("linked"),withDestinationURL:folder)
        try await rejects("symlink model") { try ModelHub.verify(root.appendingPathComponent("linked"),profile:profile) }
        let badRoot = ModelHub(root:root.appendingPathComponent("linked"),fetch:good)
        try await rejects("symlink root") { _ = try await badRoot.verified(profile) }
        let fifo = root.appendingPathComponent("fifo")
        try require(mkfifo(fifo.path,0o600) == 0,"create FIFO fixture")
        try await rejects("FIFO rejected without blocking") { try ModelHub.verifyFile(fifo,file:file) }
        try FileManager.default.removeItem(at:folder.appendingPathComponent("snapshot.json"))
        try FileManager.default.createSymbolicLink(at:folder.appendingPathComponent("snapshot.json"),withDestinationURL:fifo)
        try await rejects("receipt symlink") { _ = try await hub.verified(profile) }
        try FileManager.default.removeItem(at:folder.appendingPathComponent("snapshot.json"))
        try receipt.write(to:folder.appendingPathComponent("snapshot.json"))
        let replaced = root.appendingPathComponent("replaced")
        try bytes.write(to:replaced)
        let opened = try ModelHub.withRegular(replaced,maximum:file.bytes) { handle,_ in
            try FileManager.default.moveItem(at:replaced,to:root.appendingPathComponent("original-inode"))
            try FileManager.default.createSymbolicLink(at:replaced,withDestinationURL:fifo)
            return try handle.read(upToCount:1024)
        }
        try require(opened == bytes,"path replacement reads original descriptor only")
        let parentLink = root.appendingPathComponent("parent-link")
        let outside = root.appendingPathComponent("outside-directory")
        try FileManager.default.createDirectory(at:outside,withIntermediateDirectories:false)
        try FileManager.default.createSymbolicLink(at:parentLink,withDestinationURL:outside)
        let escaped = ModelHub(root:parentLink.appendingPathComponent("must-not-create"),fetch:good)
        try await rejects("linked parent before mkdir") { _ = try await escaped.verified(profile) }
        try require(!FileManager.default.fileExists(atPath:outside.appendingPathComponent("must-not-create").path),"no directory created through rejected link")
        for name in ["../evil","a/b","%2e%2e","..","snapshot.json"] {
            let bad = HubProfile(repository:profile.repository,revision:profile.revision,files:[HubFile(name:name,bytes:1,sha256:file.sha256)])
            try await rejects("traversal inventory") { try bad.validate() }
        }
        try await rejects("changed revision") { try HubAPI.admitInventory(Data(String(decoding:inventory,as:UTF8.self).replacingOccurrences(of:profile.revision,with:String(repeating:"b",count:40)).utf8),profile:profile) }
        try await rejects("missing file") { try HubAPI.admitInventory(Data("{\"sha\":\"\(profile.revision)\",\"siblings\":[]}".utf8),profile:profile) }
        let low = ModelHub(root:root.appendingPathComponent("low"),fetch:good,freeBytes:{ _ in 0 })
        try await rejects("low storage") { _ = try await low.download(profile,admission:admission,progress:{ _,_ in }) }
        try await rejects("memory admission") { _ = try await hub.download(profile,admission:HubAdmission(unifiedGPU:true,physicalMemory:1024,availableMemory:0),progress:{ _,_ in }) }
        try await rejects("GPU admission") { try HubAdmission(unifiedGPU:false,physicalMemory:UInt64.max,availableMemory:UInt64.max).check() }
        let interruptedRoot = root.appendingPathComponent("interrupted")
        let interrupted = ModelHub(root:interruptedRoot,fetch:{ url,destination,_,_ in
            if url.path.contains("/api/") { try inventory.write(to:destination) }
            else { try bytes.prefix(3).write(to:destination); throw HubFailure("Interrupted") }
        },freeBytes:{ _ in Int64.max / 2 })
        try await rejects("partial transfer") { _ = try await interrupted.download(profile,admission:admission,progress:{ _,_ in }) }
        try require(try FileManager.default.contentsOfDirectory(atPath:interruptedRoot.path).isEmpty,"partial cleanup")
        let started = Started(); let cancelledRoot = root.appendingPathComponent("cancelled")
        let slow = ModelHub(root:cancelledRoot,fetch:{ _,destination,_,_ in
            try bytes.write(to:destination); await started.signal(); try await Task.sleep(for:.seconds(60))
        },freeBytes:{ _ in Int64.max / 2 })
        let task = Task { try await slow.download(profile,admission:admission,progress:{ _,_ in }) }
        await started.wait()
        try await rejects("duplicate request") { _ = try await slow.download(profile,admission:admission,progress:{ _,_ in }) }
        try await rejects("search during download") { _ = try await slow.search("model") }
        task.cancel(); try await rejects("cancelled transfer") { _ = try await task.value }
        try require(try FileManager.default.contentsOfDirectory(atPath:cancelledRoot.path).isEmpty,"cancel joins and cleans stage")
        let collisionRoot = root.appendingPathComponent("collision")
        let collision = ModelHub(root:collisionRoot,fetch:{ url,destination,_,_ in
            try (url.path.contains("/api/") ? inventory : bytes).write(to:destination)
            if !url.path.contains("/api/") {
                let target = collisionRoot.appendingPathComponent(profile.folder)
                try FileManager.default.createDirectory(at:target,withIntermediateDirectories:false)
                try Data("preserve".utf8).write(to:target.appendingPathComponent("sentinel"))
            }
        },freeBytes:{ _ in Int64.max / 2 })
        try await rejects("publication collision") { _ = try await collision.download(profile,admission:admission,progress:{ _,_ in }) }
        try require(try String(contentsOf:collisionRoot.appendingPathComponent(profile.folder).appendingPathComponent("sentinel"),encoding:.utf8) == "preserve","collision preserved")
        try require(!HubTransfer.allowed(URL(string:"http://huggingface.co/x")!),"HTTP rejected")
        for url in ["https://hf.co.evil.test/x","https://evil.test/x","https://user@huggingface.co/x","https://huggingface.co:444/x"] {
            try require(!HubTransfer.allowed(URL(string:url)!),"unsafe redirect")
        }
        try require(HubTransfer.allowed(URL(string:"https://cas-bridge.xethub.hf.co/x")!),"CDN supported")
        let search = try HubAPI.searchURL("模型 & name")
        try require(URLComponents(url:search,resolvingAgainstBaseURL:false)?.queryItems?.first?.value == "模型 & name","encoded Unicode query")
        try require(try HubAPI.searchResults(Data("[{\"id\":\"org/model\"}]".utf8)).count == 1,"search decode")
        try await rejects("unsafe search repository") { _ = try HubAPI.searchResults(Data("[{\"id\":\"../model\"}]".utf8)) }
        try await rejects("oversized metadata") { _ = try HubAPI.searchResults(Data(repeating:32,count:Int(HubAPI.metadataLimit)+1)) }
        let wrongRoot = root.appendingPathComponent("wrong-hash")
        let wrong = ModelHub(root:wrongRoot,fetch:{ url,destination,_,_ in
            var value = bytes; value[0] ^= 1
            try (url.path.contains("/api/") ? inventory : value).write(to:destination)
        },freeBytes:{ _ in Int64.max / 2 })
        try await rejects("same-size wrong hash") { _ = try await wrong.download(profile,admission:admission,progress:{ _,_ in }) }
        try require(try FileManager.default.contentsOfDirectory(atPath:wrongRoot.path).isEmpty,"hash-failure cleanup")
        try JSONEncoder().encode(HubSnapshotReceipt(version:2,profile:profile)).write(to:folder.appendingPathComponent("snapshot.json"))
        try await rejects("future receipt") { _ = try await hub.verified(profile) }
        try receipt.write(to:folder.appendingPathComponent("snapshot.json"))
        final class Space: @unchecked Sendable {
            let lock = NSLock(); private var checks = 0
            func next() -> Int64 { lock.lock(); defer { lock.unlock() }; checks += 1; return checks == 1 ? Int64.max / 2 : 0 }
        }
        let counter = Space(); let fullRoot = root.appendingPathComponent("became-full")
        let full = ModelHub(root:fullRoot,fetch:good,freeBytes:{ _ in counter.next() })
        try await rejects("storage changed during operation") { _ = try await full.download(profile,admission:admission,progress:{ _,_ in }) }
        try require(try FileManager.default.contentsOfDirectory(atPath:fullRoot.path).isEmpty,"storage-change cleanup")
        let beforeRoot = root.appendingPathComponent("cancel-before-publish")
        let before = ModelHub(root:beforeRoot,fetch:{ url,destination,_,_ in
            try (url.path.contains("/api/") ? inventory : bytes).write(to:destination)
            if !url.path.contains("/api/") { withUnsafeCurrentTask { $0?.cancel() } }
        },freeBytes:{ _ in Int64.max / 2 })
        let beforeTask = Task { try await before.download(profile,admission:admission,progress:{ _,_ in }) }
        try await rejects("cancellation before publication") { _ = try await beforeTask.value }
        try require(try FileManager.default.contentsOfDirectory(atPath:beforeRoot.path).isEmpty,"pre-publication cancellation leaves no snapshot")
        let afterRoot = root.appendingPathComponent("cancel-after-publish")
        let after = ModelHub(root:afterRoot,fetch:{ url,destination,_,_ in
            try (url.path.contains("/api/") ? inventory : bytes).write(to:destination)
        },freeBytes:{ _ in Int64.max / 2 })
        let afterTask = Task { try await after.download(profile,admission:admission,progress:{ _,_ in withUnsafeCurrentTask { $0?.cancel() } }) }
        let afterFolder = try await afterTask.value
        try ModelHub.verify(afterFolder,profile:profile)
        try await largeInventoryChecks(root)
        try await transportChecks(root)
        try await rangeChecks(root)
    }
    static func largeInventoryChecks(_ root: URL) async throws {
        let hash = String(repeating:"0",count:64)
        func profile(_ sizes: [Int64]) -> HubProfile {
            HubProfile(repository:"fixture/large",revision:String(repeating:"a",count:40),
                files:sizes.enumerated().map { HubFile(name:"part-\($0.offset).safetensors",bytes:$0.element,sha256:hash) })
        }
        // No multi-GB payload is allocated: test exact 64-bit admission edges.
        try profile([HubProfile.maximumFileBytes]).validate()
        try profile([HubProfile.maximumFileBytes,2 * 1024 * 1024 * 1024]).validate()
        try await rejects("file above four GiB") { try profile([HubProfile.maximumFileBytes + 1]).validate() }
        try await rejects("inventory above six GiB") { try profile([HubProfile.maximumFileBytes,2 * 1024 * 1024 * 1024 + 1]).validate() }
        try await rejects("extreme signed file size") { try profile([Int64.max]).validate() }
        let destination = root.appendingPathComponent("large-transfer-admission")
        let configuration = URLSessionConfiguration.ephemeral; configuration.protocolClasses = [HubProtocol.self]
        HubProtocol.plan.set(403,["Content-Length":"0"],Data())
        // Reaching the owned sink proves the 64-bit maximum passes request
        // admission; the remote failure still propagates. No large-transfer claim.
        try await rejects("large admitted request remote failure") {
            try await HubTransfer.run(URL(string:"https://huggingface.co/test")!,destination,HubProfile.maximumFileBytes,{ _ in },configuration:configuration)
        }
        try require(FileManager.default.fileExists(atPath:destination.path),"large maximum admitted before HTTP failure")
        let rejected = root.appendingPathComponent("oversize-transfer")
        try await rejects("oversize rejected before creating sink") {
            try await HubTransfer.run(URL(string:"https://huggingface.co/test")!,rejected,HubProfile.maximumFileBytes + 1,{ _ in },configuration:configuration)
        }
        try require(!FileManager.default.fileExists(atPath:rejected.path),"oversize request creates no file")
    }
    static func transportChecks(_ root: URL) async throws {
        let configuration = URLSessionConfiguration.ephemeral; configuration.protocolClasses = [HubProtocol.self]
        for (index,code,length,payload,accept) in [(0,200,"4",Data("test".utf8),true),(1,403,"4",Data("deny".utf8),false),
             (2,200,"999",Data("test".utf8),false),(3,200,"4",Data("too long".utf8),false),(4,500,"4",Data("fail".utf8),false)] {
            HubProtocol.plan.set(code,["Content-Length":length],payload)
            let destination = root.appendingPathComponent("transfer-\(index)")
            let run = { try await HubTransfer.run(URL(string:"https://huggingface.co/test")!,destination,4,{ _ in },configuration:configuration) }
            if accept { try await run(); try require(try Data(contentsOf:destination) == payload,"streaming sink bytes") }
            else { try await rejects("HTTP response \(index)",run) }
            if FileManager.default.fileExists(atPath:destination.path) {
                let count = try destination.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0
                try require(count <= 4,"response sink size bound")
            }
        }
        let activeConfiguration = URLSessionConfiguration.ephemeral; activeConfiguration.protocolClasses = [SlowHubProtocol.self]
        let wrote = Started(); let activePath = root.appendingPathComponent("active-cancel")
        let active = Task {
            try await HubTransfer.run(URL(string:"https://huggingface.co/test")!,activePath,2 * 1024 * 1024,{ count in
                if count > 0 { Task { await wrote.signal() } }
            },configuration:activeConfiguration)
        }
        await wrote.wait(); active.cancel()
        try await rejects("active URLSession cancellation") { try await active.value }
        try require(SlowHubProtocol.state.isStopped(),"URLProtocol stopped before continuation completed")
        let size = try activePath.resourceValues(forKeys:[.fileSizeKey]).fileSize
        try await Task.sleep(for:.milliseconds(350))
        try require(try activePath.resourceValues(forKeys:[.fileSizeKey]).fileSize == size,"no writes after cancellation completion")
        let cancelled = Task { () throws -> Void in
            try Task.checkCancellation()
            try await HubTransfer.run(URL(string:"https://huggingface.co/test")!,root.appendingPathComponent("pre-cancel"),4,{ _ in },configuration:configuration)
        }
        cancelled.cancel(); try await rejects("early task cancellation") { try await cancelled.value }
    }
    static func rangeChecks(_ root: URL) async throws {
        let directory = root.appendingPathComponent("range-tests")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        let bytes = Data((0..<23).map(UInt8.init))
        let url = URL(string:"https://huggingface.co/test")!
        let configuration = URLSessionConfiguration.ephemeral; configuration.protocolClasses = [RangeHubProtocol.self]
        for mode in ["good","one-loss"] {
            RangeHubProtocol.plan.set(mode,bytes)
            let destination = directory.appendingPathComponent(mode)
            try await HubTransfer.ranged(url,destination,Int64(bytes.count),{ _ in },chunkBytes:5,parallelism:4,configuration:configuration)
            try require(try Data(contentsOf:destination) == bytes,"ordered range assembly after " + mode)
            try require(RangeHubProtocol.plan.valid(),"identity and exact request ranges")
            let counts = RangeHubProtocol.plan.counts()
            try require(counts.count == 5 && counts[0] == (mode == "good" ? 1 : 2) && counts.values.reduce(0,+) == (mode == "good" ? 5 : 6),"only lost chunk retried")
        }
        for mode in ["wrong-start","wrong-total","conflict","encoding","wrong-length","duplicate-length","conflicting-length","truncated","overlong","ignored","denied","always-loss"] {
            RangeHubProtocol.plan.set(mode,bytes)
            let destination = directory.appendingPathComponent(mode)
            try await rejects("range " + mode) {
                try await HubTransfer.ranged(url,destination,Int64(bytes.count),{ _ in },chunkBytes:5,parallelism:1,configuration:configuration)
            }
            let attempts = RangeHubProtocol.plan.counts()[0] ?? 0
            try require(attempts == (mode == "always-loss" ? 3 : 1),"bounded retry only for transport error " + mode)
        }
        RangeHubProtocol.plan.set("one-loss",bytes)
        do {
            try await HubTransfer.ranged(url,directory.appendingPathComponent("close-error"),Int64(bytes.count),{ _ in },chunkBytes:5,parallelism:1,configuration:configuration,
                closeHandle:{ handle in try handle.close(); throw HubFailure("Injected close error") })
            throw HubFailure("Close error unexpectedly accepted")
        } catch {
            try require(error.localizedDescription.contains("Injected close error") && error.localizedDescription.contains("original:"),"simultaneous network/close errors preserved")
        }
        try require(RangeHubProtocol.plan.counts()[0] == 1,"close failure prevents network retry")
        let existing = directory.appendingPathComponent("collision")
        try bytes.write(to:existing)
        try await rejects("range sink collision") {
            try await HubTransfer.ranged(url,existing,Int64(bytes.count),{ _ in },chunkBytes:5,configuration:configuration)
        }
        try require(try Data(contentsOf:existing) == bytes,"range collision preserves data")
        for total in [Int64(0),Int64.max] {
            let rejected = directory.appendingPathComponent("invalid-\(total)")
            try await rejects("invalid range total") { try await HubTransfer.ranged(url,rejected,total,{ _ in },configuration:configuration) }
            try require(!FileManager.default.fileExists(atPath:rejected.path),"range rejection precedes sink")
        }
        let started = Started()
        RangeHubProtocol.plan.set("slow",bytes,signal:{ Task { await started.signal() } })
        let destination = directory.appendingPathComponent("cancelled")
        let task = Task { try await HubTransfer.ranged(url,destination,Int64(bytes.count),{ _ in },chunkBytes:5,parallelism:4,configuration:configuration) }
        await started.wait(); task.cancel()
        try await rejects("active range cancellation") { try await task.value }
        let size = try destination.resourceValues(forKeys:[.fileSizeKey]).fileSize
        try await Task.sleep(for:.milliseconds(350))
        try require(try destination.resourceValues(forKeys:[.fileSizeKey]).fileSize == size,"no range writes after joined cancellation")
        try require(try FileManager.default.contentsOfDirectory(atPath:directory.path).allSatisfy { !$0.hasPrefix(".ranges-") },"all chunk directories cleaned")
        RangeHubProtocol.plan.set("slow",bytes)
        do {
            try await HubTransfer.ranged(url,directory.appendingPathComponent("deadline"),Int64(bytes.count),{ _ in },chunkBytes:5,parallelism:4,configuration:configuration,resourceTimeout:.milliseconds(20))
            throw HubFailure("Range deadline unexpectedly accepted")
        } catch let error as HubFailure { try require(error.message.contains("resource deadline"),"whole-operation deadline preserved") }
        try require(try FileManager.default.contentsOfDirectory(atPath:directory.path).allSatisfy { !$0.hasPrefix(".ranges-") },"deadline joins and cleans chunks")
        for deadline in [false,true] {
            let opened = Started()
            RangeHubProtocol.plan.set("slow",bytes,signal:{ Task { await opened.signal() } })
            let destination = directory.appendingPathComponent(deadline ? "deadline-close-error" : "cancel-close-error")
            let task = Task {
                try await HubTransfer.ranged(url,destination,Int64(bytes.count),{ _ in },chunkBytes:5,parallelism:4,configuration:configuration,
                    closeHandle:{ handle in try handle.close(); throw HubFailure("Injected cancellation close error") },
                    resourceTimeout:deadline ? .milliseconds(100) : .seconds(7200))
            }
            if !deadline { await opened.wait(); task.cancel() }
            do { try await task.value; throw HubFailure("Cleanup error unexpectedly accepted") }
            catch {
                let message = error.localizedDescription
                try require(message.contains("Injected cancellation close error") && message.contains(deadline ? "resource deadline" : "cancelled"),"cleanup cause and interruption context retained")
            }
            try require(RangeHubProtocol.plan.counts().values.allSatisfy { $0 == 1 },"interrupted close failure never retried")
            let size = try destination.resourceValues(forKeys:[.fileSizeKey]).fileSize
            try await Task.sleep(for:.milliseconds(350))
            try require(try destination.resourceValues(forKeys:[.fileSizeKey]).fileSize == size,"no writes after cleanup-error completion")
            try require(try FileManager.default.contentsOfDirectory(atPath:directory.path).allSatisfy { !$0.hasPrefix(".ranges-") },"cleanup-error children joined and chunk directory removed")
        }
        let transfer = HubTransfer(destination:directory.appendingPathComponent("redirect"),maximum:5,progress:{ _ in },range:5..<10,total:23)
        let session = URLSession(configuration:.ephemeral)
        let request = URLRequest(url:URL(string:"https://cdn.hf.co/test")!)
        let response = HTTPURLResponse(url:url,statusCode:302,httpVersion:"HTTP/1.1",headerFields:[:])!
        let pending = session.dataTask(with:url)
        transfer.urlSession(session,task:pending,willPerformHTTPRedirection:response,newRequest:request,completionHandler:{ RangeHubProtocol.plan.redirect($0) })
        try require(RangeHubProtocol.plan.redirectValue()?.value(forHTTPHeaderField:"Range") == "bytes=5-9" && RangeHubProtocol.plan.redirectValue()?.value(forHTTPHeaderField:"Accept-Encoding") == "identity","range headers survive redirect")
        session.invalidateAndCancel()
    }

}

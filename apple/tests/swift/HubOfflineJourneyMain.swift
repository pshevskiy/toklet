import Foundation
import CryptoKit
import Darwin

// Explicit ordinary-profile network/cache/controller qualification, isolated data.
@main
@MainActor
struct HubOfflineJourneyMain {
    static func check(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func record(_ event: String, _ fields: [String:Any]) throws {
        let bytes = try JSONSerialization.data(withJSONObject:["event":event,"fields":fields],options:[.sortedKeys])
        try FileHandle.standardOutput.write(contentsOf:Data(("HUB_JOURNEY " + String(decoding:bytes,as:UTF8.self) + "\n").utf8))
    }
    static func deniedSocket() throws {
        let descriptor = Darwin.socket(AF_INET,SOCK_STREAM,0)
        var operation = "socket", result = descriptor, code = errno
        if descriptor >= 0 {
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size); address.sin_family = sa_family_t(AF_INET)
            address.sin_port = UInt16(9).bigEndian; address.sin_addr.s_addr = inet_addr("127.0.0.1")
            // The address remains live/aligned for this synchronous C call.
            result = withUnsafePointer(to:&address) { pointer in
                pointer.withMemoryRebound(to:sockaddr.self,capacity:1) {
                    Darwin.connect(descriptor,$0,socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            code = errno; operation = "connect"; Darwin.close(descriptor)
        }
        try record("socket-confinement",["operation":operation,"result":result,"errno":code,"target":"127.0.0.1:9"])
        try check(result < 0 && code == EPERM,"No independent network confinement evidence")
    }
    static func workspaceBytes(_ chat: FriendlyChatStore) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(chat.workspace)
    }
    static func main() async throws {
        guard CommandLine.arguments.count == 3, ["download","offline"].contains(CommandLine.arguments[1]) else {
            throw NativeContextFailure("Expected download/offline and isolated root")
        }
        let online = CommandLine.arguments[1] == "download", root = URL(fileURLWithPath:CommandLine.arguments[2])
        if online {
            try check(!FileManager.default.fileExists(atPath:root.path),"Fresh root required")
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        } else { try check(FileManager.default.fileExists(atPath:root.path),"Downloaded root required") }
        let profile = HubProfile.smol, models = root.appendingPathComponent("Models")
        let service = ModelHub(root:models), hub = HubStore(profile:profile,hub:service)
        let run = UUID().uuidString
        let probe = root.appendingPathComponent((online ? "online-probe-" : "offline-probe-") + run + ".json")
        try check(!FileManager.default.fileExists(atPath:probe.path),"Probe destination must be fresh")
        try Data([1]).write(to:root.appendingPathComponent("probe-write-check-" + run))
        if !online { try deniedSocket() }
        var reachable = false, failure = "", networkCode: URLError.Code?, posixDenied = false
        do { try await HubTransfer.fetch(HubAPI.inventoryURL(profile),probe,HubAPI.metadataLimit,{ _ in }); reachable = true }
        catch { networkCode = HubTransfer.networkFailureCode(error); let e = error as NSError; posixDenied = e.domain == NSPOSIXErrorDomain && e.code == Int(EPERM); failure = "\(e.domain):\(e.code):\(error.localizedDescription)" }
        try record("network-probe",["online":online,"reachable":reachable,"failure":failure,"network_code":networkCode?.rawValue as Any? ?? NSNull(),"endpoint":try HubAPI.inventoryURL(profile).absoluteString])
        try check(reachable == online,"Process network availability differs from requested phase")
        if !online {
            try check(posixDenied || networkCode.map { [.notConnectedToInternet,.cannotConnectToHost,.cannotFindHost,.dnsLookupFailed,.dataNotAllowed].contains($0) } == true,"Offline probe failed for a non-network reason")
        }
        var opened: URL?
        if online {
            try check(!FileManager.default.fileExists(atPath:models.path),"Cache not empty")
            hub.downloadAndUse(profile) { opened = $0 }
        } else { hub.use(profile) { opened = $0 } }
        await hub.waitForIdle()
        guard let folder = opened else { throw NativeContextFailure("Hub open failed: \(hub.phase) / \(hub.message)") }
        try ModelHub.verify(folder,profile:profile)
        try record("verified-cache",["online":online,"repository":profile.repository,"revision":profile.revision,
            "bytes":profile.totalBytes,"files":profile.files.map { ["name":$0.name,"bytes":$0.bytes,"sha256":$0.sha256] as [String:Any] },
            "receipt_sha256":SHA256.hash(data:try Data(contentsOf:folder.appendingPathComponent("snapshot.json"))).map { String(format:"%02x",$0) }.joined()])
        let harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("Missing worker") }
        initial.close(); try check(initial.waitUntilClosed(seconds:30),"Initial close deadline")
        let worker = try NativeWorker(captureSavedPrefix:true)
        harness.worker = worker; harness.state = worker.snapshot()
        let chat = FriendlyChatStore(harness:harness,fileURL:root.appendingPathComponent(online ? "download-chat.json" : "offline-chat.json"))
        var journeyFailure: Error?
        func wait() throws -> NativeSnapshot {
            for _ in 0..<6 {
                guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Chat deadline") }
                harness.poll(); chat.observe(harness.state)
                if !worker.snapshot().busy && !chat.loadingModel && !chat.restoringPreference { return state }
            }
            throw NativeContextFailure("Chat did not settle")
        }
        func send(_ text: String) throws -> NativeSnapshot {
            chat.edit(text); try check(chat.canSend,"Send unavailable: \(chat.notice)")
            chat.sendOrStop(); return try wait()
        }
        do {
            chat.openManaged(folder,profile:profile,ifSelected:chat.workspace.selected)
            let loaded = try wait()
            guard let identity = loaded.acceptedIdentity else { throw NativeContextFailure("No accepted Smol identity: \(loaded.message)") }
            try check(identity.profile == .smol && identity.loadedLimit == 512 && chat.ready,"Wrong opened identity")
            chat.setReplyLimit(32)
            let response = try send("Say hello in one short sentence.")
            try check(response.status == 0 && response.emitted > 0 && response.chatResult?.rejected == false
                && response.chatResult?.error.isEmpty == true && [1,2].contains(response.reason)
                && response.chatResult?.text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty == false && chat.workspace.current.messages.count == 2,"Completed chat failed")
            try record("chat",["online":online,"emitted":response.emitted,"reason":response.reason,"text":response.chatResult?.text ?? ""])
            if !online {
                let badRoot = root.appendingPathComponent("PrivateBadModels"), badFolder = badRoot.appendingPathComponent(profile.folder)
                try FileManager.default.createDirectory(at:badRoot,withIntermediateDirectories:false)
                try FileManager.default.copyItem(at:folder,to:badFolder)
                let badHub = HubStore(profile:profile,hub:ModelHub(root:badRoot))
                let policy = badFolder.appendingPathComponent("tokenizer_config.json"), saved = root.appendingPathComponent("private-policy.json")
                try FileManager.default.moveItem(at:policy,to:saved)
                for defect in ["missing","mutated"] {
                    if defect == "mutated" {
                        try FileManager.default.moveItem(at:saved,to:policy)
                        let weight = badFolder.appendingPathComponent("model.safetensors")
                        let handle = try FileHandle(forUpdating:weight)
                        let byte = try handle.read(upToCount:1)!; try handle.seek(toOffset:0)
                        try handle.write(contentsOf:Data([byte[0] ^ 1])); try handle.close()
                    }
                    let before = worker.snapshot(), conversation = try workspaceBytes(chat)
                    var callback = false
                    badHub.use(profile,replacingLoadedModel:true) { url in
                        callback = true; chat.openManaged(url,profile:profile,ifSelected:chat.workspace.selected)
                    }
                    await badHub.waitForIdle()
                    let expected = defect == "missing" ? "Downloaded files changed." : "SHA-256 check."
                    try check(badHub.message.contains(expected),"Wrong cache rejection cause: \(badHub.message)")
                    let after = worker.snapshot()
                    try check(!callback && !after.busy && after.acceptedIdentity == identity
                        && after.sessionConfiguration == before.sessionConfiguration && after.savedPrefix == before.savedPrefix
                        && (try workspaceBytes(chat)) == conversation,"Bad cache changed current chat")
                    try record("cache-rejected",["defect":defect,"callback":callback,"message":badHub.message,"owner_preserved":true])
                }
                try ModelHub.verify(folder,profile:profile)
                chat.newChat(); let recovered = try send("Say hello again.")
                try check(recovered.status == 0 && recovered.emitted > 0 && [1,2].contains(recovered.reason) && recovered.chatResult?.text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty == false,"Post-rejection recovery failed")
                try record("offline-complete",["network_unavailable":true,"cache_unchanged":true,"owner_and_conversation_preserved":true,"recovery_emitted":recovered.emitted,"ui_rendered":false,"phone_qualified":false])
            }
        } catch { journeyFailure = error }
        worker.close(); try check(worker.waitUntilClosed(seconds:30),"Worker close deadline")
        if let journeyFailure { throw journeyFailure }
    }
}

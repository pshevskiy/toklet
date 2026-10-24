import SwiftUI
import UIKit
import Darwin
import CryptoKit
import Combine

// Diagnostic-only correspondence proof for the installed signed bundle. Hash
// incrementally, outside inference timing; never read the app's Documents data.
private func installedBundleProof(_ phase: String) throws {
    // Fixed signed-file names avoid comparing Foundation's /var and
    // /private/var aliases. The host independently admits the signed inventory.
    let root = Bundle.main.bundleURL
    var names = ["Info.plist","PkgInfo","Tiny/config.json","Tiny/model.safetensors",
                 "TokletHarness","_CodeSignature/CodeResources"]
    #if targetEnvironment(simulator)
    let simulator = true
    #else
    let simulator = false
    names.append("embedded.mobileprovision")
    #endif
    var files: [String:String] = [:]
    for name in names {
        let url = root.appendingPathComponent(name)
        let properties = try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey])
        guard properties.isSymbolicLink != true, properties.isRegularFile == true else {
            throw NSError(domain:"TokletBundleProof",code:2)
        }
        let handle = try FileHandle(forReadingFrom:url)
        defer { try? handle.close() }
        var hash = SHA256()
        var count = 0
        // Drain Foundation read buffers per chunk, outside inference timing.
        // Returning only EOF keeps Data within its autorelease pool.
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let data = try handle.read(upToCount:256 * 1024), !data.isEmpty else { return false }
            count += data.count
            guard count <= 256 * 1024 * 1024 else { throw NSError(domain:"TokletBundleProof",code:4) }
            hash.update(data:data)
            return true
        }) {}
        files[name] = hash.finalize().map { String(format:"%02x",$0) }.joined()
    }
    let data = try JSONSerialization.data(withJSONObject:["phase":phase,"simulator":simulator,"files":files],options:[.sortedKeys])
    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_INSTALLATION_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
}

@MainActor
extension HarnessStore {
    func lifecycle(_ phase: ScenePhase) {
        let observed = DispatchTime.now().uptimeNanoseconds
        let action: NativeLifecycleAction
        switch phase {
        case .inactive: action = .inactive
        case .background: action = .background
        case .active: action = .active
        @unknown default: action = .inactive
        }
        if let lifecycleSignals {
            lifecycleSignals.observe(action,origin:.scenePhase,applicationState:UIApplication.shared.applicationState.rawValue,observedUptimeNanoseconds:observed)
        } else if let worker { requestNativeLifecycle(action,worker:worker) }
        poll()
    }
    func memoryWarning() {
        let observed = DispatchTime.now().uptimeNanoseconds
        if let lifecycleSignals {
            lifecycleSignals.observe(.memoryWarning,origin:.systemNotification,applicationState:UIApplication.shared.applicationState.rawValue,observedUptimeNanoseconds:observed)
        } else if let worker { requestNativeLifecycle(.memoryWarning,worker:worker) }
        poll()
    }
}

@main @MainActor
struct TokletHarnessApp: App {
    @StateObject private var store: HarnessStore
    private let uiTestChat: FriendlyChatStore?
    @Environment(\.scenePhase) private var phase
    private let timer = Timer.publish(every:0.1,on:.main,in:.common).autoconnect()

    init() {
        let arguments = ProcessInfo.processInfo.arguments
        let diagnostics = ["--self-test","--measure","--quality","--tasks","--lifecycle","--state","--state-component","--chat-self-test","--hub-self-test","--hub-download-self-test"]
        let state = HarnessStore(enabled:!diagnostics.contains(where:arguments.contains),lifecycleEnabled:arguments.contains("--lifecycle") || arguments.contains("--state") || arguments.contains("--state-component"))
        _store = StateObject(wrappedValue:state)
        // UI automation exercises the production screens without reading or
        // modifying the owner's conversations. No caller-supplied file path.
        if arguments.contains("--ui-test-history") {
            UIApplication.shared.isIdleTimerDisabled = true
            uiTestChat = FriendlyChatStore(harness:state,fileURL:FileManager.default.temporaryDirectory
                .appendingPathComponent("TokletUITest-" + UUID().uuidString + ".json"))
        } else { uiTestChat = nil }
        if arguments.contains("--hub-self-test") || arguments.contains("--hub-download-self-test") {
            UIApplication.shared.isIdleTimerDisabled = true
            let fresh = arguments.contains("--hub-download-self-test")
            let root = HubStore.modelRoot
            Task.detached(priority:.userInitiated) {
                do {
                    guard (2...3).contains(arguments.count), arguments[1] == (fresh ? "--hub-download-self-test" : "--hub-self-test"),
                          let profile = HubProfile.named(arguments.count == 3 ? arguments[2] : "smol") else { throw HubFailure("Invalid hub test arguments.") }
                    try awaitNativeMeasurementForeground()
                    try installedBundleProof("before")
                    let result: HubSelfTestResult
                    if fresh { result = try await runFreshHubSelfTest(profile:profile) }
                    else { result = try await runHubSelfTest(root:root,profile:profile) }
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_HUB_CHECK_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(0)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_HUB_CHECK_ERROR=\(error)\n".utf8))
                    exit(1)
                }
            }
        } else if arguments.contains("--chat-self-test") {
            UIApplication.shared.isIdleTimerDisabled = true
            let directory = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0]
                .appendingPathComponent("M8Inputs/smollm/Model").path
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    guard arguments == [arguments[0],"--chat-self-test"] else { throw chatInputError("Invalid chat test arguments") }
                    try awaitNativeMeasurementForeground()
                    try installedBundleProof("before")
                    let result = try runNativeChatCheck(directory:directory)
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_CHAT_CHECK_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(0)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_CHAT_CHECK_ERROR=\(error)\n".utf8))
                    exit(1)
                }
            }
        } else if let position = arguments.firstIndex(of:"--state") ?? arguments.firstIndex(of:"--state-component") {
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
            let root = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0]
            let signals = state.lifecycleSignals!
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    guard position == 1, arguments.count == 4 else { throw NSError(domain:"TokletState",code:2) }
                    try awaitNativeMeasurementForeground()
                    let config = try admitLifecycleConfig(root:root,name:arguments[position+1],sha256:arguments[position+2])
                    try installedBundleProof("before")
                    let result = runNativeState(config,root:root,signals:signals,requirePrefillWitness:arguments[position] == "--state")
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_STATE_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(result.passed ? 0:1)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_STATE_ERROR=\(error)\n".utf8));exit(1)
                }
            }
        } else if let position = arguments.firstIndex(of:"--lifecycle") {
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
            let root = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0]
            let signals = state.lifecycleSignals!
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    guard position == 1, arguments.count == 4 else { throw NSError(domain:"TokletLifecycle",code:2) }
                    try awaitNativeMeasurementForeground()
                    let config = try admitLifecycleConfig(root:root,name:arguments[position+1],sha256:arguments[position+2])
                    try installedBundleProof("before")
                    let result = runNativeLifecycle(config,root:root,signals:signals,requireSystem:true) { ready in
                        let data = try JSONEncoder().encode(ready)
                        try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_LIFECYCLE_READY_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    }
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_LIFECYCLE_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(result.passed ? 0:1)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_LIFECYCLE_ERROR=\(error)\n".utf8))
                    exit(1)
                }
            }
        } else if let position = arguments.firstIndex(of:"--tasks") {
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
            let root = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0]
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    guard position == 1, arguments.count == 4 else { throw NSError(domain:"TokletTasks",code:2) }
                    try awaitNativeMeasurementForeground()
                    let config = try admitTasksConfig(root:root,name:arguments[position+1],sha256:arguments[position+2])
                    try installedBundleProof("before")
                    let result = runNativeTasks(config,root:root,onChunk:{ sample in
                        let data = try JSONEncoder().encode(sample)
                        try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_TASK_CHUNK_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    },onCase:{ sample in
                        let data = try JSONEncoder().encode(sample)
                        try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_TASK_CASE_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    })
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_TASKS_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(result.passed ? 0 : 1)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_TASKS_ERROR=\(error)\n".utf8))
                    exit(1)
                }
            }
        } else if let position = arguments.firstIndex(of:"--quality") {
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
            let root = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0]
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    guard position == 1, arguments.count == 4 else { throw NSError(domain:"TokletQuality",code:2) }
                    try awaitNativeMeasurementForeground()
                    let config = try admitQualityConfig(root:root,name:arguments[position+1],sha256:arguments[position+2])
                    try installedBundleProof("before")
                    let result = runNativeQuality(config,root:root) { sample in
                        let data = try JSONEncoder().encode(sample)
                        try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_QUALITY_SAMPLE_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    }
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_QUALITY_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(result.passed ? 0 : 1)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_QUALITY_ERROR=\(error)\n".utf8))
                    exit(1)
                }
            }
        } else if let position = arguments.firstIndex(of:"--measure") {
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
            let root = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0]
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    guard !arguments.contains("--self-test"), position + 2 < arguments.count else {
                        throw NSError(domain:"TokletMeasurement",code:2)
                    }
                    try awaitNativeMeasurementForeground()
                    let config = try admitMeasurementConfig(root:root,name:arguments[position+1],sha256:arguments[position+2])
                    try installedBundleProof("before")
                    let result = runNativeMeasurement(config,root:root) { sample in
                        let data = try JSONEncoder().encode(sample)
                        try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_MEASUREMENT_SAMPLE_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    }
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_MEASUREMENT_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(result.passed ? 0 : 1)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_MEASUREMENT_ERROR=\(error)\n".utf8))
                    exit(1)
                }
            }
        } else if arguments.contains("--self-test") {
            UIApplication.shared.isIdleTimerDisabled = true
            let mlx = arguments.contains("--mlx")
            let directory = mlx
                ? FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0].appendingPathComponent("MLXTiny").path
                : Bundle.main.url(forResource:"Tiny",withExtension:nil)?.path ?? ""
            let metal = arguments.contains("--metal") || mlx
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    try installedBundleProof("before")
                    let result = try runWorkerChecks(directory:directory,metal:metal,mlx:mlx)
                    try installedBundleProof("after")
                    let data = try JSONEncoder().encode(result)
                    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_SELFTEST_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
                    exit(0)
                } catch {
                    try? FileHandle.standardError.write(contentsOf:Data("TOKLET_SELFTEST_ERROR=\(error)\n".utf8))
                    exit(1)
                }
            }
        }
    }
    var body: some Scene {
        WindowGroup {
            Group {
                if store.diagnosticsOnly { HarnessView(store:store) }
                else { FriendlyHarnessView(store:store,chat:uiTestChat) }
            }
                .onReceive(timer) { _ in store.poll() }
                .onChange(of:phase) { _,new in store.lifecycle(new) }
                .onReceive(NotificationCenter.default.publisher(for:UIApplication.didReceiveMemoryWarningNotification)) { _ in
                    store.memoryWarning()
                }
                .onDisappear { store.worker?.close() }
        }
    }
}

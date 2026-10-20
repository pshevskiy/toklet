import SwiftUI
import UIKit
import Darwin
import CryptoKit

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
        while let data = try handle.read(upToCount:256 * 1024), !data.isEmpty {
            count += data.count
            guard count <= 256 * 1024 * 1024 else { throw NSError(domain:"TokletBundleProof",code:4) }
            hash.update(data:data)
        }
        files[name] = hash.finalize().map { String(format:"%02x",$0) }.joined()
    }
    let data = try JSONSerialization.data(withJSONObject:["phase":phase,"simulator":simulator,"files":files],options:[.sortedKeys])
    try FileHandle.standardOutput.write(contentsOf:Data(("TOKLET_INSTALLATION_JSON=" + String(decoding:data,as:UTF8.self) + "\n").utf8))
}

@MainActor
private final class HarnessStore: ObservableObject {
    let worker: NativeWorker?
    @Published var state = NativeSnapshot()
    @Published var inputError = ""

    init(enabled: Bool) {
        do { worker = enabled ? try NativeWorker() : nil }
        catch { worker = nil; inputError = error.localizedDescription }
    }
    func poll() { if let worker { state = worker.snapshot() } }
    func submit(_ command: NativeCommand) {
        inputError = ""
        if worker?.submit(command) != true { inputError = "Worker is busy or closed" }
        poll()
    }
    func lifecycle(_ phase: ScenePhase) {
        switch phase {
        case .inactive: worker?.cancel()
        case .background: worker?.interrupt(unload: false)
        case .active: break // Resume/replay is an explicit user action.
        @unknown default: worker?.cancel()
        }
        poll()
    }
}

@MainActor
private struct HarnessView: View {
    @ObservedObject var store: HarnessStore
    @State private var directory = Bundle.main.url(forResource:"Tiny",withExtension:nil)?.path ?? ""
    @State private var packed = ""
    @State private var metal = false
    @State private var capacity = "32"
    @State private var chunk = "3"
    @State private var budgetMiB = "128"
    @State private var prompt = "1,7,13,3"
    @State private var stops = ""
    @State private var outputCount = "2"
    private var idle: Bool { !store.state.busy && !store.state.closed && store.worker != nil }
    private var documents: URL { FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0] }

    private func bounded(_ value: Binding<String>, _ maximum: Int) -> Binding<String> {
        Binding(get: { value.wrappedValue }, set: { text in
            if text.utf8.count <= maximum { value.wrappedValue = text }
            else { store.inputError = "Input exceeds \(maximum) bytes" }
        })
    }
    private func tokens(_ text: String) throws -> [UInt32] {
        guard text.utf8.count <= 256 * 1024 else { throw validation("Input exceeds 256 KiB") }
        let fields = text.split { $0 == "," || $0.isWhitespace }
        guard fields.count <= 32768 else { throw validation("Too many token IDs") }
        return try fields.map {
            guard let id = UInt32($0) else { throw validation("Expected unsigned token IDs") }
            return id
        }
    }
    private func validation(_ text: String) -> NSError {
        NSError(domain:"TokletHarness",code:1,userInfo:[NSLocalizedDescriptionKey:text])
    }
    private func load() {
        do {
            guard let cap = UInt32(capacity), let size = UInt32(chunk), let mib = UInt64(budgetMiB) else { throw validation("Invalid capacity, chunk or budget") }
            let bytes = mib.multipliedReportingOverflow(by:1024 * 1024)
            guard !bytes.overflow else { throw validation("Budget overflow") }
            guard metal || packed.isEmpty else { throw validation("Packed weights require Metal") }
            var settings = NativeSettings(directory:directory)
            settings.packedDirectory = packed; settings.metal = metal
            settings.capacity = cap; settings.chunk = size; settings.numericalBudget = bytes.partialValue
            store.submit(.load(settings))
        } catch { store.inputError = error.localizedDescription }
    }
    private func generate(_ operation: UInt32) {
        do {
            guard let maximum = UInt32(outputCount), maximum <= 128 else { throw validation("Output limit must be 0 through 128") }
            let input = operation == 2 ? [] : try tokens(prompt)
            store.submit(.generate(operation:operation,tokens:input,maximum:maximum,stops:try tokens(stops)))
        } catch { store.inputError = error.localizedDescription }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Local artifacts") {
                    Text("Raw token diagnostic harness").font(.headline)
                    VStack(alignment:.leading) { Text("Model folder").font(.caption); TextField("Local path",text:bounded($directory,16384)).textInputAutocapitalization(.never).autocorrectionDisabled() }
                    HStack {
                        Button("Tiny fixture") { directory = Bundle.main.url(forResource:"Tiny",withExtension:nil)?.path ?? ""; packed = "" }
                        Button("Documents/Model") { directory = documents.appendingPathComponent("Model").path }
                    }.disabled(!idle || store.state.hasModel)
                    VStack(alignment:.leading) { Text("Packed folder (Metal Q4 v2 only)").font(.caption); TextField("Optional local path",text:bounded($packed,16384)).textInputAutocapitalization(.never).autocorrectionDisabled() }
                    Toggle("Metal",isOn:$metal)
                    HStack { Text("Capacity (tokens)"); TextField("Tokens",text:bounded($capacity,10)).keyboardType(.numberPad).multilineTextAlignment(.trailing) }
                    HStack { Text("Prefill chunk"); TextField("Tokens",text:bounded($chunk,10)).keyboardType(.numberPad).multilineTextAlignment(.trailing) }
                    HStack { Text("Numerical cap (MiB)"); TextField("MiB",text:bounded($budgetMiB,20)).keyboardType(.numberPad).multilineTextAlignment(.trailing) }
                    Button("Load") { load() }.disabled(!idle || store.state.hasModel)
                    Text("Model files use the existing local artifact format. Copy full artifacts to this app’s Documents/Model folder. The bundled fixture is for integration checks.")
                }.disabled(!idle || store.state.hasModel)
                Section("Raw token generation") {
                    VStack(alignment:.leading) { Text("Prompt or append IDs").font(.caption); TextField("IDs",text:bounded($prompt,262144),axis:.vertical).textInputAutocapitalization(.never).autocorrectionDisabled() }
                    VStack(alignment:.leading) { Text("Hidden stop IDs (optional)").font(.caption); TextField("IDs",text:bounded($stops,262144)).textInputAutocapitalization(.never).autocorrectionDisabled() }
                    HStack { Text("Maximum new IDs"); TextField("0–128",text:bounded($outputCount,10)).keyboardType(.numberPad).multilineTextAlignment(.trailing) }
                    HStack {
                        Button("Prompt") { generate(0) }
                        Button("Append") { generate(1) }
                        Button("Continue") { generate(2) }
                    }.disabled(!idle || !store.state.hasSession)
                    HStack {
                        Button("Reset") { store.submit(.reset) }.disabled(!idle || !store.state.hasSession)
                        Button("Cancel") { store.worker?.cancel() }.disabled(store.state.closed)
                    }
                    Text(store.state.ids.map(String.init).joined(separator:", ")).font(.system(.body,design:.monospaced)).textSelection(.enabled)
                }
                Section("Lifecycle") {
                    Button("Suspend session") { store.worker?.interrupt(unload:false) }.disabled(store.state.closed)
                    Button("Replay saved prefix") { store.submit(.replay) }.disabled(!idle || !store.state.hasModel || store.state.hasSession)
                    Button("Unload model") { store.worker?.interrupt(unload:true) }.disabled(store.state.closed)
                    Button("Close worker") { store.worker?.close() }.disabled(store.state.closed)
                    Text("Inactive requests cancellation. Background requests session release and retains immutable weights for explicit replay. Memory warnings request full unload and clear saved replay state after completion. Close is terminal for this worker.")
                }
                Section("State") {
                    Text(store.state.phase)
                    Text("Active history \(store.state.history), committed \(store.state.evaluated), pending \(store.state.pending)")
                    Text("Saved prefix \(store.state.savedHistory); stop reason \(store.state.reason)")
                    Text("Last numerical admission upper bound: \(store.state.admittedBytes) bytes")
                    Text("Swift buffers and framework memory are additional. Physical-device model/context limits remain unqualified.")
                    if !store.state.message.isEmpty { Text("Status \(store.state.status): \(store.state.message)").foregroundStyle(.red) }
                    if !store.inputError.isEmpty { Text(store.inputError).foregroundStyle(.red) }
                }
            }.navigationTitle("Toklet Harness")
        }
    }
}

@main @MainActor
struct TokletHarnessApp: App {
    @StateObject private var store = HarnessStore(enabled:!ProcessInfo.processInfo.arguments.contains("--self-test"))
    @Environment(\.scenePhase) private var phase
    private let timer = Timer.publish(every:0.1,on:.main,in:.common).autoconnect()

    init() {
        if ProcessInfo.processInfo.arguments.contains("--self-test") {
            UIApplication.shared.isIdleTimerDisabled = true
            let directory = Bundle.main.url(forResource:"Tiny",withExtension:nil)?.path ?? ""
            let metal = ProcessInfo.processInfo.arguments.contains("--metal")
            DispatchQueue.global(qos:.userInitiated).async {
                do {
                    try installedBundleProof("before")
                    let result = try runWorkerChecks(directory:directory,metal:metal)
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
            HarnessView(store:store)
                .onReceive(timer) { _ in store.poll() }
                .onChange(of:phase) { _,new in store.lifecycle(new) }
                .onReceive(NotificationCenter.default.publisher(for:UIApplication.didReceiveMemoryWarningNotification)) { _ in
                    store.worker?.interrupt(unload:true)
                }
                .onDisappear { store.worker?.close() }
        }
    }
}

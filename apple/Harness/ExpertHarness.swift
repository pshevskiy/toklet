import SwiftUI

// Keep platform presentation outside model/controller and numerical code.
extension View {
    @ViewBuilder func harnessLiteralInput() -> some View {
        #if os(iOS)
        self.textInputAutocapitalization(.never).autocorrectionDisabled()
        #else
        self.autocorrectionDisabled()
        #endif
    }
    @ViewBuilder func harnessNumberInput() -> some View {
        #if os(iOS)
        self.keyboardType(.numberPad)
        #else
        self
        #endif
    }
    @ViewBuilder func harnessNavigationStyle() -> some View {
        #if os(iOS)
        self.navigationBarTitleDisplayMode(.inline)
        #else
        self
        #endif
    }
    @ViewBuilder func harnessSheetSize() -> some View {
        #if os(macOS)
        self.frame(minWidth:620,minHeight:560)
        #else
        self
        #endif
    }
}

// Explicit button styling prevents a Form row from firing every button in it.
// Accessibility text sizes use one action per line.
private struct HarnessActions<Content: View>: View {
    @Environment(\.dynamicTypeSize) private var typeSize
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        Group {
            if typeSize.isAccessibilitySize { VStack(alignment:.leading,spacing:12) { content } }
            else { HStack { content } }
        }.buttonStyle(.bordered)
    }
}

@MainActor
struct HarnessView: View {
    @ObservedObject var store: HarnessStore
    @State private var directory = Bundle.main.url(forResource:"Tiny",withExtension:nil)?.path ?? ""
    @State private var packed = ""
    @State private var metal = false
    @State private var mlx = false
    @State private var capacity = "32"
    @State private var chunk = "3"
    @State private var budgetMiB = "128"
    @State private var prompt = "1,7,13,3"
    @State private var stops = ""
    @State private var outputCount = "2"
    private var idle: Bool { !store.state.busy && !store.state.closed && store.worker != nil }
    private var lifecycleHelp: String {
        #if os(iOS)
        "Inactive requests cancellation. Background requests session release and retains immutable weights for explicit replay. Memory warnings request full unload and clear saved replay state after completion. Close is terminal for this worker."
        #else
        "Switching apps keeps work running. Suspend releases the session and retains weights for explicit replay. Unload releases the model. Closing the window or quitting joins downloads and native work before saving the conversation."
        #endif
    }
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
            settings.packedDirectory = packed; settings.metal = metal; settings.mlx = mlx
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
                    VStack(alignment:.leading) { Text("Model folder").font(.caption); TextField("Local path",text:bounded($directory,16384)).harnessLiteralInput() }
                    HarnessActions {
                        Button("Tiny fixture") { directory = Bundle.main.url(forResource:"Tiny",withExtension:nil)?.path ?? ""; packed = ""; mlx = false }
                        Button("Documents/Model") { directory = documents.appendingPathComponent("Model").path }
                    }.disabled(!idle || store.state.hasModel)
                    VStack(alignment:.leading) { Text("Packed folder (Metal Q4 v2 only)").font(.caption); TextField("Optional local path",text:bounded($packed,16384)).harnessLiteralInput() }
                    Toggle("Metal",isOn:$metal)
                    Toggle("MLX-format model",isOn:$mlx)
                    if mlx { Text("Requires Metal. Leave the separate Q4 folder empty.") }
                    HStack { Text("Capacity (tokens)"); TextField("Tokens",text:bounded($capacity,10)).harnessNumberInput().multilineTextAlignment(.trailing) }
                    HStack { Text("Prefill chunk"); TextField("Tokens",text:bounded($chunk,10)).harnessNumberInput().multilineTextAlignment(.trailing) }
                    HStack { Text("Numerical cap (MiB)"); TextField("MiB",text:bounded($budgetMiB,20)).harnessNumberInput().multilineTextAlignment(.trailing) }
                    Button("Load") { load() }.disabled(!idle || store.state.hasModel)
                    Text("Select MLX-format for supported Qwen2, Llama-family and full-attention Mistral artifacts, or use the existing floating/Q4 formats. Choose a complete local artifact folder. The bundled fixture is for integration checks.")
                }.disabled(!idle || store.state.hasModel)
                Section("Raw token generation") {
                    VStack(alignment:.leading) { Text("Prompt or append IDs").font(.caption); TextField("IDs",text:bounded($prompt,262144),axis:.vertical).harnessLiteralInput() }
                    VStack(alignment:.leading) { Text("Hidden stop IDs (optional)").font(.caption); TextField("IDs",text:bounded($stops,262144)).harnessLiteralInput() }
                    HStack { Text("Maximum new IDs"); TextField("0–128",text:bounded($outputCount,10)).harnessNumberInput().multilineTextAlignment(.trailing) }
                    HarnessActions {
                        Button("Prompt") { generate(0) }
                        Button("Append") { generate(1) }
                        Button("Continue") { generate(2) }
                    }.disabled(!idle || !store.state.hasSession)
                    HarnessActions {
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
                    Text(lifecycleHelp)
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
            }.scrollDismissesKeyboard(.interactively).navigationTitle("Expert mode")
                .toolbar { if !store.diagnosticsOnly { Button("Back to Chat") { store.setExpert(false) } } }
        }
    }
}

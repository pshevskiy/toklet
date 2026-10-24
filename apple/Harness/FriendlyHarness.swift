import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#else
import AppKit
#endif


private enum ChatSheet: String, Identifiable {
    case models, history, settings, benchmark
    var id: String { rawValue }
}

// Native grouped forms expose individual controls on Mac; the phone retains
// its existing list presentation.
private struct ModelList<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    @ViewBuilder var body: some View {
        #if os(macOS)
        Form { content }.formStyle(.grouped)
        #else
        List { content }
        #endif
    }
}

// Process footprint and live OS allowance are separate from engine allocations.
@MainActor
struct RuntimeMonitorView: View {
    @ObservedObject var store: HarnessStore
    private var metrics: GenerationMetrics { GenerationMetrics(store.state) }
    static func bytes(_ value: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount:Int64(clamping:value),countStyle:.memory)
    }
    static func latency(_ value: Double?) -> String {
        value.map { $0 < 1 ? String(format:"%.0f ms",$0 * 1000) : String(format:"%.2f s",$0) } ?? "—"
    }
    static func rate(_ value: Double?) -> String { value.map { String(format:"%.1f",$0) } ?? "—" }
    private var memoryText: String {
        store.memory.map { Self.bytes($0.usedBytes) + " / " + Self.bytes($0.allowedBytes) } ?? "Unavailable"
    }
    private var memoryHelp: String {
        #if os(iOS)
        "RAM is this app’s physical footprint / current OS process allowance. The allowance can change."
        #else
        "RAM is this app’s physical footprint / configured cap (half of physical RAM, up to 6 GiB). The cap is not an OS memory limit."
        #endif
    }
    var body: some View {
        VStack(alignment:.leading,spacing:6) {
            ViewThatFits(in:.horizontal) {
                HStack(alignment:.top,spacing:18) { values }
                VStack(alignment:.leading,spacing:8) { values }
            }
            DisclosureGroup("About these measurements") {
                Text(memoryHelp + " Refreshed each second. First token includes queueing and prompt preparation after Send or a generation action; tok/s measures intervals after the first output token. Model loading is excluded. Values describe the current or last generation; — means no measurement yet.")
                    .font(.caption).foregroundStyle(.secondary)
            }.font(.caption)
        }.font(.caption).monospacedDigit()
    }
    @ViewBuilder private var values: some View {
        metric("RAM used / allowed",memoryText,id:"monitor-memory")
        metric("First token",Self.latency(metrics.firstTokenSeconds),id:"monitor-ttft")
        metric("tok/s · decode",Self.rate(metrics.decodeTokensPerSecond),id:"monitor-rate")
    }
    private func metric(_ title: String,_ value: String,id: String) -> some View {
        VStack(alignment:.leading,spacing:2) {
            Text(title).foregroundStyle(.secondary)
            Text(value).fontWeight(.semibold)
        }.accessibilityElement(children:.combine).accessibilityIdentifier(id)
    }
}

@MainActor
struct FriendlyHarnessView: View {
    @ObservedObject var store: HarnessStore
    @StateObject private var hub: HubStore
    @StateObject private var chat: FriendlyChatStore
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.scenePhase) private var phase
    @State private var sheet: ChatSheet?
    @State private var importing = false
    @State private var deleting: UUID?
    @State private var deletingAll = false
    @State private var followsLatest = true
    @State private var userScrolling = false
    @State private var atBottom = true
    @FocusState private var modelSearchFocused: Bool
    init(store: HarnessStore,chat: FriendlyChatStore? = nil,hub: HubStore? = nil) {
        self.store = store
        _chat = StateObject(wrappedValue:chat ?? FriendlyChatStore(harness:store))
        _hub = StateObject(wrappedValue:hub ?? HubStore())
    }
    #if os(iOS)
    private var canvas: Color { Color(uiColor:.systemGroupedBackground) }
    private var surface: Color { Color(uiColor:.secondarySystemGroupedBackground) }
    private var leadingPlacement: ToolbarItemPlacement { .topBarLeading }
    private var trailingPlacement: ToolbarItemPlacement { .topBarTrailing }
    private var folderHelp: String { "Downloads run while the app is open. A verified download works offline. Damaged model folders can be removed in Files → Toklet Harness → TokletModels, then downloaded again." }
    #else
    private var canvas: Color { Color(nsColor:.windowBackgroundColor) }
    private var surface: Color { Color(nsColor:.controlBackgroundColor) }
    private var leadingPlacement: ToolbarItemPlacement { .navigation }
    private var trailingPlacement: ToolbarItemPlacement { .primaryAction }
    private var folderHelp: String { "Downloads run while the app is open. A verified download works offline. Use Show models in Finder to remove a damaged model folder before downloading again." }
    #endif
    private var unlocked: Bool { !store.state.busy && !chat.running && !chat.loadingModel && chat.storageAvailable }

    var body: some View {
        Group {
            if store.expert { HarnessView(store:store) }
            else { NavigationStack { conversation } }
        }
        .tint(.indigo)
        .onReceive(store.$state) { chat.observe($0) }
        .onChange(of:chat.loadingModel) { _,loading in
            if !loading && chat.ready && sheet == .models { sheet = nil }
        }
        .onChange(of:store.expert) { _,value in if value { hub.cancel() } }
        .onChange(of:phase) { _, value in
            if value != .active { chat.save() }
            #if os(iOS)
            if value != .active { chat.stopBenchmark() }
            if value == .background { hub.cancel() }
            #endif
        }
        .sheet(item:$sheet) { destination in
            NavigationStack {
                Group {
                    switch destination {
                    case .models: models
                    case .history: history
                    case .settings: settings
                    case .benchmark: benchmark
                    }
                }
                .toolbar { ToolbarItem(placement:.confirmationAction) { Button("Close") { sheet = nil } } }
            }.tint(.indigo).harnessSheetSize()
        }
    }
    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing:0) {
                    if chat.workspace.current.messages.isEmpty {
                        VStack(spacing:18) {
                            Image(systemName:"bubble.left.and.bubble.right").font(.largeTitle).foregroundStyle(.indigo)
                                .frame(width:72,height:72).background(surface,in:RoundedRectangle(cornerRadius:22)).accessibilityHidden(true)
                            Text("What’s on your mind?").font(.title2.weight(.semibold))
                            Text(chat.ready ? "Your conversation stays on this device." : "Choose a local model to start chatting.")
                                .foregroundStyle(.secondary)
                            if !chat.ready { Button("Choose a model",systemImage:"plus") { sheet = .models }.buttonStyle(.borderedProminent).controlSize(.large) }
                        }.multilineTextAlignment(.center).frame(maxWidth:360).padding(28).padding(.top,64).frame(maxWidth:.infinity)
                    } else {
                        LazyVStack(alignment:.leading,spacing:24) {
                            ForEach(chat.workspace.current.messages) { message in
                                VStack(alignment:.leading,spacing:8) {
                                    Text(message.role == "user" ? "You" : "Toklet").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                    if message.writing { HStack { ProgressView(); Text("Writing…").foregroundStyle(.secondary) } }
                                    else {
                                        if message.role == "assistant", !message.text.isEmpty { ChatContent(text:message.text).equatable() }
                                        else {
                                            Text(message.text.isEmpty ? (message.interrupted ? "Response stopped." : "No text returned.") : message.text).textSelection(.enabled)
                                        }
                                        if message.interrupted && !message.text.isEmpty { Text("Stopped").font(.caption).foregroundStyle(.secondary) }
                                        if !message.notice.isEmpty { Text(message.notice).font(.callout).foregroundStyle(.secondary) }
                                    }
                                }
                                .padding(message.role == "user" ? 16 : 0)
                                .frame(maxWidth:.infinity,alignment:.leading)
                                .background(message.role == "user" ? Color.indigo.opacity(0.07) : Color.clear,in:RoundedRectangle(cornerRadius:20))
                            }
                        }.frame(maxWidth:570).frame(maxWidth:.infinity).padding(24)
                    }
                }.id("chat-bottom")
            }
            .defaultScrollAnchor(.bottom,for:.initialOffset)
            .onScrollGeometryChange(for:Bool.self) { geometry in
                geometry.contentSize.height + geometry.contentInsets.bottom - geometry.contentOffset.y - geometry.containerSize.height <= 64
            } action: { _,value in
                atBottom = value
                if userScrolling { followsLatest = value }
            }
            .onScrollPhaseChange { _,phase in
                let interacting = phase == .tracking || phase == .interacting || phase == .decelerating
                if !interacting && userScrolling { followsLatest = atBottom }
                userScrolling = interacting
            }
            .onChange(of:chat.workspace.selected) { _,_ in
                followsLatest = true; proxy.scrollTo("chat-bottom",anchor:.bottom)
            }
            .onChange(of:chat.workspace.current.messages.count) { old,new in
                if new > old { followsLatest = true; proxy.scrollTo("chat-bottom",anchor:.bottom) }
            }
            .onChange(of:chat.workspace.current.messages.last?.text) { _,_ in
                if followsLatest && !userScrolling { proxy.scrollTo("chat-bottom",anchor:.bottom) }
            }
            .onScrollGeometryChange(for:CGSize.self) { $0.containerSize } action: { _,_ in
                if followsLatest && !userScrolling { proxy.scrollTo("chat-bottom",anchor:.bottom) }
            }
        }
        .scrollDismissesKeyboard(.interactively)
        .background(canvas)
        .safeAreaInset(edge:.top,spacing:0) { dashboard }
        .safeAreaInset(edge:.bottom,spacing:0) { composer }
        .harnessNavigationStyle()
        .toolbar {
            ToolbarItem(placement:leadingPlacement) {
                Button("Conversations",systemImage:"sidebar.left") { sheet = .history }.labelStyle(.iconOnly).frame(minWidth:44,minHeight:44)
            }
            ToolbarItem(placement:.principal) {
                Button { sheet = .models } label: {
                    VStack(spacing:2) {
                        Text(chat.hasChatModel ? chat.modelName : "Toklet").font(.headline).lineLimit(1).truncationMode(.middle)
                        Text(chat.loadingModel ? "Opening model…" : (chat.hasChatModel ? "On device" : "Choose a model")).font(.caption).foregroundStyle(.secondary)
                    }
                }.buttonStyle(.plain).frame(minHeight:44).accessibilityLabel(chat.hasChatModel ? "Choose model. " + chat.modelName + ". On device." : "Choose model").accessibilityIdentifier("chat-model-picker")
            }
            ToolbarItem(placement:trailingPlacement) {
                Button("New chat",systemImage:"square.and.pencil") { chat.newChat() }.labelStyle(.iconOnly).frame(minWidth:44,minHeight:44).disabled(!unlocked)
            }
            ToolbarItem(placement:trailingPlacement) {
                Button("Settings",systemImage:"gearshape") { sheet = .settings }.labelStyle(.iconOnly).frame(minWidth:44,minHeight:44)
            }
        }
    }
    private var activeHubProfile: HubProfile? {
        store.state.chatEnabled ? HubProfile.catalog.first { $0.name == chat.modelName } : nil
    }
    private var dashboard: some View {
        VStack(alignment:.leading,spacing:10) {
            RuntimeMonitorView(store:store)
            ViewThatFits(in:.horizontal) {
                HStack { dashboardActions }
                VStack(alignment:.leading) { dashboardActions }
            }.buttonStyle(.bordered)
        }.frame(maxWidth:570).frame(maxWidth:.infinity).padding(.horizontal,20).padding(.vertical,10).background(canvas)
    }
    @ViewBuilder private var dashboardActions: some View {
        Button("Find best model",systemImage:"sparkle.magnifyingglass") {
            hub.findBest(active:activeHubProfile); sheet = .models
        }.accessibilityIdentifier("find-best-model")
        Button("Benchmark",systemImage:"speedometer") { sheet = .benchmark }.accessibilityIdentifier("local-benchmark")
    }
    private var composer: some View {
        VStack(alignment:.leading,spacing:8) {
            if !chat.storageError.isEmpty {
                Text(chat.storageError).font(.callout)
                Button("Retry saving or opening chats") { chat.retryStorage() }
            }
            if !chat.notice.isEmpty { Text(chat.notice).font(.callout) }
            if !chat.outputLimitAvailable {
                Text("The saved reply limit is unavailable in this version. Choose a reply limit in Settings.").font(.callout)
            }
            if chat.loadingModel { ProgressView("Opening " + chat.loadingName + "…").font(.callout) }
            if store.state.hasModel && !chat.hasChatModel && !chat.loadingModel {
                Text("Choose this conversation’s model, or start a new chat with " + chat.modelName + ".").font(.callout).foregroundStyle(.secondary)
            }
            if !store.state.message.isEmpty { Text(store.state.message).font(.callout) }
            HStack(alignment:.bottom,spacing:8) {
                TextField("Message…",text:Binding(get:{ chat.workspace.current.draft },set:{ chat.edit($0) }),axis:.vertical)
                    .lineLimit(1...(typeSize.isAccessibilitySize ? 2 : 6)).padding(.vertical,15).padding(.leading,18)
                    .disabled(!chat.storageAvailable)
                    .accessibilityLabel("Message").accessibilityIdentifier("chat-draft")
                Button { chat.sendOrStop() } label: {
                    Image(systemName:chat.running ? "stop.fill" : "arrow.up").font(.system(size:18,weight:.semibold))
                        .frame(width:44,height:44).foregroundStyle(chat.canSend || chat.running ? Color.white : Color.secondary)
                        .background(chat.canSend || chat.running ? Color.indigo : Color.secondary.opacity(0.12),in:Circle())
                }.buttonStyle(.plain).keyboardShortcut(.return,modifiers:.command).padding(5).disabled(!chat.canSend && !chat.running)
                    .accessibilityLabel(chat.running ? "Stop reply" : "Send message").accessibilityIdentifier("chat-send")
            }
            .background(surface,in:RoundedRectangle(cornerRadius:28))
            .overlay(RoundedRectangle(cornerRadius:28).stroke(.quaternary,lineWidth:1))
            Text("Only on this device.").font(.caption).foregroundStyle(.secondary).frame(maxWidth:.infinity)
        }.frame(maxWidth:570).frame(maxWidth:.infinity).padding(.horizontal,16).padding(.top,10).padding(.bottom,12).background(canvas)
    }
    private var models: some View {
        ModelList {
            Section("Text chat") {
                Text("Choose a saved model to switch immediately, or download another model.").font(.callout).foregroundStyle(.secondary)
                if store.state.chatEnabled {
                    Label("Active: " + chat.modelName,systemImage:"checkmark.circle.fill").foregroundStyle(.indigo)
                    Button("Unload model") { hub.modelChoiceChanged(); chat.unload() }.disabled(!unlocked || hub.busy)
                }
            }
            Section("Chat models") {
                Button("Find best model for this device",systemImage:"sparkle.magnifyingglass") { hub.findBest(active:activeHubProfile) }
                    .accessibilityIdentifier("hub-find-best")
                if !hub.recommendationMessage.isEmpty {
                    Text(hub.recommendationMessage).font(.callout).foregroundStyle(.secondary)
                    ForEach(Array(hub.recommendations.enumerated()),id:\.element.repository) { index,profile in
                        Text("\(index + 1). " + profile.name).font(.caption.weight(.semibold))
                        modelRow(profile)
                    }
                    if !hub.recommendations.isEmpty { Text("All supported models").font(.headline).padding(.top,8) }
                }
                ForEach(HubProfile.catalog.filter { !hub.recommendations.contains($0) },id:\.repository) { profile in modelRow(profile) }
                if chat.loadingModel { ProgressView("Loading " + chat.loadingName + "…") }
                else if hub.busy {
                    if hub.downloading {
                        ProgressView(value:Double(hub.completed),total:Double(max(1,hub.total)))
                        Text(ByteCountFormatter.string(fromByteCount:hub.completed,countStyle:.file) + " of " + ByteCountFormatter.string(fromByteCount:hub.total,countStyle:.file))
                            .font(.caption).monospacedDigit()
                    } else { ProgressView(hub.phase) }
                    Button("Cancel",role:.cancel) { hub.cancel() }.accessibilityIdentifier("hub-cancel")
                }
                if !hub.message.isEmpty { Text(hub.message).font(.callout).textSelection(.enabled) }
                Button("Check saved models",systemImage:"arrow.clockwise") { hub.inventory() }.disabled(hub.busy)
                #if os(macOS)
                Button("Show models in Finder",systemImage:"folder") { NSWorkspace.shared.open(hub.modelsRoot) }
                #endif
                Text(folderHelp).font(.footnote).foregroundStyle(.secondary)
            }
            Section("Hugging Face") {
                HStack {
                    TextField("Search MLX models",text:$hub.query).harnessLiteralInput()
                        .focused($modelSearchFocused).submitLabel(.search)
                        .onSubmit { modelSearchFocused = false; hub.search() }.accessibilityIdentifier("hub-search-query")
                    Button("Search") { modelSearchFocused = false; hub.search() }.disabled(hub.blocksSelection || hub.query.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                        .accessibilityIdentifier("hub-search")
                }
                Text("Search is public. Supported results have a Use model or Download and use button. Other results need engine and tokenizer support before they can run.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            if !hub.results.isEmpty { Section("Search results") { searchResults } }
            Section("Other local folders") {
                ForEach(chat.folders,id:\.self) { folder in
                    Button { hub.modelChoiceChanged(); chat.open(folder) } label: {
                        VStack(alignment:.leading,spacing:4) {
                            Text("Open local model folder")
                            Text(folder.path.contains("M8Inputs") ? "Saved test model folder" : "Documents model folder").font(.caption).foregroundStyle(.secondary)
                        }
                    }.disabled(!unlocked || hub.busy || store.state.closed)
                }
                Button("Choose local folder",systemImage:"folder") { importing = true }.disabled(!unlocked || hub.busy || store.state.closed)
            }
            if store.state.busy { Section { ProgressView("Opening or releasing model…") } }
            if !chat.notice.isEmpty { Section { Text(chat.notice).foregroundStyle(.secondary) } }
            if !store.state.message.isEmpty { Section { Text(store.state.message) } }
        }.scrollDismissesKeyboard(.interactively).navigationTitle("Models").onAppear { chat.refreshFolders(); hub.inventory() }
            .fileImporter(isPresented:$importing,allowedContentTypes:[.folder]) { result in
                switch result {
                case .success(let url): hub.modelChoiceChanged(); chat.open(url)
                case .failure(let error):
                    let value = error as NSError
                    if value.domain != NSCocoaErrorDomain || value.code != NSUserCancelledError { chat.notice = error.localizedDescription }
                }
            }
    }
    private func useModel(_ profile: HubProfile,download: Bool) {
        modelSearchFocused = false
        let conversation = chat.workspace.selected
        let opened: @MainActor (URL) -> Void = { folder in
            guard unlocked, !store.expert, !store.state.closed else { return }
            chat.open(folder,ifSelected:conversation)
        }
        if download { hub.downloadAndUse(profile,opened:opened) }
        else { hub.use(profile,replacingLoadedModel:store.state.hasModel,opened:opened) }
    }
    private func modelRow(_ profile: HubProfile) -> some View {
        let active = chat.hasChatModel && chat.modelName == profile.name
        let saved = active || hub.savedProfiles.contains(profile.repository)
        let checking = hub.opening && hub.profile == profile
        let loading = chat.loadingModel && chat.loadingName == profile.name
        return VStack(alignment:.leading,spacing:8) {
            HStack {
                Text(profile.name).font(.headline)
                if active { Label("Active",systemImage:"checkmark.circle.fill").font(.caption).foregroundStyle(.indigo) }
            }
            Text(saved ? "Saved on this device" : ByteCountFormatter.string(fromByteCount:profile.totalBytes,countStyle:.file) + " download")
                .font(.caption).foregroundStyle(.secondary)
            Text("Memory allowance needed to open: " + RuntimeMonitorView.bytes(profile.processAllowance))
                .font(.caption).foregroundStyle(.secondary)
            Button(active ? "Return to chat" : (saved ? "Use model" : "Download and use"),systemImage:active ? "bubble.left.and.bubble.right" : "arrow.right.circle") {
                if active { hub.cancel(); sheet = nil }
                else { useModel(profile,download:!saved) }
            }.buttonStyle(.borderedProminent).frame(minHeight:44)
                .disabled(!unlocked || hub.blocksSelection || checking || store.state.closed)
                .accessibilityIdentifier("hub-use-" + profile.key)
            if checking || loading { ProgressView(loading ? "Loading model…" : hub.phase).font(.callout) }
            if hub.profile == profile, !hub.message.isEmpty { Text(hub.message).font(.callout) }
            DisclosureGroup("Model details") {
                Text(profile.repository).font(.caption).textSelection(.enabled)
                Text("Revision " + profile.revision.prefix(7) + " · Apache-2.0").font(.caption).foregroundStyle(.secondary)
            }.font(.caption)
        }.padding(.vertical,4)
    }
    private var searchResults: some View {
        ForEach(hub.results) { model in
            if let profile = model.profile { modelRow(profile) }
            else {
                VStack(alignment:.leading,spacing:5) {
                    Text(model.id).font(.callout).textSelection(.enabled)
                    Text("Unavailable for chat: this checkpoint needs engine and tokenizer support.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
    private var history: some View {
        List {
            Button("New chat",systemImage:"square.and.pencil") { chat.newChat(); sheet = nil }.buttonStyle(.borderless).frame(minHeight:44).disabled(!unlocked)
            if !chat.storageError.isEmpty {
                Section("Storage") {
                    Text(chat.storageError).foregroundStyle(.red).textSelection(.enabled)
                    if !chat.notice.isEmpty { Text(chat.notice).font(.callout) }
                    Button("Retry saving or opening chats") { chat.retryStorage() }
                }
            }
            ForEach(chat.workspace.conversations.reversed()) { item in
                HStack {
                    Button { chat.select(item.id); sheet = nil } label: {
                        VStack(alignment:.leading,spacing:4) {
                            Text(item.title).foregroundStyle(.primary).lineLimit(2)
                            if !item.draft.isEmpty { Text(item.draft).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
                        }.frame(maxWidth:.infinity,alignment:.leading).contentShape(Rectangle())
                    }.buttonStyle(.borderless).accessibilityIdentifier("history-select-" + item.id.uuidString)
                    if item.id == chat.workspace.selected { Image(systemName:"checkmark.circle.fill").accessibilityLabel("Selected") }
                    Button("Delete conversation",systemImage:"trash",role:.destructive) { deleting = item.id }
                        .labelStyle(.iconOnly).buttonStyle(.borderless).frame(minWidth:44,minHeight:44)
                        .accessibilityIdentifier("history-delete-" + item.id.uuidString)
                }.disabled(!unlocked)
                    .swipeActions { Button("Delete",role:.destructive) { deleting = item.id }.disabled(!unlocked) }
                    .contextMenu { Button("Delete conversation",role:.destructive) { deleting = item.id }.disabled(!unlocked) }
            }
            Section {
                Button("Delete all conversations",systemImage:"trash",role:.destructive) { deletingAll = true }
                    .disabled(!unlocked).accessibilityIdentifier("history-delete-all")
            }
            Section { Button("Settings",systemImage:"gearshape") { sheet = .settings }.buttonStyle(.borderless).frame(minHeight:44) }
        }.navigationTitle("Conversations")
        .confirmationDialog("Delete this conversation?",isPresented:Binding(get:{ deleting != nil },set:{ if !$0 { deleting = nil } }),titleVisibility:.visible,presenting:deleting) { id in
            // Capture the presented ID: dismissal may clear the binding before
            // this action runs.
            Button("Delete conversation",role:.destructive) { chat.remove(id); deleting = nil }
            Button("Cancel",role:.cancel) { deleting = nil }
        }
        .confirmationDialog("Delete all conversations and drafts?",isPresented:$deletingAll,titleVisibility:.visible) {
            Button("Delete all conversations",role:.destructive) { chat.removeAll() }
            Button("Cancel",role:.cancel) {}
        } message: { Text("This permanently deletes saved dialogs on this device. Downloaded models are kept.") }
    }
    private var benchmark: some View {
        List {
            Section("Local generation benchmark") {
                Text("Runs the loaded chat model offline: 1 warm-up, then 3 measured runs, up to 64 output tokens each. Early model stops are allowed. No downloads, dialog edits or quality score.")
                    .font(.callout).foregroundStyle(.secondary)
                DisclosureGroup("Workload") { Text(FriendlyChatStore.benchmarkPrompt).font(.callout).textSelection(.enabled) }
                if chat.benchmarkRunning {
                    ProgressView(chat.benchmarkStatus)
                    Button("Stop benchmark",role:.cancel) { chat.stopBenchmark() }.accessibilityIdentifier("benchmark-stop")
                } else {
                    Button("Run benchmark",systemImage:"play.fill") { chat.startBenchmark() }
                        .disabled(!chat.ready || hub.busy).accessibilityIdentifier("benchmark-run")
                    if !chat.ready { Text("Choose and load a chat model first.").font(.callout).foregroundStyle(.secondary) }
                    if !chat.benchmarkStatus.isEmpty { Text(chat.benchmarkStatus).accessibilityIdentifier("benchmark-status") }
                }
                Text("Keep the app open. Results depend on the prompt, output length, power and temperature; this short check measures this session’s speed.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            if !chat.benchmarkSamples.isEmpty {
                Section(chat.benchmarkModel) {
                    ForEach(chat.benchmarkSamples) { sample in
                        VStack(alignment:.leading,spacing:5) {
                            Text("Run \(sample.id) · \(sample.metrics.tokens) tokens · " + (sample.stopReason == 2 ? "model stop" : "token limit")).font(.headline)
                            Text("First token " + RuntimeMonitorView.latency(sample.metrics.firstTokenSeconds) + " · " + RuntimeMonitorView.rate(sample.metrics.decodeTokensPerSecond) + " tok/s")
                            Text("Total " + RuntimeMonitorView.latency(sample.metrics.totalSeconds)).foregroundStyle(.secondary)
                        }.font(.callout).monospacedDigit()
                    }
                }
            }
        }.navigationTitle("Benchmark")
    }
    private var settings: some View {
        Form {
            Section("Responses") {
                Picker("Response limit",selection:$chat.workspace.outputLimit) {
                    ForEach(chat.availableOutputLimits,id:\.self) { limit in
                        Text("\(limit.formatted()) tokens").tag(limit)
                    }
                    if !chat.outputLimitAvailable {
                        Text("\(chat.workspace.outputLimit.formatted()) tokens (unavailable)").tag(chat.workspace.outputLimit)
                    }
                }.disabled(!unlocked).onChange(of:chat.workspace.outputLimit) { _,_ in chat.save() }
                Text("Tokens are pieces of words. This sets the maximum reply length; a model can finish earlier. Prompt, chat history and reply share 512 tokens. Choose a shorter limit or start a new chat when they do not fit.").font(.footnote).foregroundStyle(.secondary)
            }
            Section("Expert") {
                Button("Expert mode",systemImage:"slider.horizontal.3") { hub.modelChoiceChanged(); sheet = nil; chat.enterExpert() }
                    .accessibilityIdentifier("expert-mode")
                Text("Raw token generation, model paths, backend settings and engine diagnostics. Switching modes unloads the active model.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }.navigationTitle("Settings")
    }
}

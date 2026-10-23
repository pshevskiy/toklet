#if os(iOS)
import SwiftUI
import UniformTypeIdentifiers
import UIKit


private enum ChatSheet: String, Identifiable {
    case models, history, settings
    var id: String { rawValue }
}

@MainActor
struct FriendlyHarnessView: View {
    @ObservedObject var store: HarnessStore
    @StateObject private var hub = HubStore()
    @StateObject private var chat: FriendlyChatStore
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.scenePhase) private var phase
    @State private var sheet: ChatSheet?
    @State private var importing = false
    @State private var deleting: UUID?
    init(store: HarnessStore) {
        self.store = store
        _chat = StateObject(wrappedValue:FriendlyChatStore(harness:store))
    }
    private var canvas: Color { Color(uiColor:.systemGroupedBackground) }
    private var surface: Color { Color(uiColor:.secondarySystemGroupedBackground) }
    private var unlocked: Bool { !store.state.busy && !chat.running && chat.storageAvailable }

    var body: some View {
        Group {
            if store.expert { HarnessView(store:store) }
            else { NavigationStack { conversation } }
        }
        .tint(.indigo)
        .onReceive(store.$state) { chat.observe($0) }
        .onChange(of:store.expert) { _,value in if value { hub.cancel() } }
        .onChange(of:phase) { _, value in
            if value != .active { chat.save() }
            if value == .background { hub.cancel() }
        }
        .sheet(item:$sheet) { destination in
            NavigationStack {
                Group {
                    switch destination {
                    case .models: models
                    case .history: history
                    case .settings: settings
                    }
                }
                .toolbar { ToolbarItem(placement:.confirmationAction) { Button("Close") { sheet = nil } } }
            }.tint(.indigo)
        }
        .confirmationDialog("Delete this conversation?",isPresented:Binding(get:{ deleting != nil },set:{ if !$0 { deleting = nil } }),titleVisibility:.visible) {
            Button("Delete conversation",role:.destructive) { if let id = deleting { chat.remove(id) }; deleting = nil }
            Button("Cancel",role:.cancel) { deleting = nil }
        }
    }
    private var conversation: some View {
        ScrollView {
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
                                Text(message.text.isEmpty ? (message.interrupted ? "Response stopped." : "No text returned.") : message.text)
                                    .textSelection(.enabled)
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
        }
        .scrollDismissesKeyboard(.interactively)
        .background(canvas)
        .safeAreaInset(edge:.bottom,spacing:0) { composer }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement:.topBarLeading) {
                Button("Conversations",systemImage:"sidebar.left") { sheet = .history }.labelStyle(.iconOnly).frame(minWidth:44,minHeight:44)
            }
            ToolbarItem(placement:.principal) {
                Button { sheet = .models } label: {
                    VStack(spacing:2) {
                        Text(chat.hasChatModel ? chat.modelName : "Toklet").font(.headline)
                        Text(chat.hasChatModel ? "On device" : "Choose a model").font(.caption).foregroundStyle(.secondary)
                    }
                }.buttonStyle(.plain).frame(minHeight:44).accessibilityLabel("Choose model").accessibilityIdentifier("chat-model-picker")
            }
            ToolbarItem(placement:.topBarTrailing) {
                Button("New chat",systemImage:"square.and.pencil") { chat.newChat() }.labelStyle(.iconOnly).frame(minWidth:44,minHeight:44).disabled(!unlocked)
            }
        }
    }
    private var composer: some View {
        VStack(alignment:.leading,spacing:8) {
            if !chat.storageError.isEmpty {
                Text(chat.storageError).font(.callout)
                Button("Retry saving or opening chats") { chat.retryStorage() }
            }
            if !chat.notice.isEmpty { Text(chat.notice).font(.callout) }
            if !store.state.message.isEmpty { Text(store.state.message).font(.callout) }
            HStack(alignment:.bottom,spacing:8) {
                TextField("Message…",text:Binding(get:{ chat.workspace.current.draft },set:chat.edit),axis:.vertical)
                    .lineLimit(1...(typeSize.isAccessibilitySize ? 2 : 6)).padding(.vertical,15).padding(.leading,18)
                    .disabled(!chat.storageAvailable)
                    .accessibilityLabel("Message").accessibilityIdentifier("chat-draft")
                Button { chat.sendOrStop() } label: {
                    Image(systemName:chat.running ? "stop.fill" : "arrow.up").font(.system(size:18,weight:.semibold))
                        .frame(width:44,height:44).foregroundStyle(chat.canSend || chat.running ? Color.white : Color.secondary)
                        .background(chat.canSend || chat.running ? Color.indigo : Color.secondary.opacity(0.12),in:Circle())
                }.buttonStyle(.plain).padding(5).disabled(!chat.canSend && !chat.running)
                    .accessibilityLabel(chat.running ? "Stop reply" : "Send message").accessibilityIdentifier("chat-send")
            }
            .background(surface,in:RoundedRectangle(cornerRadius:28))
            .overlay(RoundedRectangle(cornerRadius:28).stroke(.quaternary,lineWidth:1))
            Text("Only on this device.").font(.caption).foregroundStyle(.secondary).frame(maxWidth:.infinity)
        }.frame(maxWidth:570).frame(maxWidth:.infinity).padding(.horizontal,16).padding(.top,10).padding(.bottom,12).background(canvas)
    }
    private var models: some View {
        List {
            Section("Text chat") {
                Text("Local text chat").font(.headline)
                Text("Download a tested model below, or open a folder already on this device. Other model formats remain available in Expert mode.").font(.callout).foregroundStyle(.secondary)
                if store.state.chatEnabled {
                    Label("Model loaded",systemImage:"checkmark.circle.fill").foregroundStyle(.indigo)
                    Button("Unload model") { hub.modelChoiceChanged(); chat.unload() }.disabled(!unlocked || hub.busy)
                }
            }
            Section("Hugging Face") {
                HStack {
                    TextField("Search MLX models",text:$hub.query).textInputAutocapitalization(.never).autocorrectionDisabled()
                        .submitLabel(.search).onSubmit { hub.search() }.accessibilityIdentifier("hub-search-query")
                    Button("Search") { hub.search() }.disabled(hub.busy || hub.query.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                        .accessibilityIdentifier("hub-search")
                }
                Text("Search is public. Only qualified text profiles can be downloaded here.")
                    .font(.footnote).foregroundStyle(.secondary)
                ForEach(hub.results) { model in
                    VStack(alignment:.leading,spacing:5) {
                        Text(model.id).font(.callout).textSelection(.enabled)
                        Text(model.profile == nil ? "Not supported for chat: this checkpoint’s engine and text profile are not qualified." : "Supported pinned profile · shown below")
                            .font(.caption).foregroundStyle(.secondary)
                        if let profile = model.profile {
                            Button("Select for chat") { hub.select(profile) }.disabled(hub.busy)
                        }
                    }
                }
            }
            Section("Available for chat") {
                Picker("Model",selection:Binding(get:{ hub.profile.repository },set:{ id in
                    if let profile = HubProfile.catalog.first(where:{ $0.repository == id }) { hub.select(profile) }
                })) {
                    ForEach(HubProfile.catalog,id:\.repository) { profile in Text(profile.name).tag(profile.repository) }
                }.disabled(hub.busy).accessibilityIdentifier("hub-profile")
                VStack(alignment:.leading,spacing:6) {
                    Text(hub.profile.name).font(.headline)
                    Text(hub.profile.repository).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    Text(ByteCountFormatter.string(fromByteCount:hub.profile.totalBytes,countStyle:.file) + " · tested revision " + hub.profile.revision.prefix(7) + " · Apache-2.0").font(.caption).foregroundStyle(.secondary)
                    Text(hub.deviceMessage).font(.footnote).foregroundStyle(.secondary)
                }
                if hub.localURL != nil {
                    Button("Open downloaded model",systemImage:"bubble.left.and.bubble.right") {
                        hub.open { folder in
                            guard unlocked, !store.expert, !store.state.closed else { return }
                            chat.open(folder)
                        }
                    }.disabled(!unlocked || hub.busy || !hub.deviceAllowed || store.state.closed)
                        .accessibilityIdentifier("hub-open")
                } else {
                    Button("Download model",systemImage:"arrow.down.circle") { hub.download() }
                        .disabled(!unlocked || hub.busy || !hub.deviceAllowed)
                        .accessibilityIdentifier("hub-download")
                }
                if hub.busy {
                    if hub.downloading {
                        ProgressView(value:Double(hub.completed),total:Double(max(1,hub.total)))
                        Text(ByteCountFormatter.string(fromByteCount:hub.completed,countStyle:.file) + " of " + ByteCountFormatter.string(fromByteCount:hub.total,countStyle:.file))
                            .font(.caption).monospacedDigit()
                    } else { ProgressView() }
                    Text(hub.phase).font(.callout)
                    Button("Cancel",role:.cancel) { hub.cancel() }.accessibilityIdentifier("hub-cancel")
                }
                if !hub.message.isEmpty { Text(hub.message).font(.callout).textSelection(.enabled) }
                Button("Check saved model and device again",systemImage:"arrow.clockwise") { hub.refresh() }.disabled(hub.busy)
                Text("Downloads run while the app is open. A verified download works offline. Damaged model folders can be removed in Files → Toklet Harness → TokletModels, then downloaded again.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Section("Other local folders") {
                ForEach(chat.folders,id:\.self) { folder in
                    Button { hub.modelChoiceChanged(); chat.open(folder) } label: {
                        VStack(alignment:.leading,spacing:4) {
                            Text("Open local model folder")
                            Text(folder.path.contains("M8Inputs") ? "Saved test model folder" : "Documents model folder").font(.caption).foregroundStyle(.secondary)
                        }
                    }.disabled(!unlocked || hub.busy || store.state.closed)
                }
                Button("Choose folder in Files",systemImage:"folder") { importing = true }.disabled(!unlocked || hub.busy || store.state.closed)
            }
            if store.state.busy { Section { ProgressView("Opening or releasing model…") } }
            if !store.state.message.isEmpty { Section { Text(store.state.message) } }
        }.navigationTitle("Models").onAppear { chat.refreshFolders(); hub.refresh() }
            .fileImporter(isPresented:$importing,allowedContentTypes:[.folder]) { result in
                switch result {
                case .success(let url): hub.modelChoiceChanged(); chat.open(url)
                case .failure(let error): chat.notice = error.localizedDescription
                }
            }
    }
    private var history: some View {
        List {
            Button("New chat",systemImage:"square.and.pencil") { chat.newChat(); sheet = nil }.disabled(!unlocked)
            ForEach(chat.workspace.conversations.reversed()) { item in
                Button { chat.select(item.id); sheet = nil } label: {
                    HStack {
                        VStack(alignment:.leading,spacing:4) {
                            Text(item.title).foregroundStyle(.primary).lineLimit(2)
                            if !item.draft.isEmpty { Text(item.draft).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
                        }
                        Spacer()
                        if item.id == chat.workspace.selected { Image(systemName:"checkmark.circle.fill") }
                    }
                }.disabled(!unlocked)
                    .swipeActions { Button("Delete",role:.destructive) { deleting = item.id }.disabled(!unlocked) }
            }
            Section { Button("Settings",systemImage:"gearshape") { sheet = .settings } }
        }.navigationTitle("Conversations")
    }
    private var settings: some View {
        Form {
            Section("Responses") {
                Picker("Response limit",selection:$chat.workspace.outputLimit) {
                    Text("32 tokens").tag(UInt32(32)); Text("64 tokens").tag(UInt32(64)); Text("128 tokens").tag(UInt32(128))
                }.disabled(!unlocked).onChange(of:chat.workspace.outputLimit) { _,_ in chat.save() }
                Text("Long conversations may need a new chat before another response can fit.").font(.footnote).foregroundStyle(.secondary)
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
#endif

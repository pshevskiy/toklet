import SwiftUI

@MainActor
struct FriendlyPlanningSection: View {
    @ObservedObject var chat: FriendlyChatStore
    @ObservedObject var store: HarnessStore
    private var planning: FriendlyPlanningState { chat.planning }
    private func bytes(_ value: UInt64) -> String { RuntimeMonitorView.bytes(value) }
    private func positions(_ value: UInt64?) -> String { value.map { $0.formatted() } ?? "Unavailable" }
    private func input(_ name: String, _ field: FriendlyPlanningDraft.Field, _ value: String) -> some View {
        LabeledContent(name) {
            TextField("",text:Binding(get:{ value },set:{ chat.editPlanning(field,$0) }))
                .autocorrectionDisabled().accessibilityLabel(name)
        }
    }
    var body: some View {
        Section("Chat window and memory") {
            Text("Preview an estimate first. Applying uses the opened model and resets its cached tokens; your conversation and draft are kept. Accepted settings are saved for supported managed models with verified identity; other models keep them only while open.")
                .font(.footnote).foregroundStyle(.secondary)
            if let target = planning.target {
                input("Window to estimate (tokens)",.upper,planning.draft.upper)
                input("Chunk size (tokens)",.chunk,planning.draft.chunk)
                input("Engine memory budget (MiB)",.budgetMiB,planning.draft.budgetMiB)
                Button("Preview only",systemImage:"magnifyingglass") { chat.previewPlanning() }
                    .disabled(store.state.busy || chat.running || chat.loadingModel || store.state.closed)
                    .accessibilityIdentifier("planning-preview")
                if planning.pending { ProgressView("Preparing estimate…") }
                DisclosureGroup("Budget details") {
                    Text("The engine bridge allowance is subtracted once from this budget. The separate text allowance is \(bytes(target.textBudget)).")
                    Text("Replacement policy reserves 272 MiB for app work, including 16 MiB for staging. This is an allowance, not a measurement of allocated memory.")
                    Text("Preview working allowance: \(bytes(UInt64(NativePreviewResult.requiredApplicationAllowance))). This is outside the engine estimate.")
                }.font(.footnote).foregroundStyle(.secondary)
            } else {
                Text("Open this conversation’s model before previewing. No model is selected or downloaded here.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            if !planning.message.isEmpty { Text(planning.message).foregroundStyle(.red) }
            if let value = planning.result {
                result(value)
                Button("Apply to current chat",systemImage:"checkmark.circle") { chat.applyPlanning() }
                    .disabled(chat.planningApplyUnavailableReason != nil)
                    .accessibilityIdentifier("planning-apply")
                if let reason = chat.planningApplyUnavailableReason {
                    Text(reason).font(.footnote).foregroundStyle(.secondary)
                }
                Text("Applying resets cached tokens. The next Send re-encodes this conversation and checks the prompt plus reply against the new window. Your reply preference is kept.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            if chat.applying.pending { ProgressView("Applying settings…") }
            if !chat.applying.message.isEmpty { Text(chat.applying.message).font(.footnote) }
            current
            DisclosureGroup("Qwen2 limit for the next load") {
                Picker("Model limit (tokens)",selection:Binding(get:{ chat.qwen2LoadUpper },set:{ chat.setQwen2LoadUpper($0) })) {
                    Text("512 (default)").tag(UInt32(512))
                    Text("1024").tag(UInt32(1024))
                }.disabled(chat.loadingModel || chat.running || store.state.busy)
                Text("For the managed Qwen2 model only. Choose Reopen with new limit in Models to unload and reopen it. Editing this choice does not change the open model or saved preferences. The initial session is still 512 tokens; Preview and Apply can change its KV capacity within the admitted model limit.")
                Text("Input and reply share the active chat window. Replies remain limited to 256 tokens. Text and message limits still apply; allocating a larger window does not establish full-window quality.")
            }.font(.footnote)
            Text("Estimates cover inference memory. App/tokenizer memory and the currently loaded model are separate. Opening a model checks its files and memory again.")
                .font(.footnote).foregroundStyle(.secondary)
        }
    }
    @ViewBuilder private var current: some View {
        if let context = store.state.modelContext, chat.hasChatModel {
            if let configuration = chat.currentSessionConfiguration {
                LabeledContent("Active chat window",value:configuration.capacity.formatted())
                LabeledContent("Active chunk size",value:configuration.chunk.formatted())
                LabeledContent("Active engine budget",value:bytes(configuration.wholeSessionBudget))
            } else {
                Text("No active session. Sending resumes the retained chat settings.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            DisclosureGroup("Active chat details") {
                LabeledContent("Loaded model window",value:positions(context.declaredTotal))
                LabeledContent("Loaded engine limit",value:context.loadedLimit.formatted())
                if let plan = store.state.runtimePlan {
                    LabeledContent("KV estimate when model opened",value:bytes(plan.sessionMemory.kvBytes))
                    LabeledContent("Engine fit when opened",value:plan.fitsRequestedBudgets ? "Fit at load" : "Did not fit")
                } else { Text("The memory report from opening this model is unavailable.") }
                if let live = store.state.liveContext {
                    LabeledContent("Engine and bridge accounted memory",value:bytes(live.numericalAdmission))
                }
                if let target = planning.target {
                    LabeledContent("Replacement coexistence limit",value:bytes(target.coexistenceBudget))
                    LabeledContent("Separate text allowance",value:bytes(target.textBudget))
                }
                Text("The load report is historical and does not describe a replacement session. Current app memory and process headroom are separate measurements.")
            }.font(.footnote)
        }
    }
    @ViewBuilder private func result(_ value: NativePreviewResult) -> some View {
        if let limits = value.limits {
            LabeledContent("Model window",value:positions(limits.declaredTotalPositions))
            LabeledContent("Engine planning limit",value:positions(limits.factoryPositionUpper))
        }
        if value.isResidentAdvisoryEstimate {
            Text("Advisory estimate from model files. Applying checks the opened model and current memory; this estimate does not identify its loaded weights or allocate a replacement.")
                .font(.footnote).foregroundStyle(.secondary)
            if let resident = value.request.resident { LabeledContent("Opened model limit",value:resident.loadedUpper.formatted()) }
        }
        LabeledContent("Requested window",value:value.request.upper.formatted())
        if let request = value.capacityRequest {
            LabeledContent("Effective search upper",value:request.upper.formatted())
        }
        switch value.outcome {
        case .failed:
            Text("Preview failed (\(value.status)): \(value.message)").foregroundStyle(.red)
        case .metadataOnly:
            Text(metadataReason(value)).font(.footnote).foregroundStyle(.secondary)
        case .planned:
            if let plan = value.capacity {
                if let selected = plan.capacity.selected {
                    LabeledContent("Estimated capacity",value:selected.capacity.formatted())
                    LabeledContent("Estimated engine memory",value:bytes(selected.footprint.admittedBytes))
                } else { Text("No capacity fits this engine budget and buffer limit.") }
                Text(plan.capacity.selected.map { $0.fitsBudget && $0.fitsBuffer } == true
                    ? "The selected estimate fits its session budget and buffer limit. Applying checks the resident model and current memory again."
                    : "No estimate fits the session budget and buffer limit.")
                    .font(.footnote).foregroundStyle(.secondary)
                DisclosureGroup("Estimate details") {
                    if let request = value.capacityRequest {
                        LabeledContent("Budget after bridge allowance",value:bytes(request.sessionBudget))
                    }
                    // Resources describe the effective requested upper, not the
                    // selected lower point, which has only its total footprint.
                    LabeledContent("KV at search upper",value:bytes(plan.resources.sessionMemory.kvBytes))
                    LabeledContent("Search-upper engine fit",value:plan.resources.fitsRequestedBudgets ? "Fits" : "Does not fit")
                    if value.isResidentAdvisoryEstimate { Text("Loading-fit details describe the reopened files. They are not a requirement for replacing a session on already loaded weights.") }
                    LabeledContent("Source loading fit",value:plan.resources.fitsSourceLoadBudget ? "Fits" : "Does not fit")
                    LabeledContent("Resident loading fit",value:plan.resources.fitsResidentLoadBudget ? "Fits" : "Does not fit")
                }.font(.footnote)
            }
        }
    }
    private func metadataReason(_ value: NativePreviewResult) -> String {
        switch value.limitation {
        case .auditedProfileHasNoCapacityAPI: "Model information only: this profile has no capacity estimator."
        case .backendUnavailable: "Model information only: this build has no native capacity estimator."
        case .metadataUnsupported: "This model does not provide a supported engine planning range."
        case nil: "No capacity estimate is available."
        }
    }
}

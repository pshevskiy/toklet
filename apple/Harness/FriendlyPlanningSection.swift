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
        Section("Preview only") {
            Text("Estimate a window and memory budget. This does not change Chat or load a model. There is nothing to apply or save.")
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
                    Text("Preview working allowance: \(bytes(UInt64(NativePreviewResult.requiredApplicationAllowance))). This is outside the engine estimate.")
                }.font(.footnote).foregroundStyle(.secondary)
            } else {
                Text("Open this conversation’s model before previewing. No model is selected or downloaded here.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            if !planning.message.isEmpty { Text(planning.message).foregroundStyle(.red) }
            if let value = planning.result { result(value) }
            current
            Text("Estimates cover inference memory. App/tokenizer memory and the currently loaded model are separate. Opening a model checks its files and memory again.")
                .font(.footnote).foregroundStyle(.secondary)
        }
    }
    @ViewBuilder private var current: some View {
        if let context = store.state.modelContext, chat.hasChatModel {
            LabeledContent("Active chat window",value:context.configuredCapacity.formatted())
            Text("Chat still uses 512 positions and chunk 32. This preview does not change those settings.")
                .font(.footnote).foregroundStyle(.secondary)
            DisclosureGroup("Active chat details") {
                LabeledContent("Loaded model window",value:positions(context.declaredTotal))
                LabeledContent("Loaded engine limit",value:context.loadedLimit.formatted())
                if let plan = store.state.runtimePlan {
                    LabeledContent("Fixed KV allocation",value:bytes(plan.sessionMemory.kvBytes))
                    LabeledContent("Engine fit when opened",value:plan.fitsRequestedBudgets ? "Fit at load" : "Did not fit")
                } else { Text("The memory report from opening this model is unavailable.") }
                if let live = store.state.liveContext {
                    LabeledContent("Engine and bridge accounted memory",value:bytes(live.numericalAdmission))
                }
                if let target = planning.target {
                    LabeledContent("Current engine memory budget",value:bytes(target.wholeBudget))
                    LabeledContent("Separate text allowance",value:bytes(target.textBudget))
                }
                Text("More history uses the same reserved KV space. Current app memory and OS headroom are separate measurements.")
            }.font(.footnote)
        }
    }
    @ViewBuilder private func result(_ value: NativePreviewResult) -> some View {
        if let limits = value.limits {
            LabeledContent("Model window",value:positions(limits.declaredTotalPositions))
            LabeledContent("Engine planning limit",value:positions(limits.factoryPositionUpper))
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
                Text(plan.selectedFitsNumericalBudgets ? "This estimate fits the engine memory checks only." : "No estimate satisfies all engine loading and session checks.")
                    .font(.footnote).foregroundStyle(.secondary)
                DisclosureGroup("Estimate details") {
                    if let request = value.capacityRequest {
                        LabeledContent("Budget after bridge allowance",value:bytes(request.sessionBudget))
                    }
                    // Resources describe the effective requested upper, not the
                    // selected lower point, which has only its total footprint.
                    LabeledContent("KV at search upper",value:bytes(plan.resources.sessionMemory.kvBytes))
                    LabeledContent("Search-upper engine fit",value:plan.resources.fitsRequestedBudgets ? "Fits" : "Does not fit")
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

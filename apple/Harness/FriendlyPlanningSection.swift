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
                DisclosureGroup("Advanced KV ceiling") {
                    input("KV ceiling (MiB; blank is unlimited)",.kvBudgetMiB,planning.draft.kvBudgetMiB)
                    Text("Limits cache buffers including overhead, alongside the engine budget. Zero permits no KV. This ceiling applies only to this Preview and Apply; it is not saved.")
                }.font(.footnote)
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
            DisclosureGroup("Loading-memory observation") {
                Toggle("Measure model loads while enabled",isOn:Binding(get:{ chat.measureLoadingMemory },set:{ chat.setMeasureLoadingMemory($0) }))
                    .disabled(chat.loadingModel || chat.running || store.state.busy)
                Text("Opt-in process-footprint checkpoints around loading and initial setup. To measure an open model, unload it in Models and open it again. No model-only attribution or device-fit guarantee.")
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
                LabeledContent("Declared model window",value:positions(context.declaredTotal))
                LabeledContent("Declared input limit",value:positions(context.declaredInput))
                LabeledContent("Declared output limit",value:positions(context.declaredOutput))
                LabeledContent("Global engine ceiling",value:context.engineGlobalCeiling.formatted())
                LabeledContent("Loaded engine limit",value:context.loadedLimit.formatted())
                let limits = NativeGenerationPolicy.limits(.chat,model:context)
                LabeledContent("Active input ceiling",value:limits.maximumPrompt.formatted())
                LabeledContent("Reply ceiling",value:limits.maximumOutput.formatted())
                Text("Prompt and reply must fit together. Each request allows up to 63 messages and 64 KiB of serialized conversation; drafts allow 16 KiB.")
                if let plan = store.state.runtimePlan {
                    LabeledContent("KV estimate when model opened",value:bytes(plan.sessionMemory.kvBytes))
                    LabeledContent("Engine fit when opened",value:plan.fitsRequestedBudgets ? "Fit at load" : "Did not fit")
                } else if store.state.acceptedIdentity?.profile != .gemmaBF16 {
                    Text("The memory report from opening this model is unavailable.")
                }
                if let live = store.state.liveContext, store.state.acceptedIdentity?.profile != .gemmaBF16 {
                    LabeledContent("Current admission with allowances",value:bytes(live.numericalAdmission))
                }
                if let memory = store.state.sessionMemory,
                   memory.matches(store.state.liveContext,configuration:store.state.sessionConfiguration) { ownerMemory(memory) }
                if let load = store.state.loadMemory,
                   load.matches(model:context,loadRequestID:store.state.acceptedLoadRequestID) {
                    Text("Process footprint around load and initial setup").font(.footnote.weight(.semibold))
                    LabeledContent("Initial session window",value:load.requestedCapacity.formatted())
                    if case .available(let before) = load.before {
                        LabeledContent("Physical footprint before load",value:bytes(before.physicalBytes))
                        LabeledContent("Lifetime peak before load",value:bytes(before.lifetimePeakPhysicalBytes))
                    }
                    if case .available(let after) = load.after {
                        LabeledContent("Physical footprint after load",value:bytes(after.physicalBytes))
                        LabeledContent("Lifetime peak after load",value:bytes(after.lifetimePeakPhysicalBytes))
                    }
                    if let peak = load.establishedProcessPeakBytes {
                        LabeledContent("Process peak established during load",value:bytes(peak))
                    }
                    Text(load.peakExplanation)
                    Text("Includes validation, compatibility inspection, initial session/text setup and synchronous cleanup. It excludes later resizing and generation. Footprints and owner counts must not be added together.")
                }
                if let target = planning.target {
                    LabeledContent("Replacement coexistence limit",value:bytes(target.coexistenceBudget))
                    LabeledContent("Separate text allowance",value:bytes(target.textBudget))
                }
                Text("The load report is historical and does not describe a replacement session. Current app memory and process headroom are separate measurements.")
                Text("Full-window quality and whole-device fit are not evaluated by these reports.")
            }.font(.footnote)
        }
    }
    @ViewBuilder private func ownerMemory(_ memory: NativeSessionMemorySnapshot) -> some View {
        switch memory.reading {
        case .available(let shared,let owned,let driver,let bridge,let storage):
            Text("Owner bytes at session creation or replacement").font(.footnote.weight(.semibold))
            LabeledContent("Shared learned weights",value:bytes(shared.learnedBytes))
            LabeledContent("Shared position buffers",value:bytes(shared.positionalBytes))
            LabeledContent("Shared model tables",value:bytes(shared.modelTableBytes))
            LabeledContent("KV storage",value:storage)
            LabeledContent("KV buffers",value:bytes(owned.kvBytes))
            LabeledContent("Scratch buffers",value:bytes(owned.scratchBytes))
            LabeledContent("History storage",value:bytes(owned.historyBytes))
            LabeledContent("Logits storage",value:bytes(owned.logitsBytes))
            LabeledContent("Session tables",value:bytes(owned.sessionTableBytes))
            LabeledContent("Separate driver allowance",value:bytes(driver))
            LabeledContent("Separate bridge allowance",value:bytes(bridge))
            Text("Shared weights are counted once. This snapshot is not refreshed per token; history storage may grow. It excludes loading peaks and app/process memory.")
        case .officialGemma(let load,let session,let kv,let codec):
            LabeledContent("Combined model load admission",value:bytes(load))
            LabeledContent("Combined session admission",value:bytes(session))
            LabeledContent("KV buffers",value:bytes(kv))
            LabeledContent("Codec allowance included above",value:bytes(codec))
        case .unavailable(let reason):
            Text("Owner memory report unavailable: " + reason)
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
        if let ceiling = value.request.kvBudgetBytes { LabeledContent("KV ceiling",value:bytes(ceiling)) }
        switch value.outcome {
        case .failed:
            Text("Preview failed (\(value.status)): \(value.message)").foregroundStyle(.red)
        case .metadataOnly:
            Text(metadataReason(value)).font(.footnote).foregroundStyle(.secondary)
        case .planned:
            if let plan = value.capacity {
                if let selected = plan.capacity.selected {
                    LabeledContent("Estimated capacity",value:selected.capacity.formatted())
                    if let kv = selected.footprint.kvBytes { LabeledContent("Estimated KV buffers",value:bytes(kv)) }
                    LabeledContent("Estimated engine memory",value:bytes(selected.footprint.admittedBytes))
                } else { Text("No capacity fits the configured limits.") }
                Text(plan.capacity.selected.map { $0.fits } == true
                    ? "The selected estimate fits the configured limits. Applying checks the resident model and current memory again."
                    : "No estimate fits the configured limits.")
                    .font(.footnote).foregroundStyle(.secondary)
                if let excluded = plan.capacity.firstExcluded, excluded.fitsKvBudget == false {
                    LabeledContent("First window excluded by KV ceiling",value:excluded.capacity.formatted() + " tokens")
                    if let kv = excluded.footprint.kvBytes { LabeledContent("KV at excluded window",value:bytes(kv)) }
                }
                DisclosureGroup("Estimate details") {
                    if let request = value.capacityRequest {
                        LabeledContent("Budget after bridge allowance",value:bytes(request.sessionBudget))
                    }
                    // Resources describe the effective requested upper, not the
                    // selected lower point, whose KV footprint is optional.
                    LabeledContent("KV at search upper",value:bytes(plan.resources.sessionMemory.kvBytes))
                    LabeledContent("Search-upper numerical budget fit",value:plan.resources.fitsRequestedBudgets ? "Fits" : "Does not fit")
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

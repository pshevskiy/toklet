import Foundation
import CToklet

// Correlation with an already loaded chat, never file authentication.
struct FriendlyPlanningTarget: Equatable {
    let worker: ObjectIdentifier
    let modelInstance: UInt64
    let conversation: UUID
    let modelKey, directory: String
    let profile: NativeContextLimits.LoadProfile
    let wholeBudget, textBudget: UInt64
    var configuration: NativeSessionConfiguration? = nil
    var coexistenceBudget: UInt64 = 0
}

// Created only after accepted load enqueue, promoted only at its completed
// successful Chat load. Keeping the directory does not extend a security lease.
struct FriendlyPlanningLoad {
    let worker: ObjectIdentifier
    let settings: NativeSettings
    var coexistenceBudget: UInt64 = 0
    private(set) var modelInstance: UInt64? = nil
    mutating func complete(worker: ObjectIdentifier, instance: UInt64?, chatEnabled: Bool) {
        guard worker == self.worker, chatEnabled, let instance, instance > 0 else {
            modelInstance = nil; return
        }
        modelInstance = instance
    }
    func target(worker: ObjectIdentifier, instance: UInt64, conversation: UUID,
                loadedKey: String, selectedKey: String, configuration: NativeSessionConfiguration? = nil) -> FriendlyPlanningTarget? {
        guard worker == self.worker, modelInstance == instance, instance > 0,
              !loadedKey.isEmpty, loadedKey == selectedKey else { return nil }
        return FriendlyPlanningTarget(worker:worker,modelInstance:instance,conversation:conversation,
            modelKey:loadedKey,directory:settings.directory,profile:settings.loadIdentity == .managedSmol ? .smolAudited : .ordinary,
            wholeBudget:configuration?.wholeSessionBudget ?? settings.numericalBudget,textBudget:settings.textBudget,
            configuration:configuration,coexistenceBudget:coexistenceBudget)
    }
}

struct FriendlyPlanningDraft: Equatable {
    enum Field: Equatable { case upper, chunk, budgetMiB }
    var upper = "512"
    var chunk = "32"
    var budgetMiB = "512"

    private func decimal(_ text: String, name: String) throws -> UInt64 {
        guard !text.isEmpty, text.utf8.count <= 20,
              text.utf8.allSatisfy({ (48...57).contains($0) }),
              let value = UInt64(text) else { throw NativeContextFailure("Enter a whole number for \(name).") }
        return value
    }
    func request(selection: NativePreviewSelection) throws -> NativePreviewRequest {
        let upperValue = try decimal(upper,name:"context upper")
        let chunkValue = try decimal(chunk,name:"chunk size")
        let mib = try decimal(budgetMiB,name:"native budget")
        let (whole,overflow) = mib.multipliedReportingOverflow(by:1024 * 1024)
        guard !overflow, let upper = UInt32(exactly:upperValue), let chunk = UInt32(exactly:chunkValue) else {
            throw NativeContextFailure("Planning values exceed their supported range.")
        }
        let numerical = try NativeResourcePlan.numericalBudget(wholeBudget:whole)
        let request = NativePreviewRequest(id:UUID(),selection:selection,upper:upper,chunk:chunk,offset:0,
            contextCeiling:nil,loadBudget:numerical,sessionBudget:numerical)
        try request.validate()
        return request
    }
}

// One ephemeral draft/result. There are deliberately no load/session/chat APIs.
struct FriendlyPlanningState {
    private(set) var draft = FriendlyPlanningDraft()
    private(set) var target: FriendlyPlanningTarget?
    private(set) var selection: NativePreviewSelection?
    private(set) var request: NativePreviewRequest?
    private(set) var result: NativePreviewResult?
    private(set) var message = ""
    var pending: Bool { request != nil && result == nil }

    @discardableResult
    mutating func select(_ next: FriendlyPlanningTarget?) -> Bool {
        guard next != target else { return false }
        invalidate(); target = next
        selection = next.map { NativePreviewSelection(id:UUID(),directory:$0.directory,profile:$0.profile) }
        draft = FriendlyPlanningDraft()
        if let next {
            draft.budgetMiB = String(next.wholeBudget / (1024 * 1024))
            if let current = next.configuration { draft.upper = String(current.capacity); draft.chunk = String(current.chunk) }
        }
        return true
    }
    mutating func invalidate() { request = nil; result = nil; message = "" }
    mutating func edit(_ field: FriendlyPlanningDraft.Field, _ text: String) {
        invalidate()
        // Retain invalid input for an honest error, but bound pasted UI text.
        let bounded = String(bytes:text.utf8.prefix(64),encoding:.utf8) ?? "<invalid text>"
        switch field {
        case .upper: draft.upper = bounded
        case .chunk: draft.chunk = bounded
        case .budgetMiB: draft.budgetMiB = bounded
        }
    }
    mutating func begin() throws -> NativePreviewRequest {
        invalidate()
        do {
            guard let selection, target != nil else { throw NativeContextFailure("Open this conversation’s model before previewing.") }
            let next = try draft.request(selection:selection); request = next
            return next
        } catch { message = error.localizedDescription; throw error }
    }
    mutating func reject(_ text: String) { invalidate(); message = text }
    mutating func observe(_ value: NativePreviewResult?, busy: Bool) {
        guard let request else { result = nil; return }
        if let value, value.request == request, value.request.selection == selection {
            result = value
        } else if result != nil || !busy { invalidate() }
    }
}

// One ephemeral application. Native IDs correlate a resident owner; preview
// files and saved paths never authorize replacement. No workspace is retained.
struct FriendlyApplyRequest {
    let target: FriendlyPlanningTarget
    let previewID: UUID
    let transition: NativeTransitionRequest
}

struct FriendlyApplyState {
    private(set) var request: FriendlyApplyRequest?
    private var ownerTarget: FriendlyPlanningTarget?
    private(set) var message = ""
    private(set) var committed = false
    var pending: Bool { request != nil }

    static func prepare(planning: FriendlyPlanningState, current: NativeSnapshot,
                        target: FriendlyPlanningTarget, reply: UInt32) throws -> FriendlyApplyRequest {
        guard !current.busy, !current.closed, !current.restartRequired, current.hasSession,
              let configuration = current.sessionConfiguration, let live = current.liveContext,
              configuration == target.configuration,
              configuration.modelInstance == target.modelInstance, target.profile == .ordinary,
              live.modelInstance == configuration.modelInstance, live.sessionInstance == configuration.sessionInstance,
              planning.target == target, let selection = planning.selection, let requested = planning.request,
              let result = planning.result, result.request == requested, result.request.selection == selection,
              result.outcome == .planned, result.status == 0,
              let capacityRequest = result.capacityRequest, let plan = result.capacity,
              let selected = plan.capacity.selected, selected.fitsBudget, selected.fitsBuffer,
              capacityRequest.chunk == requested.chunk, capacityRequest.sessionBudget == requested.sessionBudget,
              capacityRequest.offset == 0, requested.offset == 0,
              plan.capacity.limits.budgetBytes == capacityRequest.sessionBudget,
              target.coexistenceBudget > 0 else {
            throw NativeContextFailure("Preview again for this idle, active chat session before applying.")
        }
        guard UInt64(selected.capacity) <= live.loadedLimit else {
            throw NativeContextFailure("This estimate exceeds the opened model’s limit. Reloading with a larger limit is a separate action.")
        }
        guard let model = current.modelContext else { throw NativeContextFailure("The model context is unavailable.") }
        try FriendlyModelPreference.validateReply(reply,capacity:UInt64(selected.capacity),model:model)
        let whole = try NativeTransitionPlan.sum([capacityRequest.sessionBudget,UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES)])
        let draftRequest = try planning.draft.request(selection:selection)
        guard draftRequest.upper == requested.upper, draftRequest.chunk == requested.chunk,
              draftRequest.sessionBudget == requested.sessionBudget, draftRequest.loadBudget == requested.loadBudget else {
            throw NativeContextFailure("The displayed estimate no longer matches your edited values. Preview again.")
        }
        let owner = NativeTransitionSelection(id:UUID(),modelInstance:configuration.modelInstance,
            sessionInstance:configuration.sessionInstance)
        var transition = NativeTransitionRequest(id:UUID(),selection:owner,capacity:selected.capacity,
            chunk:capacityRequest.chunk,wholeSessionBudget:whole,coexistenceBudget:target.coexistenceBudget,
            replayReserve:0,textReserve:target.textBudget,applicationReserve:NativeTransitionHeadroomCheck.applicationReserve)
        transition.memoryPolicy = .friendly
        try transition.validate()
        return FriendlyApplyRequest(target:target,previewID:result.request.id,transition:transition)
    }
    mutating func begin(_ value: FriendlyApplyRequest) { request = value; ownerTarget = value.target; message = ""; committed = false }
    mutating func reject(_ text: String) { request = nil; message = text; committed = false }
    mutating func invalidate() { request = nil; ownerTarget = nil; message = ""; committed = false }
    mutating func observe(target: FriendlyPlanningTarget?, snapshot: NativeSnapshot) {
        if let owner = ownerTarget {
            guard let target, target.worker == owner.worker, target.modelInstance == owner.modelInstance,
                  target.conversation == owner.conversation, target.modelKey == owner.modelKey,
                  target.directory == owner.directory else { invalidate(); return }
        }
        guard let pending = request else { return }
        guard let target, target.worker == pending.target.worker,
              target.modelInstance == pending.target.modelInstance,
              target.conversation == pending.target.conversation, target.modelKey == pending.target.modelKey,
              target.directory == pending.target.directory else { invalidate(); return }
        let expected = pending.transition
        if let configuration = snapshot.sessionConfiguration,
           configuration.acceptedRequestID == expected.id,
           configuration.modelInstance == expected.selection.modelInstance,
           configuration.sessionInstance != expected.selection.sessionInstance,
           configuration.capacity == expected.capacity, configuration.chunk == expected.chunk,
           configuration.wholeSessionBudget == expected.wholeSessionBudget {
            request = nil; committed = true
            message = snapshot.restartRequired
                ? "Settings were applied, but cleanup failed. Restart before continuing. Your conversation and draft are kept."
                : "Applied. Cached tokens were reset; your conversation and draft are kept. The next Send re-encodes the conversation."
        } else if let result = snapshot.transition, result.request == expected, !result.committed {
            reject(result.message.isEmpty ? "The settings were not applied. Your current chat and edited values are kept." : result.message)
        } else if !snapshot.busy {
            request = nil; committed = false
            message = snapshot.sessionConfiguration == pending.target.configuration
                ? "Application cancelled. Current chat settings and your edited values are unchanged."
                : "Application ended without an active confirmation. Check the current chat settings before continuing."
        }
    }
}

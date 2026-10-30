import Foundation

// Correlation with an already loaded chat, never file authentication.
struct FriendlyPlanningTarget: Equatable {
    let worker: ObjectIdentifier
    let modelInstance: UInt64
    let conversation: UUID
    let modelKey, directory: String
    let profile: NativeContextLimits.LoadProfile
    let wholeBudget, textBudget: UInt64
}

// Created only after accepted load enqueue, promoted only at its completed
// successful Chat load. Keeping the directory does not extend a security lease.
struct FriendlyPlanningLoad {
    let worker: ObjectIdentifier
    let settings: NativeSettings
    private(set) var modelInstance: UInt64? = nil
    mutating func complete(worker: ObjectIdentifier, instance: UInt64?, chatEnabled: Bool) {
        guard worker == self.worker, chatEnabled, let instance, instance > 0 else {
            modelInstance = nil; return
        }
        modelInstance = instance
    }
    func target(worker: ObjectIdentifier, instance: UInt64, conversation: UUID,
                loadedKey: String, selectedKey: String) -> FriendlyPlanningTarget? {
        guard worker == self.worker, modelInstance == instance, instance > 0,
              !loadedKey.isEmpty, loadedKey == selectedKey else { return nil }
        return FriendlyPlanningTarget(worker:worker,modelInstance:instance,conversation:conversation,
            modelKey:loadedKey,directory:settings.directory,profile:.ordinary,
            wholeBudget:settings.numericalBudget,textBudget:settings.textBudget)
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
        if let next { draft.budgetMiB = String(next.wholeBudget / (1024 * 1024)) }
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

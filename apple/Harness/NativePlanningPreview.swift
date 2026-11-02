import Foundation
import CToklet

// Correlation identities, never artifact/payload authentication or load permits.
struct NativePreviewSelection: Sendable, Equatable {
    let id: UUID
    let directory: String
    let profile: NativeContextLimits.LoadProfile
}

struct NativePreviewRequest: Sendable, Equatable {
    // Profile metadata and numerical estimator provenance are independent.
    // An advisory estimate never proves that reopened files are resident bytes.
    enum Estimator: String, Sendable { case sourceProfile, residentAdvisory }
    struct Resident: Sendable, Equatable {
        let modelInstance, sessionInstance: UInt64
        let loadedUpper: UInt32
    }
    let id: UUID
    let selection: NativePreviewSelection
    let upper, chunk, offset: UInt32
    let contextCeiling: UInt32?
    let loadBudget, sessionBudget: UInt64
    var kvBudgetBytes: UInt64? = nil
    var estimator: Estimator = .sourceProfile
    var resident: Resident? = nil
    var normalization: NativeCapacityPlan.Normalization = .compensated
    var readback: NativeCapacityPlan.Readback = .fullLogits
    var attention: NativeCapacityPlan.AttentionPolicy = .nativeFactory

    func validate() throws {
        guard !selection.directory.isEmpty, selection.directory.utf8.count <= 16384,
              !selection.directory.utf8.contains(0), (1...32768).contains(upper),
              (1...1024).contains(chunk), chunk <= upper, offset < 32768,
              contextCeiling.map({ (1...32768).contains($0) }) ?? true,
              loadBudget > 0, loadBudget <= 6 * 1024 * 1024 * 1024,
              sessionBudget <= 6 * 1024 * 1024 * 1024,
              kvBudgetBytes.map({ $0 <= 6 * 1024 * 1024 * 1024 }) ?? true else {
            throw NativeContextFailure("Invalid planning preview request")
        }
        switch estimator {
        case .sourceProfile:
            guard resident == nil else { throw NativeContextFailure("Profile preview cannot claim a resident owner") }
        case .residentAdvisory:
            guard selection.profile == .smolAudited, let resident,
                  resident.modelInstance > 0, resident.sessionInstance > 0,
                  resident.modelInstance != resident.sessionInstance,
                  (1...512).contains(resident.loadedUpper), offset == 0,
                  normalization == .compensated, readback == .fullLogits, attention == .nativeFactory else {
                throw NativeContextFailure("Invalid resident advisory estimate")
            }
        }
    }

    // Called under the worker condition at enqueue/inspection/publication. Busy
    // is intentionally not tested here: this preview itself owns the busy slot.
    func matchesResident(_ state: NativeSnapshot) -> Bool {
        if estimator == .sourceProfile { return resident == nil }
        guard selection.profile == .smolAudited, let resident,
              !state.closed, !state.restartRequired, state.hasModel, state.hasSession, state.chatEnabled,
              let model = state.modelContext, let live = state.liveContext,
              let configuration = state.sessionConfiguration, let identity = state.acceptedIdentity,
              identity.profile == .smol, identity.matches(model) else { return false }
        return model.modelInstance == resident.modelInstance && model.loadedLimit == UInt64(resident.loadedUpper)
            && live.modelInstance == resident.modelInstance && live.sessionInstance == resident.sessionInstance
            && live.loadedLimit == UInt64(resident.loadedUpper)
            && configuration.modelInstance == resident.modelInstance && configuration.sessionInstance == resident.sessionInstance
            && UInt64(configuration.capacity) == live.capacity
    }

    // The upper is relative capacity; context and offset are absolute positions.
    // This derives a request from metadata only, never a source identity proof.
    func capacityRequest(planningUpper: UInt32) throws -> NativeCapacityPlan.Request {
        try validate()
        guard planningUpper > 0, planningUpper <= 32768 else { throw NativeContextFailure("Invalid planning upper") }
        let sourceUpper = min(planningUpper,contextCeiling ?? planningUpper)
        let context = min(sourceUpper,resident?.loadedUpper ?? sourceUpper)
        let (remaining,underflow) = context.subtractingReportingOverflow(offset)
        guard !underflow, remaining > 0 else { throw NativeContextFailure("Offset leaves no planning capacity") }
        let capacity = min(upper,remaining)
        var request = NativeCapacityPlan.Request(context:context,upper:capacity,chunk:chunk,offset:offset,
            loadBudget:loadBudget,sessionBudget:sessionBudget)
        request.kvBudgetBytes = kvBudgetBytes
        request.normalization = normalization; request.readback = readback; request.attention = attention
        try request.validate()
        return request
    }
}

struct NativePreviewResult: Sendable {
    enum Outcome: String, Sendable { case planned, metadataOnly, failed }
    enum Limitation: String, Sendable { case auditedProfileHasNoCapacityAPI, metadataUnsupported, backendUnavailable }
    let request: NativePreviewRequest
    let outcome: Outcome
    let limits: NativeContextLimits?
    let capacityRequest: NativeCapacityPlan.Request?
    let capacity: NativeCapacityPlan?
    let limitation: Limitation?
    let status: UInt32
    let message: String
    var isResidentAdvisoryEstimate: Bool { request.estimator == .residentAdvisory }
    // A bounded planning/application reservation, separate from numerical fit.
    // One current result only; no history. Does not include an existing model's
    // coexistence cost, tokenizer or a whole-app/device qualification.
    static let requiredApplicationAllowance = NativeContextLimits.requiredApplicationAllowance
        + NativeCapacityPlan.requiredApplicationAllowance + 64 * 1024
}

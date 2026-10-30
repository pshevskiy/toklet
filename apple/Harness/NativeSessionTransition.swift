import Foundation
import CToklet

// Caller correlation, not a path-based authorization. Native IDs are checked
// again against the actual owners on their thread and at locked publication.
struct NativeTransitionSelection: Equatable, Sendable {
    let id: UUID
    let modelInstance, sessionInstance: UInt64
}
struct NativeTransitionRequest: Equatable, Sendable {
    let id: UUID
    let selection: NativeTransitionSelection
    let capacity, chunk: UInt32
    let wholeSessionBudget, coexistenceBudget: UInt64
    let replayReserve, textReserve, applicationReserve: UInt64
    func validate() throws {
        guard selection.modelInstance > 0, selection.sessionInstance > 0,
              selection.modelInstance != selection.sessionInstance,
              (1...32768).contains(capacity), (1...1024).contains(chunk), chunk <= capacity,
              wholeSessionBudget > UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES),
              [wholeSessionBudget,coexistenceBudget,replayReserve,textReserve,applicationReserve]
                .allSatisfy({ $0 <= 6 * 1024 * 1024 * 1024 }) else {
            throw NativeContextFailure("Invalid empty-session transition request")
        }
    }
    func options() -> TokletOptions {
        TokletOptions(abi_version:TOKLET_ABI_VERSION,struct_size:UInt32(MemoryLayout<TokletOptions>.size),
            capacity:capacity,chunk:chunk,memory_limit_bytes:wholeSessionBudget)
    }
    func accounting() -> TokletTransitionPlanningOptions {
        TokletTransitionPlanningOptions(abi_version:TOKLET_ABI_VERSION,
            struct_size:UInt32(MemoryLayout<TokletTransitionPlanningOptions>.size),
            expected_model_instance:selection.modelInstance,expected_session_instance:selection.sessionInstance,
            transition_budget_bytes:coexistenceBudget,replay_reserve_bytes:replayReserve,
            text_reserve_bytes:textReserve,application_reserve_bytes:applicationReserve,reserved:0)
    }
}

// Strict bounded consumer of D1. Fit is accounting, not a reservation or app/
// device qualification. D2 independently re-admits actual candidate storage.
struct NativeTransitionPlan: Decodable, Sendable {
    struct Configuration: Decodable, Sendable {
        let capacity, chunk, positionOffset, individualNumericalBudgetBytes: UInt64
    }
    struct Shared: Decodable, Sendable {
        let learnedBytes, positionalBytes, modelTableBytes, totalBytes: UInt64
    }
    struct Owned: Decodable, Sendable {
        let kvBytes, scratchBytes, historyBytes, logitsBytes, sessionTableBytes, totalBytes: UInt64
    }
    struct Reserves: Decodable, Sendable {
        let currentMetadataDriverBytes, candidateMetadataDriverBytes, currentBridgeBytes, candidateBridgeBytes: UInt64
        let jsonSnapshotBytes, replayBytes, textBytes, applicationBytes, totalBytes: UInt64
    }
    struct Modes: Decodable, Sendable { let projection, normalization, attention, readback: String }
    let schemaVersion: UInt32
    let reportKind, backend, modeSelection: String
    let modelInstance, sessionInstance, loadedPositionLimit: UInt64
    let current, candidate: Configuration
    let historyTokens, committedTokens, pendingTokens: UInt64
    let sharedModel: Shared
    let currentAllocated, candidatePlanned: Owned
    let reserves: Reserves
    let executionModes: Modes
    let aggregateRequiredBytes, transitionBudgetBytes, candidateConstructorRequiredBytes, candidateWholeBudgetBytes: UInt64
    let candidateLargestDeviceBufferBytes, deviceMaximumBufferBytes: UInt64?
    let fitsTransitionBudget, fitsCandidateIndividualBudget: Bool
    let candidateAllocated, reservesVerified, creationRevalidationRequired: Bool
    let contextQualificationEvaluated, wholeAppDeviceFitEvaluated: Bool

    static func sum(_ parts: [UInt64]) throws -> UInt64 {
        var total: UInt64 = 0
        for part in parts {
            let next = total.addingReportingOverflow(part)
            guard !next.overflow else { throw NativeContextFailure("Transition accounting overflow") }
            total = next.partialValue
        }
        return total
    }
    static func decode(_ data: Data, request: NativeTransitionRequest,
                       current facts: NativeContextFacts, chunk: UInt32, wholeBudget: UInt64) throws -> Self {
        try request.validate()
        guard !data.isEmpty, data.count <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES) else {
            throw NativeContextFailure("Invalid transition report length")
        }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self,from:data)
        let bridge = UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES)
        guard wholeBudget > bridge,
              value.schemaVersion == 1, value.reportKind == "same_model_session_transition",
              value.modeSelection == "existing_native_factory",
              value.modelInstance == request.selection.modelInstance, value.modelInstance == facts.modelInstance,
              value.sessionInstance == request.selection.sessionInstance, value.sessionInstance == facts.sessionInstance,
              value.backend == (facts.backend == TOKLET_CONTEXT_CPU ? "cpu" : "metal"),
              value.loadedPositionLimit == facts.loadedLimit,
              value.current.capacity == facts.capacity, value.current.chunk == UInt64(chunk),
              value.current.positionOffset == facts.offset,
              value.current.individualNumericalBudgetBytes == wholeBudget - bridge,
              value.candidate.capacity == UInt64(request.capacity), value.candidate.chunk == UInt64(request.chunk),
              value.candidate.positionOffset == 0, UInt64(request.capacity) <= facts.loadedLimit,
              value.candidate.individualNumericalBudgetBytes == request.wholeSessionBudget - bridge,
              value.historyTokens == facts.history, value.committedTokens == facts.committed,
              value.pendingTokens == facts.pending,
              !value.candidateAllocated, !value.reservesVerified, value.creationRevalidationRequired,
              !value.contextQualificationEvaluated, !value.wholeAppDeviceFitEvaluated,
              value.transitionBudgetBytes == request.coexistenceBudget,
              value.candidateWholeBudgetBytes == request.wholeSessionBudget else {
            throw NativeContextFailure("Stale or inconsistent transition report")
        }
        let shared = value.sharedModel, old = value.currentAllocated, next = value.candidatePlanned, r = value.reserves
        // CPU Memory includes the shared model table; Metal SessionPlan does not.
        // The aggregate below still counts that physical table exactly once.
        let constructorModelTable = facts.backend == TOKLET_CONTEXT_CPU ? shared.modelTableBytes : 0
        func owned(_ x: Owned) throws -> Bool {
            x.totalBytes == (try sum([x.kvBytes,x.scratchBytes,x.historyBytes,x.logitsBytes,x.sessionTableBytes]))
        }
        guard shared.totalBytes == (try sum([shared.learnedBytes,shared.positionalBytes,shared.modelTableBytes])),
              try owned(old), try owned(next),
              r.currentMetadataDriverBytes == 64 * 1024 * 1024,
              r.candidateMetadataDriverBytes == 64 * 1024 * 1024,
              r.currentBridgeBytes == bridge, r.candidateBridgeBytes == bridge,
              r.jsonSnapshotBytes == 2 * UInt64(TOKLET_MAX_COMPATIBILITY_JSON_BYTES) + 256,
              r.replayBytes == request.replayReserve, r.textBytes == request.textReserve,
              r.applicationBytes == request.applicationReserve,
              r.totalBytes == (try sum([r.currentMetadataDriverBytes,r.candidateMetadataDriverBytes,
                r.currentBridgeBytes,r.candidateBridgeBytes,r.jsonSnapshotBytes,r.replayBytes,r.textBytes,r.applicationBytes])),
              value.aggregateRequiredBytes == (try sum([shared.totalBytes,old.totalBytes,next.totalBytes,r.totalBytes])),
              value.candidateConstructorRequiredBytes == (try sum([shared.learnedBytes,shared.positionalBytes,constructorModelTable,
                next.totalBytes,r.candidateMetadataDriverBytes,bridge])),
              value.fitsTransitionBudget == (value.aggregateRequiredBytes <= request.coexistenceBudget),
              value.fitsCandidateIndividualBudget == (value.candidateConstructorRequiredBytes <= request.wholeSessionBudget),
              value.executionModes.readback == "FullLogits" else {
            throw NativeContextFailure("Transition ledger or fit flags differ")
        }
        // Context admission is historical; D1 counts current actual capacities.
        // They need not be equal after retained host vectors grow.
        if facts.backend == TOKLET_CONTEXT_CPU {
            guard value.candidateLargestDeviceBufferBytes == nil, value.deviceMaximumBufferBytes == nil,
                  value.executionModes.projection == "Scalar", value.executionModes.normalization == "Scalar",
                  value.executionModes.attention == "Scalar" else { throw NativeContextFailure("Invalid CPU transition modes") }
        } else {
            guard let largest = value.candidateLargestDeviceBufferBytes, let maximum = value.deviceMaximumBufferBytes,
                  largest > 0, largest <= maximum, largest <= UInt64(Int.max), largest <= value.candidateConstructorRequiredBytes,
                  ["MlxAffine","Q4CompensatedTiled","HalfCompensated"].contains(value.executionModes.projection),
                  ["Original","Compensated"].contains(value.executionModes.normalization),
                  ["Reference","Online"].contains(value.executionModes.attention) else { throw NativeContextFailure("Invalid Metal transition modes") }
        }
        return value
    }

    static func validateCandidate(_ next: NativeContextFacts, old: NativeContextFacts,
                                  request: NativeTransitionRequest) throws {
        guard next.modelInstance == old.modelInstance, next.sessionInstance != old.sessionInstance,
              next.backend == old.backend, next.declaredTotal == old.declaredTotal,
              next.declaredInput == old.declaredInput, next.declaredOutput == old.declaredOutput,
              next.engineGlobalCeiling == old.engineGlobalCeiling, next.loadedLimit == old.loadedLimit,
              next.capacity == UInt64(request.capacity), next.offset == 0,
              next.history == 0, next.committed == 0, next.pending == 0,
              next.numericalAdmission <= request.wholeSessionBudget else {
            throw NativeContextFailure("Candidate model/context differs or cache is not empty")
        }
    }

    // Only owning-thread Engine calls this. No raw native handle is retained.
    static func inspect(model: OpaquePointer, session: OpaquePointer, request: NativeTransitionRequest,
                        current: NativeContextFacts, chunk: UInt32, wholeBudget: UInt64) throws -> Self {
        var options = request.options(), accounting = request.accounting(), error = TokletError()
        var owner: OpaquePointer?
        func checked(_ code: UInt32) throws {
            guard code == TOKLET_OK else {
                let message = withUnsafeBytes(of:&error.message) { String(decoding:$0.prefix { $0 != 0 },as:UTF8.self) }
                throw NSError(domain:"TokletTransition",code:Int(code),userInfo:[NSLocalizedDescriptionKey:message])
            }
        }
        try checked(toklet_session_transition_plan_create(model,session,&options,&accounting,&owner,&error))
        guard let owner else { throw NativeContextFailure("Missing transition report") }
        defer { _ = toklet_mlx_inspect_release(owner,nil) }
        var count = 0
        try checked(toklet_mlx_inspect_json(owner,nil,0,&count,&error))
        guard (1...Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES)).contains(count) else { throw NativeContextFailure("Invalid transition report size") }
        var bytes = [UInt8](repeating:0,count:count)
        let expected = count
        try bytes.withUnsafeMutableBufferPointer { try checked(toklet_mlx_inspect_json(owner,$0.baseAddress,$0.count,&count,&error)) }
        guard count == expected else { throw NativeContextFailure("Transition report changed size") }
        return try decode(Data(bytes),request:request,current:current,chunk:chunk,wholeBudget:wholeBudget)
    }
}

struct NativeTransitionResult: Sendable {
    enum Outcome: String, Sendable { case rejected, committed, cleanupFailed }
    let request: NativeTransitionRequest
    let outcome: Outcome
    let committed: Bool
    let plan: NativeTransitionPlan?
    let candidate: NativeContextFacts?
    let status: UInt32
    let message: String
}

// Diagnostic-only retirement controls. Production leaves this nil; the panic
// case actually releases first and then reports PANIC, never double-frees.
enum NativeTransitionRetirementFault: Sendable, Equatable { case busy, consumedPanic }

import Foundation
import CToklet

// Cold metadata only. No architecture interpretation, text profile lookup or
// inference owner is duplicated here. Release C ownership before returning.
struct NativeCompatibility: Decodable, Sendable {
    enum Backend: String, Decodable, Sendable { case metal, scalarReference = "scalar_reference" }
    enum Status: String, Decodable, Sendable { case unknown, unsupported, compatibleUntested = "compatible_untested" }
    enum Admission: String, Decodable, Sendable {
        case notEvaluated = "not_evaluated", exceedsBudget = "exceeds_budget"
        case fitsScalarLoadEstimate = "fits_scalar_load_estimate", devicePlanningRequired = "device_planning_required"
    }
    enum Reason: String, Decodable, Sendable {
        case invalidRequest = "invalid_request", invalidConfiguration = "invalid_configuration"
        case unknownArchitecture = "unknown_architecture", artifactRejected = "artifact_rejected"
        case unsupportedGeometry = "unsupported_geometry", loadBudgetExceeded = "load_budget_exceeded"
        case backendUnavailable = "backend_unavailable", metadataAccepted = "metadata_accepted"
    }
    struct Geometry: Decodable, Sendable { let hiddenSize, queryWidth, kvWidth: UInt64 }
    struct Resources: Decodable, Sendable { let sourcePayloadBytes: UInt64; let metalRuntimePlanningRequired: Bool }
    let schemaVersion: UInt32
    let backend: Backend
    let status: Status
    let reason: Reason
    let message: String
    let architecture: String?
    let requestedContext: UInt32
    let requestedMemoryLimitBytes: UInt64
    let geometry: Geometry?
    let resources: Resources?
    let tensorMetadataValidated: Bool
    let backendAvailableInBuild: Bool
    let loadAdmission: Admission
    let payloadValidationRequired, runtimeDeviceChecksRequired, sessionAdmissionRequired: Bool
    let textProfileChecked, qualificationEvaluated: Bool

    // Buffers plus bounded decoded fields, outside numerical admission. This is
    // an additional allowance requirement, not an enforced reservation or a
    // numerical, allocator-overhead or process-peak claim.
    static let requiredApplicationAllowance: UInt64 = UInt64(TOKLET_COMPATIBILITY_JSON_ALLOWANCE_BYTES) + 512 * 1024
    var mayAttemptLoad: Bool {
        status == .compatibleUntested && backendAvailableInBuild && loadAdmission != .exceedsBudget
    }
    var summary: String {
        switch status {
        case .unknown: "Architecture not implemented"
        case .unsupported: "Metadata rejected: " + String(message.prefix(512))
        case .compatibleUntested:
            loadAdmission == .exceedsBudget ? "Metadata supported; load estimate exceeds budget" :
                !backendAvailableInBuild ? "Metadata supported; backend unavailable in this build" :
                "Metadata supported; runtime and text checks remain"
        }
    }
    var rejectionCode: UInt32 {
        switch reason {
        case .invalidRequest: TOKLET_INVALID
        case .unknownArchitecture, .unsupportedGeometry, .backendUnavailable: TOKLET_UNSUPPORTED
        default: TOKLET_LOAD_ERROR
        }
    }
    static func decode(_ data: Data, backend: Backend, context: UInt32, budget: UInt64) throws -> Self {
        guard !data.isEmpty, data.count <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES) else { throw failure("Invalid metadata report length") }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self,from:data)
        guard value.schemaVersion == 1, value.backend == backend,
              value.requestedContext == context, value.requestedMemoryLimitBytes == budget,
              value.payloadValidationRequired, value.sessionAdmissionRequired,
              value.runtimeDeviceChecksRequired == (backend == .metal),
              !value.textProfileChecked, !value.qualificationEvaluated else { throw failure("Metadata report does not match the request or schema") }
        if value.status == .compatibleUntested {
            guard value.tensorMetadataValidated, value.geometry != nil, value.resources != nil,
                  value.architecture?.isEmpty == false,
                  value.resources?.metalRuntimePlanningRequired == (backend == .metal),
                  value.loadAdmission == .exceedsBudget || value.loadAdmission == (backend == .metal ? .devicePlanningRequired : .fitsScalarLoadEstimate),
                  [.metadataAccepted,.backendUnavailable,.loadBudgetExceeded].contains(value.reason) else { throw failure("Incomplete compatible metadata report") }
        }
        switch value.status {
        case .compatibleUntested:
            let ordinary: Admission = backend == .metal ? .devicePlanningRequired : .fitsScalarLoadEstimate
            switch (value.reason,value.backendAvailableInBuild,value.loadAdmission) {
            case (.metadataAccepted,true,let admission), (.backendUnavailable,false,let admission):
                guard admission == ordinary else { throw failure("Contradictory backend admission report") }
            case (.loadBudgetExceeded,_,.exceedsBudget): break
            default: throw failure("Contradictory metadata admission report")
            }
        case .unknown:
            guard value.reason == .unknownArchitecture else { throw failure("Contradictory unknown architecture report") }
        case .unsupported:
            guard [.invalidRequest,.invalidConfiguration,.artifactRejected,.unsupportedGeometry,.loadBudgetExceeded].contains(value.reason) else { throw failure("Contradictory unsupported report") }
        }
        guard backend != .scalarReference || value.backendAvailableInBuild else { throw failure("Scalar backend must be available") }
        return value
    }
    private static func failure(_ message: String, code: UInt32 = TOKLET_INVALID) -> NSError {
        NSError(domain:"TokletCompatibility",code:Int(code),userInfo:[NSLocalizedDescriptionKey:message])
    }
    // Synchronous: create, query, copy and release all stay on this OS thread.
    static func inspect(directory: String, backend: Backend = .metal, context: UInt32, budget: UInt64) throws -> Self {
        let path = Array(directory.utf8)
        guard !path.isEmpty, path.count <= 16384, !path.contains(0) else { throw failure("Invalid artifact path") }
        var options = TokletCompatibilityOptions(abi_version:TOKLET_ABI_VERSION,
            struct_size:UInt32(MemoryLayout<TokletCompatibilityOptions>.size),backend:backend == .metal ? 1:0,
            context_limit:context,memory_limit_bytes:budget)
        var owner: OpaquePointer?; var error = TokletError()
        func checked(_ code: UInt32) throws {
            if code != TOKLET_OK {
                let message = withUnsafeBytes(of:&error.message) { String(decoding:$0.prefix { $0 != 0 },as:UTF8.self) }
                throw failure(message.isEmpty ? "Metadata inspection failed" : message,code:code)
            }
        }
        try path.withUnsafeBufferPointer { try checked(toklet_mlx_inspect_create($0.baseAddress,$0.count,&options,&owner,&error)) }
        guard let owner else { throw failure("Missing metadata report") }
        defer { _ = toklet_mlx_inspect_release(owner,nil) }
        var count = 0
        try checked(toklet_mlx_inspect_json(owner,nil,0,&count,&error))
        guard (1...Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES)).contains(count) else { throw failure("Invalid metadata report size") }
        var bytes = [UInt8](repeating:0,count:count)
        try bytes.withUnsafeMutableBufferPointer { try checked(toklet_mlx_inspect_json(owner,$0.baseAddress,$0.count,&count,&error)) }
        guard count == bytes.count else { throw failure("Metadata report size changed") }
        return try decode(Data(bytes),backend:backend,context:context,budget:budget)
    }
}

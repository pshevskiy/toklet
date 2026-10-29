import Foundation
import CToklet

// Historical metadata for one existing loader profile. No payload identity,
// runtime budget, loaded instance or context qualification is established here.
struct NativeContextLimits: Decodable, Sendable {
    enum LoadProfile: String, Decodable, Sendable {
        case ordinary, smolAudited = "smol_audited"
        var raw: UInt32 { self == .ordinary ? 0 : 1 }
    }
    let schemaVersion: UInt32
    let reportKind: String
    let loadProfile: LoadProfile
    let status: NativeCompatibility.Status
    let reason: NativeCompatibility.Reason
    let message: String
    let architecture: String?
    let declaredTotalPositions, graphPositionCap, declaredSlidingWindow, factoryPositionUpper: UInt64?
    let engineGlobalPositionCeiling, metadataReaderLimitBytes: UInt64
    let backendAvailableInBuild, tensorMetadataValidated, auditedSourceMetadataValidated: Bool
    let completeSourceAuthenticationRequired, payloadValidationRequired: Bool
    let runtimeDeviceChecksRequired, sessionAdmissionRequired: Bool
    let budgetFitEvaluated, qualificationEvaluated, textProfileChecked, wholeAppDeviceFitEvaluated: Bool
    static let requiredApplicationAllowance = NativeCompatibility.requiredApplicationAllowance

    // A usable metadata upper for later planning, not permission to load.
    var planningUpper: UInt32? {
        guard status == .compatibleUntested, backendAvailableInBuild,
              let upper = factoryPositionUpper else { return nil }
        return UInt32(exactly:upper)
    }
    static func decode(_ data: Data, profile: LoadProfile) throws -> Self {
        guard !data.isEmpty, data.count <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES) else { throw failure("Invalid native limits length") }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self,from:data)
        guard value.schemaVersion == 1, value.reportKind == "native_mlx_context_limits",
              value.loadProfile == profile, value.engineGlobalPositionCeiling == 32768,
              value.metadataReaderLimitBytes == 6 * 1024 * 1024 * 1024,
              value.payloadValidationRequired, value.runtimeDeviceChecksRequired, value.sessionAdmissionRequired,
              !value.budgetFitEvaluated, !value.qualificationEvaluated, !value.textProfileChecked,
              !value.wholeAppDeviceFitEvaluated,
              value.completeSourceAuthenticationRequired == (profile == .smolAudited),
              value.auditedSourceMetadataValidated == (profile == .smolAudited && value.tensorMetadataValidated),
              value.architecture.map({ !$0.isEmpty && $0.utf8.count <= 128 }) ?? true else {
            throw failure("Native limits do not match the requested profile or metadata contract")
        }
        if value.tensorMetadataValidated {
            guard value.architecture != nil, let declared = value.declaredTotalPositions, declared > 0 else { throw failure("Missing validated declaration") }
        } else {
            guard value.architecture == nil, value.declaredTotalPositions == nil else { throw failure("Unvalidated declaration presented as a fact") }
        }
        switch value.status {
        case .compatibleUntested:
            guard value.tensorMetadataValidated, let declared = value.declaredTotalPositions,
                  let upper = value.factoryPositionUpper, upper > 0,
                  value.graphPositionCap.map({ $0 > 0 }) ?? true,
                  value.declaredSlidingWindow.map({ $0 > 0 }) ?? true,
                  upper == min(min(declared,value.engineGlobalPositionCeiling),
                               min(value.graphPositionCap ?? UInt64.max,value.declaredSlidingWindow ?? UInt64.max)),
                  value.reason == (value.backendAvailableInBuild ? .metadataAccepted : .backendUnavailable) else {
                throw failure("Contradictory accepted native limits")
            }
        case .unknown:
            guard value.reason == .unknownArchitecture, !value.tensorMetadataValidated,
                  profile == .ordinary else { throw failure("Contradictory unknown native limits") }
        case .unsupported:
            guard [.invalidConfiguration,.artifactRejected,.unsupportedGeometry,.loadBudgetExceeded].contains(value.reason),
                  value.tensorMetadataValidated == (value.reason == .unsupportedGeometry) else { throw failure("Contradictory rejected native limits") }
        }
        if value.status != .compatibleUntested {
            guard value.factoryPositionUpper == nil, value.graphPositionCap == nil,
                  value.declaredSlidingWindow == nil else { throw failure("Rejected metadata carries a usable limit") }
        }
        return value
    }
    private static func failure(_ message: String, code: UInt32 = TOKLET_INVALID) -> NSError {
        NSError(domain:"TokletContextLimits",code:Int(code),userInfo:[NSLocalizedDescriptionKey:message])
    }
    // Same OS thread owns create/query/copy/release. Metadata is revalidated by
    // actual loading; directory equality is never a source identity guarantee.
    static func inspect(directory: String, profile: LoadProfile = .ordinary) throws -> Self {
        let path = Array(directory.utf8)
        guard !path.isEmpty, path.count <= 16384, !path.contains(0) else { throw failure("Invalid artifact path") }
        var options = TokletNativeLimitsOptions(abi_version:TOKLET_ABI_VERSION,
            struct_size:UInt32(MemoryLayout<TokletNativeLimitsOptions>.size),load_profile:profile.raw,reserved:0)
        var owner: OpaquePointer?; var error = TokletError()
        func checked(_ code: UInt32) throws {
            if code != TOKLET_OK {
                let message = withUnsafeBytes(of:&error.message) { String(decoding:$0.prefix { $0 != 0 },as:UTF8.self) }
                throw failure(message.isEmpty ? "Native limits inspection failed" : message,code:code)
            }
        }
        try path.withUnsafeBufferPointer { try checked(toklet_mlx_context_limits_create($0.baseAddress,$0.count,&options,&owner,&error)) }
        guard let owner else { throw failure("Missing native limits report") }
        defer { _ = toklet_mlx_inspect_release(owner,nil) }
        var count = 0
        try checked(toklet_mlx_inspect_json(owner,nil,0,&count,&error))
        guard (1...Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES)).contains(count) else { throw failure("Invalid native limits size") }
        var bytes = [UInt8](repeating:0,count:count)
        try bytes.withUnsafeMutableBufferPointer { try checked(toklet_mlx_inspect_json(owner,$0.baseAddress,$0.count,&count,&error)) }
        guard count == bytes.count else { throw failure("Native limits size changed") }
        return try decode(Data(bytes),profile:profile)
    }
}

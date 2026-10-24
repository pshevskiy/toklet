import Foundation
import CToklet

// Cold resource facts only. The C factory still revalidates and owns allocations.
struct NativeResourcePlan: Decodable, Sendable {
    enum Attention: String, Decodable, Sendable { case reference = "Reference", online = "Online" }
    struct Memory: Decodable, Sendable {
        let weightsBytes, kvBytes, scratchBytes, historyBytes, logitsBytes: UInt64
        let metadataBytes, metadataAllowanceBytes, admittedBytes: UInt64
        var sum: UInt64? {
            var total: UInt64 = 0
            for bytes in [weightsBytes,kvBytes,scratchBytes,historyBytes,logitsBytes,metadataBytes,metadataAllowanceBytes] {
                let (next,overflow) = total.addingReportingOverflow(bytes)
                if overflow { return nil }; total = next
            }
            return total
        }
    }
    let schemaVersion: UInt32
    let deviceName: String
    let maximumBufferBytes: UInt64
    let requestedLoadContext, requestedCapacity, requestedChunkSize, requestedPositionOffset: UInt32
    let requestedLoadBudgetBytes, requestedSessionBudgetBytes: UInt64
    let projection, normalization, readback: String
    let attention: Attention
    let sourcePayloadBytes, sourceAdmittedLoadBytes: UInt64
    let residentWeightsBytes, residentAdmittedLoadBytes: UInt64
    let sessionMemory: Memory
    let fitsSourceLoadBudget, fitsResidentLoadBudget, fitsSessionBudget, fitsRequestedBudgets: Bool
    let numericalBuffersAllocated, payloadValidationRequired, qualificationEvaluated: Bool

    // Conservative logical requirement for both retained summaries/decoding.
    // This is outside numerical admission, not an enforced reservation or peak.
    static let requiredApplicationAllowance = NativeCompatibility.requiredApplicationAllowance * 2
    var summary: String {
        fitsRequestedBudgets ? "Memory plan fits; payload and text checks remain" : "Memory plan exceeds the numerical budget"
    }
    static func numericalBudget(wholeBudget: UInt64) throws -> UInt64 {
        let bridge = UInt64(TOKLET_MODEL_SESSION_BRIDGE_BYTES)
        guard wholeBudget > bridge, wholeBudget <= 6 * 1024 * 1024 * 1024 else { throw failure("Native budget must be above its owner allowance and at most 6 GiB",code:TOKLET_INVALID) }
        return wholeBudget - bridge
    }
    static func decode(_ data: Data, capacity: UInt32, chunk: UInt32, budget: UInt64) throws -> Self {
        guard !data.isEmpty, data.count <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES) else { throw failure("Invalid runtime report length") }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self,from:data)
        guard value.schemaVersion == 1, !value.deviceName.isEmpty, value.deviceName.utf8.count <= 512,
              value.maximumBufferBytes > 0,
              value.requestedLoadContext == capacity, value.requestedCapacity == capacity,
              value.requestedChunkSize == chunk, value.requestedPositionOffset == 0,
              value.requestedLoadBudgetBytes == budget, value.requestedSessionBudgetBytes == budget,
              value.projection == "MlxAffine", value.normalization == "Compensated", value.readback == "FullLogits",
              !value.numericalBuffersAllocated, value.payloadValidationRequired, !value.qualificationEvaluated,
              value.sessionMemory.weightsBytes == value.residentWeightsBytes,
              value.sessionMemory.sum == value.sessionMemory.admittedBytes,
              value.residentWeightsBytes > 0, value.sourcePayloadBytes > 0,
              value.sourceAdmittedLoadBytes >= value.sourcePayloadBytes,
              value.residentAdmittedLoadBytes >= value.residentWeightsBytes,
              value.fitsSourceLoadBudget == (value.sourceAdmittedLoadBytes <= budget),
              value.fitsResidentLoadBudget == (value.residentAdmittedLoadBytes <= budget),
              value.fitsSessionBudget == (value.sessionMemory.admittedBytes <= budget),
              value.fitsRequestedBudgets == (value.fitsSourceLoadBudget && value.fitsResidentLoadBudget && value.fitsSessionBudget) else {
            throw failure("Runtime report does not match the request or accounting contract")
        }
        return value
    }
    private static func failure(_ message: String, code: UInt32 = TOKLET_INVALID) -> NSError {
        NSError(domain:"TokletResourcePlan",code:Int(code),userInfo:[NSLocalizedDescriptionKey:message])
    }
    // Exact numerical budget, after the native owner allowance. Native policy2
    // resolves attention; the app does not interpret architecture names.
    static func inspect(directory: String, capacity: UInt32, chunk: UInt32, budget: UInt64) throws -> Self {
        let path = Array(directory.utf8)
        guard !path.isEmpty, path.count <= 16384, !path.contains(0),
              (1...32768).contains(capacity), chunk > 0, chunk <= 1024, chunk <= capacity,
              budget > 0, budget <= 6 * 1024 * 1024 * 1024 else { throw failure("Invalid runtime inspection request") }
        var options = TokletRuntimePlanningOptions(abi_version:TOKLET_ABI_VERSION,
            struct_size:UInt32(MemoryLayout<TokletRuntimePlanningOptions>.size),context_limit:capacity,
            capacity:capacity,chunk:chunk,position_offset:0,normalization:1,readback:0,attention:2,reserved:0,
            load_memory_limit_bytes:budget,session_memory_limit_bytes:budget)
        var owner: OpaquePointer?; var error = TokletError()
        func checked(_ code: UInt32) throws {
            if code != TOKLET_OK {
                let message = withUnsafeBytes(of:&error.message) { String(decoding:$0.prefix { $0 != 0 },as:UTF8.self) }
                throw failure(message.isEmpty ? "Runtime inspection failed" : message,code:code)
            }
        }
        try path.withUnsafeBufferPointer { try checked(toklet_mlx_plan_create($0.baseAddress,$0.count,&options,&owner,&error)) }
        guard let owner else { throw failure("Missing runtime report") }
        defer { _ = toklet_mlx_inspect_release(owner,nil) }
        var count = 0
        try checked(toklet_mlx_inspect_json(owner,nil,0,&count,&error))
        guard (1...Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES)).contains(count) else { throw failure("Invalid runtime report size") }
        var bytes = [UInt8](repeating:0,count:count)
        try bytes.withUnsafeMutableBufferPointer { try checked(toklet_mlx_inspect_json(owner,$0.baseAddress,$0.count,&count,&error)) }
        guard count == bytes.count else { throw failure("Runtime report size changed") }
        return try decode(Data(bytes),capacity:capacity,chunk:chunk,budget:budget)
    }
}

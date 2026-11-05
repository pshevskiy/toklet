import Foundation
import CToklet

// Cold resource facts only. The C factory still revalidates and owns allocations.
struct NativeResourcePlan: Decodable, Sendable {
    enum Attention: String, Decodable, Sendable { case reference = "Reference", online = "Online" }
    struct Memory: Decodable, Sendable {
        let weightsBytes, kvBytes, scratchBytes, historyBytes, logitsBytes: UInt64
        let metadataBytes, metadataAllowanceBytes, admittedBytes: UInt64
        let positionalBytes: UInt64?
        var sum: UInt64? {
            var total: UInt64 = 0
            for bytes in [weightsBytes,kvBytes,scratchBytes,historyBytes,logitsBytes,metadataBytes,metadataAllowanceBytes,positionalBytes ?? 0] {
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
    let positionalResidentBytes: UInt64?
    let sessionMemory: Memory
    let fitsSourceLoadBudget, fitsResidentLoadBudget, fitsSessionBudget, fitsRequestedBudgets: Bool
    let numericalBuffersAllocated, payloadValidationRequired, qualificationEvaluated: Bool
    private var modelResidentBytes: UInt64? {
        let (sum, overflow) = residentWeightsBytes.addingReportingOverflow(positionalResidentBytes ?? 0)
        return overflow ? nil : sum
    }

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
    static func decode(_ data: Data, capacity: UInt32, chunk: UInt32, budget: UInt64, context: UInt32? = nil) throws -> Self {
        let context = context ?? capacity
        guard (1...32768).contains(context), capacity > 0, capacity <= context else { throw failure("Invalid load context or session capacity") }
        guard !data.isEmpty, data.count <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES) else { throw failure("Invalid runtime report length") }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self,from:data)
        return try value.validated(context:context,capacity:capacity,chunk:chunk,offset:0,
                                   loadBudget:budget,sessionBudget:budget,
                                   normalization:"Compensated",readback:"FullLogits",expectedAttention:nil)
    }
    // Context owns model positional resources; capacity owns session KV/scratch.
    // Same accounting guard used by the old single-capacity decoder. Request
    // domain validation belongs to each caller; the legacy decoder is unchanged.
    func validated(context: UInt32, capacity: UInt32, chunk: UInt32, offset: UInt32,
                   loadBudget: UInt64, sessionBudget: UInt64, normalization: String,
                   readback: String, expectedAttention: Attention?) throws -> Self {
        guard self.schemaVersion == 1, !self.deviceName.isEmpty, self.deviceName.utf8.count <= 512,
              self.maximumBufferBytes > 0,
              self.requestedLoadContext == context, self.requestedCapacity == capacity,
              self.requestedChunkSize == chunk, self.requestedPositionOffset == offset,
              self.requestedLoadBudgetBytes == loadBudget, self.requestedSessionBudgetBytes == sessionBudget,
              self.projection == "MlxAffine", self.normalization == normalization, self.readback == readback,
              expectedAttention == nil || self.attention == expectedAttention,
              !self.numericalBuffersAllocated, self.payloadValidationRequired, !self.qualificationEvaluated,
              self.sessionMemory.weightsBytes == self.residentWeightsBytes,
              (self.sessionMemory.positionalBytes ?? 0) == (self.positionalResidentBytes ?? 0),
              self.sessionMemory.sum == self.sessionMemory.admittedBytes,
              self.residentWeightsBytes > 0, self.sourcePayloadBytes > 0,
              self.sourceAdmittedLoadBytes >= self.sourcePayloadBytes,
              let residentBytes = self.modelResidentBytes,
              self.residentAdmittedLoadBytes >= residentBytes,
              self.fitsSourceLoadBudget == (self.sourceAdmittedLoadBytes <= loadBudget),
              self.fitsResidentLoadBudget == (self.residentAdmittedLoadBytes <= loadBudget),
              self.fitsSessionBudget == (self.sessionMemory.admittedBytes <= sessionBudget),
              self.fitsRequestedBudgets == (self.fitsSourceLoadBudget && self.fitsResidentLoadBudget && self.fitsSessionBudget) else {
            throw Self.failure("Runtime report does not match the request or accounting contract")
        }
        return self
    }
    private static func failure(_ message: String, code: UInt32 = TOKLET_INVALID) -> NSError {
        NSError(domain:"TokletResourcePlan",code:Int(code),userInfo:[NSLocalizedDescriptionKey:message])
    }
    // Exact numerical budget, after the native owner allowance. Native policy2
    // resolves attention; the app does not interpret architecture names.
    static func inspect(directory: String, capacity: UInt32, chunk: UInt32, budget: UInt64, context: UInt32? = nil, attention: NativeCapacityPlan.AttentionPolicy = .nativeFactory) throws -> Self {
        let context = context ?? capacity
        let path = Array(directory.utf8)
        guard !path.isEmpty, path.count <= 16384, !path.contains(0),
              (1...32768).contains(context), capacity > 0, capacity <= context, chunk > 0, chunk <= 1024, chunk <= capacity,
              budget > 0, budget <= 6 * 1024 * 1024 * 1024 else { throw failure("Invalid runtime inspection request") }
        var options = TokletRuntimePlanningOptions(abi_version:TOKLET_ABI_VERSION,
            struct_size:UInt32(MemoryLayout<TokletRuntimePlanningOptions>.size),context_limit:context,
            capacity:capacity,chunk:chunk,position_offset:0,normalization:1,readback:0,attention:attention.raw,reserved:0,
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
        let plan = try decode(Data(bytes),capacity:capacity,chunk:chunk,budget:budget,context:context)
        if let expected = attention.expected, plan.attention != expected {
            throw failure("Runtime plan attention differs from the request")
        }
        return plan
    }
}

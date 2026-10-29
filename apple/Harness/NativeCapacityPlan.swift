import Foundation
import CToklet

// A budget search within an already admitted caller upper. This API does not
// authenticate an audited-Smol profile or bind a separate limits report's files.
struct NativeCapacityPlan: Decodable, Sendable {
    enum Normalization: String, Sendable { case original = "Original", compensated = "Compensated" }
    enum Readback: String, Sendable { case fullLogits = "FullLogits", greedyToken = "GreedyToken" }
    enum AttentionPolicy: Sendable, Equatable {
        case reference, online, nativeFactory
        var raw: UInt32 { switch self { case .reference: 0; case .online: 1; case .nativeFactory: 2 } }
        var expected: NativeResourcePlan.Attention? {
            switch self { case .reference: .reference; case .online: .online; case .nativeFactory: nil }
        }
    }
    struct Request: Sendable, Equatable {
        let context, upper, chunk, offset: UInt32
        let loadBudget, sessionBudget: UInt64
        var normalization: Normalization = .compensated
        var readback: Readback = .fullLogits
        var attention: AttentionPolicy = .nativeFactory
        func validate() throws {
            let (end,overflow) = offset.addingReportingOverflow(upper)
            guard (1...32768).contains(context), upper > 0,
                  chunk > 0, chunk <= 1024, chunk <= upper, !overflow, end <= context,
                  loadBudget > 0, loadBudget <= 6 * 1024 * 1024 * 1024,
                  sessionBudget <= 6 * 1024 * 1024 * 1024 else { throw NativeCapacityPlan.failure("Invalid capacity request") }
        }
    }
    struct Footprint: Decodable, Sendable {
        let admittedBytes, largestBufferBytes: UInt64
    }
    struct Point: Decodable, Sendable {
        let capacity: UInt32
        let footprint: Footprint
        let fitsBudget, fitsBuffer: Bool
    }
    struct Limits: Decodable, Sendable { let budgetBytes, maximumBufferBytes: UInt64 }
    struct Range: Decodable, Sendable {
        let minimum, maximum: UInt32
        let limits: Limits
        let selected, firstExcluded: Point?
    }
    let resources: NativeResourcePlan
    let capacity: Range
    static let requiredApplicationAllowance = NativeResourcePlan.requiredApplicationAllowance
    // Does not include source values, profile authentication, live allocation,
    // tokenizer/application memory or qualified context. No load authorization.
    var selectedFitsNumericalBudgets: Bool {
        capacity.selected != nil && resources.fitsSourceLoadBudget && resources.fitsResidentLoadBudget
    }
    static func decode(_ data: Data, request: Request) throws -> Self {
        try request.validate()
        guard !data.isEmpty, data.count <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES) else { throw failure("Invalid capacity report length") }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self,from:data)
        _ = try value.resources.validated(context:request.context,capacity:request.upper,
            chunk:request.chunk,offset:request.offset,loadBudget:request.loadBudget,
            sessionBudget:request.sessionBudget,normalization:request.normalization.rawValue,
            readback:request.readback.rawValue,expectedAttention:request.attention.expected)
        let range = value.capacity, resources = value.resources
        guard range.minimum == request.chunk, range.maximum == request.upper,
              range.limits.budgetBytes == request.sessionBudget,
              range.limits.maximumBufferBytes == resources.maximumBufferBytes else { throw failure("Capacity interval or limits differ from request") }
        let (resident,overflow) = resources.residentWeightsBytes.addingReportingOverflow(resources.positionalResidentBytes ?? 0)
        guard !overflow else { throw failure("Resident accounting overflow") }
        func point(_ point: Point) throws {
            let footprint = point.footprint
            guard point.capacity >= range.minimum, point.capacity <= range.maximum,
                  footprint.admittedBytes >= resident,
                  footprint.admittedBytes <= resources.sessionMemory.admittedBytes,
                  footprint.largestBufferBytes > 0, footprint.largestBufferBytes <= footprint.admittedBytes,
                  point.fitsBudget == (footprint.admittedBytes <= range.limits.budgetBytes),
                  point.fitsBuffer == (footprint.largestBufferBytes <= range.limits.maximumBufferBytes && footprint.largestBufferBytes <= UInt64(Int.max)) else { throw failure("Invalid capacity point accounting or fit flags") }
            if point.capacity == range.maximum {
                guard footprint.admittedBytes == resources.sessionMemory.admittedBytes else { throw failure("Upper footprint differs from resource ledger") }
            }
        }
        if let selected = range.selected { try point(selected) }
        if let excluded = range.firstExcluded { try point(excluded) }
        switch (range.selected,range.firstExcluded) {
        case (let selected?,nil):
            guard selected.capacity == range.maximum, selected.fitsBudget, selected.fitsBuffer else { throw failure("Missing adjacent exclusion or nonfitting selection") }
        case (nil,let excluded?):
            guard excluded.capacity == range.minimum, !excluded.fitsBudget || !excluded.fitsBuffer else { throw failure("Invalid empty capacity result") }
        case (let selected?,let excluded?):
            let (next,overflow) = selected.capacity.addingReportingOverflow(1)
            guard !overflow, next == excluded.capacity,
                  selected.fitsBudget, selected.fitsBuffer, !excluded.fitsBudget || !excluded.fitsBuffer,
                  selected.footprint.admittedBytes <= excluded.footprint.admittedBytes,
                  selected.footprint.largestBufferBytes <= excluded.footprint.largestBufferBytes else { throw failure("Invalid adjacent capacity witnesses") }
        case (nil,nil): throw failure("Capacity search has no witnesses")
        }
        return value
    }
    private static func failure(_ message: String, code: UInt32 = TOKLET_INVALID) -> NSError {
        NSError(domain:"TokletCapacityPlan",code:Int(code),userInfo:[NSLocalizedDescriptionKey:message])
    }
    // Synchronous create/query/copy/release on one OS thread. Budgets are exact
    // numerical values supplied by the caller, not whole-app memory allowances.
    static func inspect(directory: String, request: Request) throws -> Self {
        try request.validate()
        let path = Array(directory.utf8)
        guard !path.isEmpty, path.count <= 16384, !path.contains(0) else { throw failure("Invalid artifact path") }
        var options = TokletRuntimePlanningOptions(abi_version:TOKLET_ABI_VERSION,
            struct_size:UInt32(MemoryLayout<TokletRuntimePlanningOptions>.size),context_limit:request.context,
            capacity:request.upper,chunk:request.chunk,position_offset:request.offset,
            normalization:request.normalization == .original ? 0:1,readback:request.readback == .fullLogits ? 0:1,
            attention:request.attention.raw,reserved:0,load_memory_limit_bytes:request.loadBudget,
            session_memory_limit_bytes:request.sessionBudget)
        var owner: OpaquePointer?; var error = TokletError()
        func checked(_ code: UInt32) throws {
            if code != TOKLET_OK {
                let message = withUnsafeBytes(of:&error.message) { String(decoding:$0.prefix { $0 != 0 },as:UTF8.self) }
                throw failure(message.isEmpty ? "Capacity inspection failed" : message,code:code)
            }
        }
        try path.withUnsafeBufferPointer { try checked(toklet_mlx_capacity_plan_create($0.baseAddress,$0.count,&options,&owner,&error)) }
        guard let owner else { throw failure("Missing capacity report") }
        defer { _ = toklet_mlx_inspect_release(owner,nil) }
        var count = 0
        try checked(toklet_mlx_inspect_json(owner,nil,0,&count,&error))
        guard (1...Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES)).contains(count) else { throw failure("Invalid capacity report size") }
        var bytes = [UInt8](repeating:0,count:count)
        try bytes.withUnsafeMutableBufferPointer { try checked(toklet_mlx_inspect_json(owner,$0.baseAddress,$0.count,&count,&error)) }
        guard count == bytes.count else { throw failure("Capacity report size changed") }
        return try decode(Data(bytes),request:request)
    }
}

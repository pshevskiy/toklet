import Foundation
import CToklet

@main
struct RuntimePlanningMain {
    static func main() throws {
        precondition(CommandLine.arguments.count == 2)
        precondition(MemoryLayout<TokletRuntimePlanningOptions>.size == 56)
        precondition(MemoryLayout<TokletRuntimePlanningOptions>.offset(of: \.load_memory_limit_bytes) == 40)
        precondition(MemoryLayout<TokletRuntimePlanningOptions>.offset(of: \.session_memory_limit_bytes) == 48)
        var options = TokletRuntimePlanningOptions()
        options.abi_version = TOKLET_ABI_VERSION
        options.struct_size = UInt32(MemoryLayout<TokletRuntimePlanningOptions>.size)
        options.context_limit = 32; options.capacity = 31; options.chunk = 3
        options.position_offset = 1; options.normalization = 1
        options.load_memory_limit_bytes = 256 * 1024 * 1024
        options.session_memory_limit_bytes = 256 * 1024 * 1024
        var error = TokletError(), handle: OpaquePointer?
        let path = Array(CommandLine.arguments[1].utf8)
        let status = path.withUnsafeBufferPointer {
            toklet_mlx_plan_create($0.baseAddress, $0.count, &options, &handle, &error)
        }
        precondition(status == TOKLET_OK)
        guard let owner = handle else { fatalError("Missing runtime report") }
        defer { precondition(toklet_mlx_inspect_release(owner, nil) == TOKLET_OK) }
        var count = 0
        precondition(toklet_mlx_inspect_json(owner, nil, 0, &count, &error) == TOKLET_OK)
        precondition(count > 0 && count <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES))
        var bytes = [UInt8](repeating: 0xa5, count: count)
        let copied = bytes.withUnsafeMutableBufferPointer {
            toklet_mlx_inspect_json(owner, $0.baseAddress, $0.count, &count, &error)
        }
        precondition(copied == TOKLET_OK && count == bytes.count)
        let value = try JSONSerialization.jsonObject(with: Data(bytes)) as! [String: Any]
        precondition(value["fits_requested_budgets"] as? Bool == true)
        precondition(value["numerical_buffers_allocated"] as? Bool == false)
        precondition(value["qualification_evaluated"] as? Bool == false)
        var capacityOwner: OpaquePointer?
        let capacityStatus = path.withUnsafeBufferPointer {
            toklet_mlx_capacity_plan_create($0.baseAddress, $0.count, &options, &capacityOwner, &error)
        }
        precondition(capacityStatus == TOKLET_OK)
        guard let capacityOwner else { fatalError("Missing capacity report") }
        defer { precondition(toklet_mlx_inspect_release(capacityOwner, nil) == TOKLET_OK) }
        var capacityCount = 0
        precondition(toklet_mlx_inspect_json(capacityOwner, nil, 0, &capacityCount, &error) == TOKLET_OK)
        precondition(capacityCount > 0 && capacityCount <= Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES))
        var capacityBytes = [UInt8](repeating: 0xa5, count: capacityCount)
        let capacityCopied = capacityBytes.withUnsafeMutableBufferPointer {
            toklet_mlx_inspect_json(capacityOwner, $0.baseAddress, $0.count, &capacityCount, &error)
        }
        precondition(capacityCopied == TOKLET_OK && capacityCount == capacityBytes.count)
        let capacityValue = try JSONSerialization.jsonObject(with: Data(capacityBytes)) as! [String: Any]
        let resources = capacityValue["resources"] as! [String: Any]
        precondition(NSDictionary(dictionary: resources).isEqual(to: value))
        let range = capacityValue["capacity"] as! [String: Any]
        precondition((range["minimum"] as? NSNumber)?.uint32Value == 3)
        precondition((range["maximum"] as? NSNumber)?.uint32Value == 31)
        precondition(range["first_excluded"] is NSNull)
        FileHandle.standardOutput.write(Data(bytes)); FileHandle.standardOutput.write(Data([10]))
    }
}

// Standalone public-API toolchain/capability and FP32 dispatch check.
// This is preparation only: it neither imports nor changes the Rust decoder.
import CryptoKit
import Foundation
import Metal
import MetalPerformanceShaders

enum ProbeError: Error { case failure(String) }

func require(_ condition: Bool, _ message: String) throws {
    guard condition else { throw ProbeError.failure(message) }
}

func seconds(_ start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
}

func sha256(_ path: String) throws -> String {
    SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: path)))
        .map { String(format: "%02x", $0) }.joined()
}

@available(macOS 15, *)
func probe() throws -> [String: Any] {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw ProbeError.failure("No Metal device/queue")
    }
    let options = MTLCompileOptions()
    options.languageVersion = .version2_3
    options.mathMode = .safe
    options.mathFloatingPointFunctions = .precise
    let shader = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void checked_add(device const float *x [[buffer(0)]],
                            device const float *y [[buffer(1)]],
                            device float *z [[buffer(2)]],
                            constant uint &count [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
        if (index < count) z[index] = x[index] + y[index];
    }
    """
    let libraryStart = DispatchTime.now().uptimeNanoseconds
    let library = try device.makeLibrary(source: shader, options: options)
    let librarySeconds = seconds(libraryStart)
    guard let function = library.makeFunction(name: "checked_add") else {
        throw ProbeError.failure("Missing compiled function")
    }
    let pipelineStart = DispatchTime.now().uptimeNanoseconds
    let pipeline = try device.makeComputePipelineState(function: function)
    let pipelineSeconds = seconds(pipelineStart)
    let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup,
                    device.maxThreadsPerThreadgroup.width)
    try require(width > 0, "Device/pipeline has no supported threadgroup width")
    var cases = [[String: Any]]()
    for count in [1, 17, 257] {
        let allocated = count + 7
        let bytes = allocated * MemoryLayout<Float>.stride
        try require(bytes <= device.maxBufferLength, "Buffer exceeds device admission")
        guard let x = device.makeBuffer(length: bytes, options: .storageModeShared),
              let y = device.makeBuffer(length: bytes, options: .storageModeShared),
              let z = device.makeBuffer(length: bytes, options: .storageModeShared),
              let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw ProbeError.failure("Failed to allocate buffers or command resources")
        }
        try require(command.retainedReferences, "Commands must retain encoded buffers")
        // Shared allocations are owned by x/y/z, are long enough for every
        // bound Float, and are aligned before binding. Float is a trivial type.
        // All CPU writes precede submission; no pointer is accessed while GPU
        // work is outstanding. Retained command references and lexical owners
        // keep all resources alive through the checked synchronous completion.
        for buffer in [x, y, z] {
            try require(Int(bitPattern: buffer.contents()) % MemoryLayout<Float>.alignment == 0,
                        "Shared allocation alignment differs")
        }
        let xp = x.contents().bindMemory(to: Float.self, capacity: allocated)
        let yp = y.contents().bindMemory(to: Float.self, capacity: allocated)
        let zp = z.contents().bindMemory(to: Float.self, capacity: allocated)
        xp.initialize(repeating: .nan, count: allocated)
        yp.initialize(repeating: .nan, count: allocated)
        zp.initialize(repeating: .nan, count: allocated)
        let sentinel = zp[count].bitPattern
        for index in 0..<count {
            xp[index] = Float(index) / 8
            yp[index] = -Float(index % 17) / 4
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(y, offset: 0, index: 1)
        encoder.setBuffer(z, offset: 0, index: 2)
        var admittedCount = UInt32(count)
        encoder.setBytes(&admittedCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: (count + width - 1) / width, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        command.commit()
        command.waitUntilCompleted()
        let completedSeconds = seconds(start)
        try require(command.status == .completed, "GPU command failed: \(String(describing: command.error))")
        for index in 0..<count {
            try require(zp[index].bitPattern == (xp[index] + yp[index]).bitPattern, "FP32 output differs at \(index)")
        }
        for index in count..<allocated {
            try require(zp[index].bitPattern == sentinel, "Shader wrote outside admitted output")
        }
        let gpuSeconds = command.gpuEndTime - command.gpuStartTime
        cases.append(["elements": count, "logical_buffer_bytes": 3 * bytes,
                      "exact_outputs": true, "padding_untouched": true,
                      "completed_host_seconds": completedSeconds,
                      "gpu_seconds": gpuSeconds > 0 ? gpuSeconds as Any : NSNull()])
        withExtendedLifetime((x, y, z, command, encoder)) {}
    }
    let families: [(String, MTLGPUFamily)] = [
        ("apple1", .apple1), ("apple2", .apple2), ("apple3", .apple3), ("apple4", .apple4),
        ("apple5", .apple5), ("apple6", .apple6), ("apple7", .apple7), ("apple8", .apple8),
        ("apple9", .apple9), ("mac2", .mac2)]
    var supported = Dictionary(uniqueKeysWithValues: families.map { ($0.0, device.supportsFamily($0.1)) })
    if #available(macOS 26, *) { supported["apple10"] = device.supportsFamily(.apple10) }
    let strides = [1, 3, 8, 896, 4864].map { columns in
        ["columns": columns,
         "fp16_row_bytes": MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: .float16),
         "fp32_row_bytes": MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: .float32)]
    }
    return ["format": "toklet-metal-toolchain-probe-v1", "source_sha256": try sha256(#filePath),
            "binary_sha256": try sha256(CommandLine.arguments[0]), "device_name": device.name,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "deployment_requirement": "macOS15; actual Mac execution only",
            "has_unified_memory": device.hasUnifiedMemory,
            "recommended_max_working_set_bytes": device.recommendedMaxWorkingSetSize,
            "max_buffer_length": device.maxBufferLength, "supported_families": supported,
            "pipeline_execution_width": pipeline.threadExecutionWidth,
            "pipeline_max_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup,
            "chosen_threadgroup_width": width, "metal_language": "2.3",
            "math_mode": "safe", "math_functions": "precise",
            "library_compile_seconds": librarySeconds, "pipeline_creation_seconds": pipelineSeconds,
            "mps_supports_device": MPSSupportsMTLDevice(device), "mps_recommended_row_bytes": strides,
            "gpu_trace_capture_destination_supported": MTLCaptureManager.shared().supportsDestination(.gpuTraceDocument),
            "cases": cases,
            "scope": "Functional toolchain/FP32-add only; no decoder, MPS multiplication, FP16, throughput, total memory, GPU trace or device-portability qualification. Single timings are diagnostic observations without warmups/repetitions."]
}

do {
    guard #available(macOS 15, *) else { throw ProbeError.failure("Requires macOS15") }
    let result = try autoreleasepool { try probe() }
    let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    FileHandle.standardError.write(Data("Metal probe failed: \(error)\n".utf8))
    exit(1)
}

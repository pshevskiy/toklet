// Standalone public MPS primitive evaluation, independent of Rust inference.
import CryptoKit
import Foundation
import Metal
import MetalPerformanceShaders

enum ProbeError: Error { case failure(String) }
func require(_ condition: Bool, _ message: @autoclosure () -> String) throws {
    guard condition else { throw ProbeError.failure(message()) }
}
func seconds(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds-start)/1e9 }
func sha256(_ path: String) throws -> String {
    SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: path))).map { String(format: "%02x", $0) }.joined()
}
func emit(_ value: [String: Any]) throws {
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]))
    FileHandle.standardOutput.write(Data("\n".utf8))
}
struct Case: Decodable {
    let name: String
    let kind: String
    let shape: [Int]
    let integer_table: [[Int]]?
    let left: [[Float]]?
    let right: [[Float]]?
    let bias: [Float]?
    let expected: [[Double]]?
    let fp16_output_admissible: Bool
    func validate() throws {
        try require(shape.count == 3, "Matrix shape must have three dimensions")
        try require(shape[0] > 0 && shape[0] <= 128 && shape[1] > 0 && shape[1] <= 4864 && shape[2] > 0 && shape[2] <= 151936,
                    "Fixture shape outside bounded probe scope")
        if kind == "synthetic" {
            try require(integer_table?.count == 13 && integer_table!.allSatisfy { $0.count == 17 }, "Integer oracle shape differs")
        } else {
            try require(kind == "manual" && left?.count == shape[0] && right?.count == shape[2] && bias?.count == shape[2] && expected?.count == shape[0],
                        "Manual oracle shape differs")
            try require(left!.allSatisfy { $0.count == shape[1] } && right!.allSatisfy { $0.count == shape[1] } && expected!.allSatisfy { $0.count == shape[2] },
                        "Manual row shape differs")
            try require((left!.flatMap { $0 }+right!.flatMap { $0 }+bias!).allSatisfy { $0.isFinite && abs($0) <= 65504 }
                        && expected!.flatMap { $0 }.allSatisfy { $0.isFinite && Float($0).isFinite }, "Manual oracle range differs")
        }
    }
    func input(_ t: Int, _ i: Int) -> Float { kind == "synthetic" ? Float((t*7+i)%13-6)/32 : left![t][i] }
    func weight(_ o: Int, _ i: Int) -> Float { kind == "synthetic" ? Float((o+3*i)%17-8)/64 : right![o][i] }
    func biasValue(_ o: Int) -> Float { kind == "synthetic" ? Float(o%7-3)/256 : bias![o] }
    func result(_ t: Int, _ o: Int) -> Double {
        kind == "synthetic" ? Double(integer_table![t%13][o%17])/2048+Double(biasValue(o)) : expected![t][o]
    }
}
struct Fixture: Decodable { let format: String; let cases: [Case] }
struct ProjectionPipelines {
    let bias: MTLComputePipelineState
    let store: MTLComputePipelineState
    let sourceHash: String
    let prepareSeconds: Double
    init(device: MTLDevice, path: String) throws {
        let start = DispatchTime.now().uptimeNanoseconds
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let source = String(data: data, encoding: .utf8) else { throw ProbeError.failure("Expected UTF-8 shader") }
        sourceHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let options = MTLCompileOptions()
        options.languageVersion = .version2_3
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: source, options: options)
        guard let biasFunction = library.makeFunction(name: "projection_bias"),
              let storeFunction = library.makeFunction(name: "projection_store") else { throw ProbeError.failure("Missing projection functions") }
        bias = try device.makeComputePipelineState(function: biasFunction)
        store = try device.makeComputePipelineState(function: storeFunction)
        prepareSeconds = seconds(start)
    }
    func encode(_ command: MTLCommandBuffer, pipeline: MTLComputePipelineState, left: MTLBuffer, right: MTLBuffer,
                tokens: Int, outputs: Int, floatStride: Int, halfStride: Int, device: MTLDevice) throws {
        let (count, overflow) = tokens.multipliedReportingOverflow(by: outputs)
        let (floatElements, floatOverflow) = tokens.multipliedReportingOverflow(by: floatStride)
        let (halfElements, halfOverflow) = tokens.multipliedReportingOverflow(by: halfStride)
        try require(tokens > 0 && outputs > 0 && !overflow && count <= Int(UInt32.max) && floatStride >= outputs && halfStride >= outputs
                    && floatStride <= Int(UInt32.max) && halfStride <= Int(UInt32.max), "Projection dispatch shape overflows")
        try require(!floatOverflow && !halfOverflow && floatElements <= Int(UInt32.max) && halfElements <= Int(UInt32.max),
                    "Projection physical stride indexing overflows")
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup, device.maxThreadsPerThreadgroup.width)
        try require(width > 0, "No qualified projection dispatch width")
        guard let encoder = command.makeComputeCommandEncoder() else { throw ProbeError.failure("Projection encoder allocation failed") }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(left, offset: 0, index: 0)
        encoder.setBuffer(right, offset: 0, index: 1)
        var shape = SIMD4<UInt32>(UInt32(tokens), UInt32(outputs), UInt32(floatStride), UInt32(halfStride))
        encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: (count+width-1)/width, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
struct Matrix {
    let value: MPSMatrix
    let buffer: MTLBuffer
    let half: Bool
    let elements: Int
    let payloadBytes: Int
    let guardData: Data
    var rowBytes: Int { value.rowBytes }
}

func allocation(rows: Int, columns: Int, half: Bool) throws -> (rowBytes: Int, payload: Int, bytes: Int) {
    try require(rows > 0 && columns > 0 && rows <= 151936 && columns <= 151936, "Invalid allocation dimensions")
    let size = half ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
    let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: half ? .float16 : .float32)
    let (payload, overflow) = rows.multipliedReportingOverflow(by: rowBytes)
    let (bytes, extraOverflow) = payload.addingReportingOverflow(7*size)
    try require(!overflow && !extraOverflow && rowBytes >= columns*size && rowBytes%size == 0, "Matrix allocation overflows")
    return (rowBytes, payload, bytes)
}

// Typed binding covers only aligned, sized, owned shared allocations. These
// trivial values are initialized once. All pointer access is before submission
// or after checked completion. Matrix/command lexical owners retain resources.
func populate<T: BinaryFloatingPoint>(_ pointer: UnsafeMutablePointer<T>, rows: Int, columns: Int, stride: Int,
                                      value: (Int, Int) -> Float) {
    for row in 0..<rows { for column in 0..<columns { pointer[row*stride+column] = T(value(row, column)) } }
}
func matrix(_ device: MTLDevice, rows: Int, columns: Int, half: Bool, value: (Int, Int) -> Float) throws -> Matrix {
    let type: MPSDataType = half ? .float16 : .float32
    let size = half ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
    let alignment = half ? MemoryLayout<Float16>.alignment : MemoryLayout<Float>.alignment
    let layout = try allocation(rows: rows, columns: columns, half: half)
    let rowBytes = layout.rowBytes, payload = layout.payload, bytes = layout.bytes
    try require(bytes <= device.maxBufferLength && bytes <= 6*1024*1024*1024, "Matrix exceeds memory admission")
    guard let buffer = device.makeBuffer(length: bytes, options: [.storageModeShared, .hazardTrackingModeTracked]) else { throw ProbeError.failure("Shared allocation failed") }
    try require(buffer.hazardTrackingMode == .tracked, "Projection buffers must track encoder dependencies")
    try require(Int(bitPattern: buffer.contents())%alignment == 0, "Shared alignment differs")
    let elements = bytes/size
    if half {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: elements)
        pointer.initialize(repeating: Float16(bitPattern: 0x7e57), count: elements)
        populate(pointer, rows: rows, columns: columns, stride: rowBytes/size, value: value)
    } else {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: elements)
        pointer.initialize(repeating: Float(bitPattern: 0x7fc05678), count: elements)
        populate(pointer, rows: rows, columns: columns, stride: rowBytes/size, value: value)
    }
    let descriptor = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: rowBytes, dataType: type)
    return Matrix(value: MPSMatrix(buffer: buffer, descriptor: descriptor), buffer: buffer, half: half, elements: elements,
                  payloadBytes: payload, guardData: Data(bytes: buffer.contents().advanced(by: payload), count: 7*size))
}
func payloadHash(_ matrix: Matrix) -> String {
    var hash = SHA256()
    for offset in stride(from: 0, to: matrix.payloadBytes, by: 65536) {
        hash.update(data: Data(bytes: matrix.buffer.contents().advanced(by: offset), count: min(65536, matrix.payloadBytes-offset)))
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
func guardUnchanged(_ matrix: Matrix) -> Bool {
    Data(bytes: matrix.buffer.contents().advanced(by: matrix.payloadBytes), count: matrix.guardData.count) == matrix.guardData
}
func halfPaddingUnchanged(_ matrix: Matrix) -> Bool {
    let pointer = matrix.buffer.contents().assumingMemoryBound(to: Float16.self)
    for row in 0..<matrix.value.rows {
        for column in matrix.value.columns..<matrix.rowBytes/2 {
            if pointer[row*matrix.rowBytes/2+column].bitPattern != 0x7e57 { return false }
        }
    }
    return true
}

func json(_ path: String) throws -> [String: Any] {
    guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any] else {
        throw ProbeError.failure("Expected object manifest")
    }
    return value
}
func array(_ directory: String, _ manifest: [String: Any], _ name: String) throws -> [Float] {
    guard let arrays = manifest["arrays"] as? [String: [String: Any]], let item = arrays[name],
          let file = item["file"] as? String, let dtype = item["dtype"] as? String,
          let shape = item["shape"] as? [Int], let expectedBytes = item["bytes"] as? Int, let hash = item["sha256"] as? String else {
        throw ProbeError.failure("Missing typed array metadata: \(name)")
    }
    try require(file == URL(fileURLWithPath: file).lastPathComponent && ["f16le", "f32le"].contains(dtype), "Unexpected array path/dtype")
    var count = 1
    for dimension in shape {
        let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
        try require(dimension > 0 && !overflow && next <= 65536, "Array shape outside tiny probe bounds")
        count = next
    }
    let size = dtype == "f16le" ? 2 : 4
    try require(expectedBytes == count*size, "Tiny byte metadata differs from shape")
    let path = URL(fileURLWithPath: directory).appendingPathComponent(file).path
    let fileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? fileHandle.close() }
    let data = try fileHandle.read(upToCount: expectedBytes+1) ?? Data()
    let loadedHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    try require(data.count == expectedBytes && data.count == count*size && loadedHash == hash, "Tiny payload identity differs")
    // Data guarantees byte alignment only; assemble little-endian bits without
    // making an aligned UInt16/UInt32 load or borrowing its backing allocation.
    var values = [Float](); values.reserveCapacity(count)
    for offset in stride(from: 0, to: data.count, by: size) {
        let low = UInt32(data[offset]) | UInt32(data[offset+1])<<8
        let value = size == 2 ? Float(Float16(bitPattern: UInt16(low)))
                             : Float(bitPattern: low | UInt32(data[offset+2])<<16 | UInt32(data[offset+3])<<24)
        try require(value.isFinite, "Nonfinite tiny payload")
        values.append(value)
    }
    return values
}
func tinyProjections(_ halfDirectory: String, _ sourceDirectory: String, mode: String,
                     device: MTLDevice, queue: MTLCommandQueue, pipelines: ProjectionPipelines?) throws -> [[String: Any]] {
    let halfPath = URL(fileURLWithPath: halfDirectory).appendingPathComponent("manifest.json").path
    let sourcePath = URL(fileURLWithPath: sourceDirectory).appendingPathComponent("manifest.json").path
    let half = try json(halfPath), source = try json(sourcePath)
    try require(half["format"] as? String == "toklet-tiny-fp16-v1" && half["source_manifest_sha256"] as? String == sha256(sourcePath), "Tiny reference identity differs")
    guard let cases = half["cases"] as? [String: [String: Any]], let arrays = half["arrays"] as? [String: [String: Any]] else {
        throw ProbeError.failure("Missing tiny inventory")
    }
    let weights = try array(halfDirectory, half, "weights.layers.0.q_proj.weight")
    let bias = try array(halfDirectory, half, "weights.layers.0.q_proj.bias")
    try require(weights.count == 32*32 && bias.count == 32, "Tiny query shape differs")
    var results = [[String: Any]]()
    for name in cases.keys.sorted() {
        let prefix = "cases.\(name).layers.0"
        let x = try array(halfDirectory, half, "\(prefix).input_norm")
        let expected = try array(sourceDirectory, source, "\(prefix).q_projection")
        guard let tokens = cases[name]!["tokens"] as? [Int], let gate = arrays["\(prefix).q_projection"]?["gate_against_source_fp32"] as? [String: Double],
              let atol = gate["atol"], let rtol = gate["rtol"] else { throw ProbeError.failure("Missing frozen tiny gate") }
        try require(!tokens.isEmpty && tokens.count <= 32 && x.count == tokens.count*32 && expected.count == x.count && atol.isFinite && atol >= 0 && rtol == 1e-4,
                    "Tiny projection shape/gate differs")
        let a = try matrix(device, rows: tokens.count, columns: 32, half: mode != "f32", value: { row,col in x[row*32+col] })
        let b = try matrix(device, rows: 32, columns: 32, half: mode != "f32", value: { row,col in weights[row*32+col] })
        let c = try matrix(device, rows: tokens.count, columns: 32, half: mode == "f16", value: { _,col in bias[col] })
        let stored = try pipelines.map { _ in try matrix(device, rows: tokens.count, columns: 32, half: true, value: { _,_ in .nan }) }
        let biasMatrix = try pipelines.map { _ in try matrix(device, rows: 1, columns: 32, half: true, value: { _,col in bias[col] }) }
        let biasHash = biasMatrix.map { payloadHash($0) }
        let aHash = payloadHash(a), bHash = payloadHash(b)
        let kernel = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                             resultRows: tokens.count, resultColumns: 32, interiorColumns: 32, alpha: 1, beta: 1)
        guard let command = queue.makeCommandBuffer() else { throw ProbeError.failure("Tiny command allocation failed") }
        try require(command.retainedReferences, "Tiny command must retain resources")
        if let pipelines, let stored, let biasMatrix {
            try pipelines.encode(command, pipeline: pipelines.bias, left: biasMatrix.buffer, right: c.buffer,
                                 tokens: tokens.count, outputs: 32, floatStride: c.rowBytes/4, halfStride: stored.rowBytes/2, device: device)
        }
        kernel.encode(commandBuffer: command, leftMatrix: a.value, rightMatrix: b.value, resultMatrix: c.value)
        if let pipelines, let stored {
            try pipelines.encode(command, pipeline: pipelines.store, left: c.buffer, right: stored.buffer,
                                 tokens: tokens.count, outputs: 32, floatStride: c.rowBytes/4, halfStride: stored.rowBytes/2, device: device)
        }
        command.commit(); command.waitUntilCompleted()
        try require(command.status == .completed, "Tiny GPU command failed")
        var maxError = 0.0, maxFraction = 0.0
        for row in 0..<tokens.count { for col in 0..<32 {
            let value = c.half ? Float(c.buffer.contents().assumingMemoryBound(to: Float16.self)[row*c.rowBytes/2+col])
                               : c.buffer.contents().assumingMemoryBound(to: Float.self)[row*c.rowBytes/4+col]
            // The resident mode reads GPU-stored Half. Other FP32-result modes
            // emulate the storage cast on CPU outside operation timing.
            let halfValue = stored.map { Float($0.buffer.contents().assumingMemoryBound(to: Float16.self)[row*$0.rowBytes/2+col]) } ?? Float(Float16(value))
            try require(value.isFinite && halfValue.isFinite, "Nonfinite tiny projection")
            try require(Float16(halfValue).bitPattern == Float16(value).bitPattern, "Tiny GPU storage cast differs")
            let reference = Double(expected[row*32+col])
            let error = abs(Double(halfValue)-reference), fraction = error/(atol+rtol*abs(reference))
            maxError = max(maxError,error); maxFraction = max(maxFraction,fraction)
        } }
        try require(guardUnchanged(a) && guardUnchanged(b) && guardUnchanged(c), "Tiny physical guard changed")
        try require(payloadHash(a) == aHash && payloadHash(b) == bHash, "Tiny immutable projection input changed")
        if let stored, let biasMatrix {
            try require(guardUnchanged(stored) && halfPaddingUnchanged(stored) && guardUnchanged(biasMatrix) && payloadHash(biasMatrix) == biasHash, "Tiny resident bias/store changed guards, padding or immutable bias")
        }
        let result: [String: Any] = ["case": name, "maximum_absolute_error": maxError, "maximum_tolerance_fraction": maxFraction,
                        "atol": atol, "rtol": rtol, "guards_untouched": true,
                        "scope": "First-layer biased Q projection only, from frozen FP16 normalized input; result stored as FP16 before original FP32 gate comparison. Single functional check, no timing claim."]
        try emit(["event": "tiny_projection_checked", "result": result])
        try require(maxFraction <= 1, "Tiny frozen gate failed at \(name): \(maxFraction)")
        results.append(result)
        withExtendedLifetime((a,b,c,stored,biasMatrix,pipelines,kernel,command)) {}
    }
    return results
}

func runCase(_ item: Case, mode: String, device: MTLDevice, queue: MTLCommandQueue, pipelines: ProjectionPipelines?) throws -> [String: Any] {
    try item.validate()
    let tokens = item.shape[0], inputs = item.shape[1], outputs = item.shape[2]
    let halfInput = mode != "f32", halfResult = mode == "f16"
    if (halfResult || pipelines != nil) && !item.fp16_output_admissible {
        return ["name": item.name, "skipped": "exact stored result exceeds finite FP16 range; FP32-only result modes test it"]
    }
    var layouts = [try allocation(rows: tokens, columns: inputs, half: halfInput),
                   try allocation(rows: outputs, columns: inputs, half: halfInput),
                   try allocation(rows: tokens, columns: outputs, half: halfResult)]
    if pipelines != nil {
        layouts.append(try allocation(rows: tokens, columns: outputs, half: true))
        layouts.append(try allocation(rows: 1, columns: outputs, half: true))
    }
    var admitted = 0
    for layout in layouts {
        let (sum, overflow) = admitted.addingReportingOverflow(layout.bytes)
        try require(!overflow && sum <= 6*1024*1024*1024 && layout.bytes <= device.maxBufferLength, "Combined allocation exceeds budget/device")
        admitted = sum
    }
    let a = try matrix(device, rows: tokens, columns: inputs, half: halfInput, value: item.input)
    let b = try matrix(device, rows: outputs, columns: inputs, half: halfInput, value: item.weight)
    let c = try matrix(device, rows: tokens, columns: outputs, half: halfResult, value: { _, column in item.biasValue(column) })
    let stored = try pipelines.map { _ in try matrix(device, rows: tokens, columns: outputs, half: true, value: { _,_ in .nan }) }
    let biasMatrix = try pipelines.map { _ in try matrix(device, rows: 1, columns: outputs, half: true, value: { _,column in item.biasValue(column) }) }
    let logicalBytes = a.buffer.length+b.buffer.length+c.buffer.length+(stored?.buffer.length ?? 0)+(biasMatrix?.buffer.length ?? 0)
    try require(logicalBytes == admitted, "Allocated lengths differ from admission")
    let aHash = payloadHash(a), bHash = payloadHash(b)
    let biasHash = biasMatrix.map { payloadHash($0) }
    let start = DispatchTime.now().uptimeNanoseconds
    // C starts as the bias in every row. beta=1 requests bias addition within
    // multiplication; the independent rounding edge qualifies its behavior.
    let kernel = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                         resultRows: tokens, resultColumns: outputs, interiorColumns: inputs, alpha: 1, beta: 1)
    let prepareSeconds = seconds(start)
    var samples = [[String: Any]]()
    for repetition in 0..<13 {
        let resetStart = DispatchTime.now().uptimeNanoseconds
        if pipelines != nil {
            // GPU bias initialization is encoded with projection and storage.
        } else if halfResult {
            populate(c.buffer.contents().assumingMemoryBound(to: Float16.self), rows: tokens, columns: outputs,
                     stride: c.rowBytes/2, value: { _, column in item.biasValue(column) })
        } else {
            populate(c.buffer.contents().assumingMemoryBound(to: Float.self), rows: tokens, columns: outputs,
                     stride: c.rowBytes/4, value: { _, column in item.biasValue(column) })
        }
        let resetSeconds = pipelines == nil ? seconds(resetStart) : 0
        let timedStart = DispatchTime.now().uptimeNanoseconds
        guard let command = queue.makeCommandBuffer() else { throw ProbeError.failure("Command allocation failed") }
        try require(command.retainedReferences, "Command must retain matrix resources")
        if let pipelines, let stored, let biasMatrix {
            try pipelines.encode(command, pipeline: pipelines.bias, left: biasMatrix.buffer, right: c.buffer,
                                 tokens: tokens, outputs: outputs, floatStride: c.rowBytes/4, halfStride: stored.rowBytes/2, device: device)
        }
        kernel.encode(commandBuffer: command, leftMatrix: a.value, rightMatrix: b.value, resultMatrix: c.value)
        if let pipelines, let stored {
            try pipelines.encode(command, pipeline: pipelines.store, left: c.buffer, right: stored.buffer,
                                 tokens: tokens, outputs: outputs, floatStride: c.rowBytes/4, halfStride: stored.rowBytes/2, device: device)
        }
        command.commit()
        command.waitUntilCompleted()
        let completedSeconds = seconds(timedStart)
        try require(command.status == .completed, "Command failed: \(String(describing: command.error))")
        for row in 0..<tokens { for column in 0..<outputs {
            let expected = item.result(row, column)
            if halfResult {
                let actual = c.buffer.contents().assumingMemoryBound(to: Float16.self)[row*c.rowBytes/2+column]
                if actual.bitPattern != Float16(expected).bitPattern {
                    try emit(["event": "numerical_failure", "case": item.name, "row": row, "column": column,
                              "iteration": repetition, "phase": repetition < 3 ? "warmup" : "measurement",
                              "actual": Double(actual), "expected": Double(Float16(expected)),
                              "actual_bits": actual.bitPattern, "expected_bits": Float16(expected).bitPattern])
                }
                try require(actual.bitPattern == Float16(expected).bitPattern, "Exact FP16 oracle differs at \(item.name)[\(row),\(column)]: \(actual) vs \(Float16(expected))")
            } else {
                let actual = c.buffer.contents().assumingMemoryBound(to: Float.self)[row*c.rowBytes/4+column]
                if actual.bitPattern != Float(expected).bitPattern {
                    try emit(["event": "numerical_failure", "case": item.name, "row": row, "column": column,
                              "iteration": repetition, "phase": repetition < 3 ? "warmup" : "measurement",
                              "actual": Double(actual), "expected": Double(Float(expected)),
                              "actual_bits": actual.bitPattern, "expected_bits": Float(expected).bitPattern])
                }
                try require(actual.bitPattern == Float(expected).bitPattern, "Exact FP32 oracle differs at \(item.name)[\(row),\(column)]: \(actual) vs \(Float(expected))")
            }
            if let stored {
                let actual = stored.buffer.contents().assumingMemoryBound(to: Float16.self)[row*stored.rowBytes/2+column]
                try require(actual.bitPattern == Float16(expected).bitPattern, "Exact resident FP16 output differs at \(item.name)[\(row),\(column)]")
            }
        } }
        try require(guardUnchanged(a) && guardUnchanged(b) && guardUnchanged(c), "MPS wrote outside physical matrix payload")
        if let stored, let biasMatrix {
            try require(guardUnchanged(stored) && halfPaddingUnchanged(stored) && guardUnchanged(biasMatrix), "Resident projection wrote outside physical payload or into stored row padding")
        }
        let gpuSeconds = command.gpuEndTime-command.gpuStartTime
        if repetition >= 3 {
            let sample: [String: Any] = ["sample_index": repetition-3, "completed_host_seconds": completedSeconds,
                                         "gpu_seconds": gpuSeconds > 0 ? gpuSeconds as Any : NSNull(), "excluded_cpu_bias_reset_seconds": resetSeconds]
            samples.append(sample)
            try emit(["event": "sample", "case": item.name, "sample": sample])
        }
        withExtendedLifetime((a, b, c, stored, biasMatrix, pipelines, command, kernel)) {}
    }
    try require(payloadHash(a) == aHash && payloadHash(b) == bHash, "Immutable matrix input changed")
    if let biasMatrix { try require(payloadHash(biasMatrix) == biasHash, "Immutable resident bias changed") }
    return ["name": item.name, "shape": item.shape, "logical_buffer_bytes": logicalBytes,
            "left_row_bytes": a.rowBytes, "right_row_bytes": b.rowBytes, "result_row_bytes": c.rowBytes,
            "left_payload_sha256": aHash, "right_payload_sha256": bHash, "result_payload_sha256": payloadHash(c),
            "kernel_object_prepare_seconds": prepareSeconds, "exact_outputs": true, "guards_untouched": true, "samples": samples,
            "resident_fp16_storage": pipelines != nil, "stored_row_bytes": stored?.rowBytes ?? 0,
            "stored_payload_sha256": stored.map { payloadHash($0) } ?? "", "stored_row_padding_untouched": stored.map { halfPaddingUnchanged($0) } ?? true,
            "cpu_bias_reset_required": pipelines == nil]
}

do {
    let args = CommandLine.arguments
    try require(args.count == 6 && ["f32", "f16", "f16-to-f32", "f16-stored"].contains(args[1]), "Usage: probe mode fixture.json fp16-fixture-dir source-fixture-dir shader-path")
    let mode = args[1], path = args[2]
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    try require(fixture.format == "toklet-mps-matrix-v1", "Fixture version differs")
    guard let device = MTLCreateSystemDefaultDevice(), MPSSupportsMTLDevice(device), let queue = device.makeCommandQueue() else {
        throw ProbeError.failure("No supported Metal/MPS device")
    }
    let shaderHash = try sha256(args[5])
    let pipelines = try mode == "f16-stored" ? ProjectionPipelines(device: device, path: args[5]) : nil
    if let pipelines { try require(pipelines.sourceHash == shaderHash, "Loaded projection shader identity differs") }
    var cases = [[String: Any]]()
    for item in fixture.cases {
        try emit(["event": "case_started", "name": item.name])
        let result = try autoreleasepool { try runCase(item, mode: mode, device: device, queue: queue, pipelines: pipelines) }
        cases.append(result)
        try emit(["event": "case_completed", "case": result])
    }
    // Independent manual rounding edge already checked in the committed
    // Python reference test before this primitive evaluation: rounding the dot
    // to Half before adding bias would produce 1 instead of 1+2^-10.
    let biasEdge = Case(name: "single_cast_bias_edge", kind: "manual", shape: [1,2,1], integer_table: nil,
                        left: [[1,1]], right: [[1,1.0/2048]], bias: [1.0/2048], expected: [[1.0+1.0/1024]], fp16_output_admissible: true)
    try emit(["event": "case_started", "name": biasEdge.name])
    let edge = try autoreleasepool { try runCase(biasEdge, mode: mode, device: device, queue: queue, pipelines: pipelines) }
    cases.append(edge)
    try emit(["event": "case_completed", "case": edge])
    let tiny = try tinyProjections(args[3], args[4], mode: mode, device: device, queue: queue, pipelines: pipelines)
    try emit(["event": "complete", "format": "toklet-mps-primitive-probe-v1", "mode": mode, "device": device.name,
              "source_sha256": try sha256(#filePath), "binary_sha256": try sha256(args[0]), "fixture_sha256": try sha256(path),
              "os": ProcessInfo.processInfo.operatingSystemVersionString, "has_unified_memory": device.hasUnifiedMemory,
              "shader_sha256": shaderHash, "projection_pipeline_prepare_seconds": pipelines?.prepareSeconds ?? 0,
              "timing_scope": pipelines == nil ? "MPS encode, command submission and successful completed work; CPU bias initialization, checks, allocation, hashes and kernel object construction excluded" : "GPU bias initialization, MPS projection and FP16 storage encoded into one command with a single completion wait; allocation, pipeline preparation, hashes, checks and transport excluded",
              "bias_policy": pipelines == nil ? "C initialized from bias before each request, beta=1; a resident engine would need GPU bias initialization or another qualified path" : "Resident FP16 bias initializes FP32 C on GPU, beta=1 projection, then GPU final FP16 cast; no CPU bias restaging",
              "accumulation": "Not inferred from storage dtype; exact dyadic/manual and range probes only",
              "warmups": 3, "repetitions": 10, "cases": cases, "tiny_projection_checks": tiny,
              "tiny_fp16_manifest_sha256": try sha256(URL(fileURLWithPath: args[3]).appendingPathComponent("manifest.json").path),
              "tiny_source_manifest_sha256": try sha256(URL(fileURLWithPath: args[4]).appendingPathComponent("manifest.json").path)])
    try require(readLine() == "quit", "Expected release acknowledgement after idle footprint inspection")
} catch {
    FileHandle.standardError.write(Data("MPS probe failed: \(error)\n".utf8))
    // Swift failures unwind after completion (or before submission); preserve
    // an idle inspection boundary even for a rejected numerical result.
    try? emit(["event": "failure", "message": String(describing: error)])
    _ = readLine()
    exit(1)
}

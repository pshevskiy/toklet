// Mac command-line transport for the same sealed jobs used by the iOS app.
// Results retain the runners' false qualification flags; the host admits evidence.
import Foundation
import Darwin

private func emitReal<T: Encodable>(_ marker: String, _ value: T) throws {
    let data = try JSONEncoder().encode(value)
    try FileHandle.standardOutput.write(contentsOf:Data((marker+String(decoding:data,as:UTF8.self)+"\n").utf8))
}
private func realRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw NSError(domain:"TokletMLXRealJob",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
@main private struct MLXRealMain {
    static func main() {
        DispatchQueue.global(qos:.userInitiated).async {
            do { exit(try run() ? 0:1) }
            catch {
                try? FileHandle.standardError.write(contentsOf:Data("TOKLET_MAC_JOB_ERROR=\(error)\n".utf8))
                exit(1)
            }
        }
        dispatchMain()
    }
    private static func run() throws -> Bool {
        let arguments = CommandLine.arguments
        try realRequire(arguments.count == 5,"Expected mode, root, relative job and exact SHA")
        let root = URL(fileURLWithPath:arguments[2],isDirectory:true)
        let name = arguments[3];let sha = arguments[4]
        switch arguments[1] {
        case "measurement":
            let config = try admitMeasurementConfig(root:root,name:name,sha256:sha)
            try realRequire(config.version == 2 && config.mlx == true,"Explicit version-two MLX job required")
            let result = runNativeMeasurement(config,root:root) { sample in
                try emitReal("TOKLET_MEASUREMENT_SAMPLE_JSON=",sample)
            }
            try emitReal("TOKLET_MEASUREMENT_JSON=",result)
            return result.passed
        case "quality":
            let config = try admitQualityConfig(root:root,name:name,sha256:sha)
            try realRequire(config.version == 2 && config.mlx == true,"Explicit version-two MLX job required")
            let result = runNativeQuality(config,root:root) { sample in
                try emitReal("TOKLET_QUALITY_SAMPLE_JSON=",sample)
            }
            try emitReal("TOKLET_QUALITY_JSON=",result)
            return result.passed
        case "tasks":
            let config = try admitTasksConfig(root:root,name:name,sha256:sha)
            try realRequire(config.version == 2 && config.mlx == true,"Explicit version-two MLX job required")
            let result = runNativeTasks(config,root:root,onChunk:{ sample in
                try emitReal("TOKLET_TASK_CHUNK_JSON=",sample)
            },onCase:{ sample in
                try emitReal("TOKLET_TASK_CASE_JSON=",sample)
            })
            try emitReal("TOKLET_TASKS_JSON=",result)
            return result.passed
        case "state-component":
            let config = try admitLifecycleConfig(root:root,name:name,sha256:sha)
            try realRequire(config.version == 2 && config.mlx == true,"Explicit version-two MLX job required")
            // This Mac component route makes no OS-event or physical prefill-window claim.
            let result = runNativeState(config,root:root,signals:NativeLifecycleSignals(),requirePrefillWitness:false)
            try emitReal("TOKLET_STATE_JSON=",result)
            return result.passed
        default:
            throw NSError(domain:"TokletMLXRealJob",code:2,userInfo:[NSLocalizedDescriptionKey:"Unsupported diagnostic mode"])
        }
    }
}

import Foundation
@main struct StateChecks {
    static func main() async throws {
        try await Task.detached { try run() }.value
    }
    private static func run() throws {
        let args = ProcessInfo.processInfo.arguments
        guard args.count == 5, ["unit","small"].contains(args[4]) else { throw NSError(domain:"TokletStateChecks",code:1) }
        let root = URL(fileURLWithPath:args[1],isDirectory:true)
        let config = try admitLifecycleConfig(root:root,name:args[2],sha256:args[3])
        let result = runNativeState(config,root:root,signals:NativeLifecycleSignals(),requirePrefillWitness:args[4] == "small")
        let bytes = try JSONEncoder().encode(result)
        try FileHandle.standardOutput.write(contentsOf:bytes+Data("\n".utf8))
        if !result.passed { exit(1) }
    }
}

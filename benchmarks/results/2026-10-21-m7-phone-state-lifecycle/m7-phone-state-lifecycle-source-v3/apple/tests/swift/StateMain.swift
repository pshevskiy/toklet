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
        let failedWorker = try NativeWorker(captureSavedPrefix:true)
        guard failedWorker.submit(.score(target:0)), let failed = failedWorker.waitUntilIdle(seconds:120), failed.status == 1 else {
            throw NSError(domain:"TokletStateChecks",code:3)
        }
        failedWorker.close();let closed = failedWorker.waitUntilClosed(seconds:120);let released = failedWorker.snapshot()
        let errorWitness = NativeLifecycleState(released)
        guard closed, released.closed, released.workerFinished, released.status == 0, released.message.isEmpty,
              released.admittedBytes == 0, released.ids.isEmpty, released.tokenTimings.isEmpty,
              released.history == 0, released.evaluated == 0, released.pending == 0, released.savedHistory == 0,
              released.savedPrefix == nil, released.reason == 0, !released.hasModel, !released.hasSession else {
            throw NSError(domain:"TokletStateChecks",code:4)
        }
        struct Checked: Encodable {
            let diagnostic: NativeStateResult
            let errorBeforeClose: NativeLifecycleState
            let errorAfterClose: NativeLifecycleState
            let errorCloseAdmission: UInt64
            let errorCloseCallbacks: Int
            let errorCloseMessage: String
        }
        let result = runNativeState(config,root:root,signals:NativeLifecycleSignals(),requirePrefillWitness:args[4] == "small")
        let bytes = try JSONEncoder().encode(Checked(diagnostic:result,errorBeforeClose:NativeLifecycleState(failed),errorAfterClose:errorWitness,errorCloseAdmission:released.admittedBytes,errorCloseCallbacks:released.tokenTimings.count,errorCloseMessage:released.message))
        try FileHandle.standardOutput.write(contentsOf:bytes+Data("\n".utf8))
        if !result.passed { exit(1) }
    }
}

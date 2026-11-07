import Foundation
import CryptoKit
import CToklet

private final class UIControlFaults: @unchecked Sendable {
    private let lock = NSLock()
    private var decodeFault = false, block = false
    let entered = DispatchSemaphore(value:0), release = DispatchSemaphore(value:0)
    func armDecode() { lock.lock(); decodeFault = true; lock.unlock() }
    func armBlock() { lock.lock(); block = true; lock.unlock() }
    func waitUntilEntered() async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos:.userInitiated).async {
                continuation.resume(returning:self.entered.wait(timeout:.now()+30) == .success)
            }
        }
    }
    func decode() throws {
        lock.lock(); let fail = decodeFault; decodeFault = false; lock.unlock()
        if fail { throw NativeContextFailure("Injected UI output-decoding failure") }
    }
    func dequeue(_ command: NativeCommand) {
        guard case .chat = command else { return }
        lock.lock(); let wait = block; block = false; lock.unlock()
        if wait { entered.signal(); _ = release.wait(timeout:.now()+30) }
    }
}

// Real controller -> worker -> main-actor publication. A 100ms controller poll
// models the Mac timer; this executable does not create a window or claim paint.
@main
@MainActor
struct UIPublicationLoadedMain {
    static let prompt = "Name the capital of France."
    static let expected: [UInt32] = [818,5279,529,7001,563,5213,50429,84750]
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func main() async throws {
        guard CommandLine.arguments.count == 4, ["paired","controls"].contains(CommandLine.arguments[3]) else {
            throw NativeContextFailure("Expected model, fresh output directory and paired/controls")
        }
        let output = URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let controls = CommandLine.arguments[3] == "controls", faults = UIControlFaults()
        let harness = HarnessStore(enabled:true)
        if let initial = harness.worker {
            initial.close(); try require(initial.waitUntilClosed(seconds:30),"Initial close failed")
        }
        harness.worker = nil
        let dequeue: @Sendable (NativeCommand) -> Void = { faults.dequeue($0) }
        let decode: @Sendable () throws -> Void = { try faults.decode() }
        let worker = try NativeWorker(requestSummaryMode:.summary,
            beforeCommandDequeue:controls ? dequeue:nil,beforeChatDecode:controls ? decode:nil)
        harness.worker = worker; harness.state = worker.snapshot()
        let chat = FriendlyChatStore(harness:harness,fileURL:output.appendingPathComponent("chat.json"),
            preferences:FriendlyPreferenceStore(directory:output.appendingPathComponent("preferences")))
        defer {
            if let data = try? worker.exportRequestSummaries() {
                try? data.write(to:output.appendingPathComponent("terminal-summaries.jsonl"),options:.withoutOverwriting)
            }
            worker.close(); _ = worker.waitUntilClosed(seconds:30)
        }
        func poll(cancelAfterFirst: Bool = false, requireClosed: Bool = false) async throws -> NativeSnapshot {
            let start = DispatchTime.now().uptimeNanoseconds
            var cancelled = false
            repeat {
                harness.poll(); chat.observe(harness.state)
                let value = harness.state
                if !value.busy && !chat.loadingModel && !chat.running && (!requireClosed || value.closed) { return value }
                if cancelAfterFirst && !cancelled, let submitted = value.requestStartedUptimeNanoseconds,
                   value.tokenTimings.contains(where:{ $0.callbackUptimeNanoseconds >= submitted }) {
                    chat.sendOrStop(); cancelled = true
                }
                try await Task.sleep(for:cancelAfterFirst ? .milliseconds(5):.milliseconds(100))
            } while DispatchTime.now().uptimeNanoseconds-start < 180_000_000_000
            throw NativeContextFailure("UI controller timeout")
        }
        chat.openManaged(URL(fileURLWithPath:CommandLine.arguments[1]),profile:.gemmaBF16,ifSelected:chat.workspace.selected)
        let loaded = try await poll()
        try require(loaded.status == 0 && loaded.acceptedIdentity?.profile == .gemmaBF16 && chat.ready,loaded.message + " " + chat.notice)
        chat.setReplyLimit(32)
        try require(worker.requestSummaryUISubmissionTime() != nil,"UI collector not admitted")
        func summaryRows() throws -> [[String:Any]] {
            let data: Data = try worker.exportRequestSummaries()
            return try data.split(separator:10).map { line in
                try JSONSerialization.jsonObject(with:Data(line)) as! [String:Any]
            }
        }
        func latestUI(_ expectedOutcome: String, nonempty: Bool) throws {
            let rows = try summaryRows()
            guard let row = rows.last, let ui = row["ui"] as? [String:Any],
                  let publication = ui["terminalHandlingMilliseconds"] as? Double,
                  let total = row["requestMilliseconds"] as? Double else {
                throw NativeContextFailure("UI summary missing")
            }
            try require(ui["outcome"] as? String == expectedOutcome && publication >= total,
                        "UI terminal outcome or nesting differs")
            try require((ui["firstNonemptyPublicationMilliseconds"] != nil) == nonempty &&
                        (ui["firstAnswerPublicationMilliseconds"] != nil) == nonempty,"UI text availability differs")
        }
        func prepare(_ text: String = prompt) throws {
            chat.removeAll(); chat.edit(text)
            try require(chat.canSend,"Send unavailable: \(chat.notice)")
        }
        if !controls {
            var samples = [[String:Any]]()
            for trial in 0..<13 {
                for enabled in trial % 2 == 0 ? [false,true]:[true,false] {
                    try require(worker.setRequestSummariesEnabled(enabled),"UI mode switch failed")
                    try prepare()
                    let start = DispatchTime.now().uptimeNanoseconds
                    chat.sendOrStop()
                    let value = try await poll()
                    let end = DispatchTime.now().uptimeNanoseconds
                    let footprint = try NativeFootprint.read()
                    try require(footprint.lifetimePeakPhysicalBytes <= 6*1024*1024*1024,"UI component process exceeded 6 GiB")
                    guard let result = value.chatResult, let first = value.tokenTimings.first else {
                        throw NativeContextFailure("UI result missing")
                    }
                    try require(value.status == 0 && value.ids == expected && value.history == 24 && value.evaluated == 23 && value.pending == 1,
                                "UI stock token/state differs")
                    try require(chat.workspace.current.messages.last?.text == result.text && !chat.running,
                                "UI publication missing")
                    if enabled { try latestUI("published",nonempty:true) }
                    samples.append(["request_id":result.id.uuidString,"trial":trial,"warmup":trial<3,
                        "mode":enabled ? "summary":"disabled","submitted_uptime_ns":start,"completed_uptime_ns":end,
                        "ttft_ms":Double(first.callbackUptimeNanoseconds-start)/1e6,"total_ms":Double(end-start)/1e6,
                        "tokens":value.ids,"history":value.history,"committed":value.evaluated,"pending":value.pending,
                        "footprint_current_bytes":footprint.physicalBytes,"process_lifetime_peak_bytes":footprint.lifetimePeakPhysicalBytes,
                        "thermal_state":ProcessInfo.processInfo.thermalState.rawValue,
                        "text_sha256":SHA256.hash(data:Data(result.text.utf8)).map { String(format:"%02x",$0) }.joined()])
                }
            }
            try JSONSerialization.data(withJSONObject:samples,options:[.prettyPrinted,.sortedKeys])
                .write(to:output.appendingPathComponent("samples.json"),options:.withoutOverwriting)
        } else {
            try prepare(); chat.sendOrStop()
            let good = try await poll()
            try require(good.status == 0 && good.ids == expected,"UI success control")
            try latestUI("published",nonempty:true)
            try prepare(String(repeating:"1",count:1000)); chat.sendOrStop()
            let rejected = try await poll()
            try require(rejected.chatResult?.rejected == true && chat.workspace.current.messages.isEmpty &&
                        chat.workspace.current.draft == String(repeating:"1",count:1000) && rejected.history == good.history &&
                        rejected.evaluated == good.evaluated && rejected.pending == good.pending,"UI rejection mutated draft/prefix")
            try latestUI("workerRejected",nonempty:false)
            faults.armDecode(); try prepare(); chat.sendOrStop()
            let failure = try await poll()
            try require(failure.status != 0 && failure.emitted == 8 && failure.history == 24 &&
                        chat.workspace.current.messages.last?.text.isEmpty == true,"UI output fault lost state")
            try latestUI("published",nonempty:false)
            try prepare(); chat.sendOrStop()
            let cancelled = try await poll(cancelAfterFirst:true)
            // A UI Stop can arrive during a synchronous decode. That step may
            // commit the last emitted ID before cancellation is acknowledged;
            // unlike callback-injected Stop, pending may therefore be zero.
            try require(cancelled.reason == 3 && cancelled.pending <= 1 && cancelled.emitted > 0 &&
                        cancelled.ids == Array(expected.prefix(Int(cancelled.emitted))) &&
                        cancelled.history == 15 + cancelled.emitted &&
                        cancelled.history == cancelled.evaluated + cancelled.pending &&
                        chat.workspace.current.messages.last?.interrupted == true,
                        "UI Stop control reason=\(cancelled.reason) emitted=\(cancelled.emitted) pending=\(cancelled.pending)")
            try latestUI("published",nonempty:true)
            try prepare(); faults.armBlock(); chat.sendOrStop()
            try require(await faults.waitUntilEntered(),"UI pre-start Stop gate timeout")
            chat.sendOrStop(); faults.release.signal()
            let preCancelled = try await poll()
            try require(preCancelled.reason == 3 && preCancelled.emitted == 0 &&
                        chat.workspace.current.messages.last?.text.isEmpty == true,"UI pre-start Stop control")
            try latestUI("published",nonempty:false)
            try prepare(); faults.armBlock(); chat.sendOrStop()
            try require(await faults.waitUntilEntered(),"UI queued control timeout")
            worker.close(); faults.release.signal()
            let closed = try await poll(requireClosed:true)
            try require(closed.closed && !chat.running && chat.workspace.current.messages.isEmpty &&
                        chat.workspace.current.draft == prompt,"UI queued close did not restore draft")
            // The real worker supplies a rejected/pre-execution terminal result
            // before its final closed snapshot. The controller restores input;
            // the no-terminal-result fallback is separately tested as a DTO.
            try latestUI("workerRejected",nonempty:false)
        }
        let rows = try summaryRows()
        try require(rows[0]["uiUpdateErrors"] as? Int == 0 && rows[0]["droppedRecords"] as? Int == 0 &&
                    rows[0]["budgetSkippedRecords"] as? Int == 0,"UI observation coverage incomplete")
        try worker.exportRequestSummaries().write(to:output.appendingPathComponent("summaries.jsonl"),options:Data.WritingOptions.withoutOverwriting)
        print("UI_PUBLICATION_LOADED_PASS mode=\(CommandLine.arguments[3]) records=\(rows.count-1) no_render_claim=true")
    }
}

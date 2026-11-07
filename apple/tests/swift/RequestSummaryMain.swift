import Foundation
import CToklet

private final class SummaryFaults: @unchecked Sendable {
    private let lock = NSLock()
    var block = false
    var failDecode = false
    var denySummary = false
    let entered = DispatchSemaphore(value:0), release = DispatchSemaphore(value:0)
    func armBlock() { lock.lock(); block = true; lock.unlock() }
    func armBudgetSkip() { lock.lock(); denySummary = true; lock.unlock() }
    func allowSummary() -> Bool {
        lock.lock(); defer { lock.unlock() }
        let allowed = !denySummary; denySummary = false; return allowed
    }
    func armDecode() { lock.lock(); failDecode = true; lock.unlock() }
    func dequeue(_ command: NativeCommand) {
        guard case .chat = command else { return }
        lock.lock(); let wait = block; block = false; lock.unlock()
        if wait { entered.signal(); _ = release.wait(timeout:.now()+30) }
    }
    func decode() throws {
        lock.lock(); let fail = failDecode; failDecode = false; lock.unlock()
        if fail { throw NativeContextFailure("Injected output failure") }
    }
}

// Explicit real-model pilot, separate from ordinary no-download CPU checks.
@main
struct RequestSummaryMain {
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func rows(_ worker: NativeWorker) throws -> [[String:Any]] {
        try worker.exportRequestSummaries().split(separator:10).map {
            try JSONSerialization.jsonObject(with:Data($0)) as! [String:Any]
        }
    }
    static func main() throws {
        // Deterministic metric formulas, empty/single events and ring pressure.
        var nativeFixture = TokletGenerationMetrics()
        nativeFixture.abi_version = TOKLET_ABI_VERSION
        nativeFixture.struct_size = UInt32(MemoryLayout<TokletGenerationMetrics>.size)
        nativeFixture.known_fields = UInt32(TOKLET_GENERATION_REPORT_KNOWN)
        nativeFixture.session_instance = 7
        try require(NativeGenerationMetrics(nativeFixture,expectedSession:7)?.prefillTokensPerSecond == nil,"Empty prefill rate")
        nativeFixture.known_fields = 0
        try require(NativeGenerationMetrics(nativeFixture,expectedSession:7) == nil,"Unknown native report accepted")
        nativeFixture.known_fields = UInt32(TOKLET_GENERATION_REPORT_KNOWN)
        nativeFixture.total_seconds = .nan
        try require(NativeGenerationMetrics(nativeFixture,expectedSession:7) == nil,"NaN native report accepted")
        let allowance = NativeRequestSummaryStore.reservedBytes
        try require(NativeRequestSummaryStore.fitsBudget(admitted:10,budget:10+allowance),"Exact-fit allowance rejected")
        try require(!NativeRequestSummaryStore.fitsBudget(admitted:10,budget:9+allowance),"Insufficient allowance accepted")
        try require(!NativeRequestSummaryStore.fitsBudget(admitted:UInt64.max,budget:0),"Admission subtraction underflow")
        let store = NativeRequestSummaryStore()
        var empty = store.begin(id:UUID(),maximum:1,generation:1,now:10,context:nil,configuration:nil)
        empty.startedUptimeNanoseconds = 20
        empty.finish(observation:nil,failure:true,status:1,reason:0,now:30)
        try require(empty.callbackTTFTMilliseconds == nil && empty.nativeDecodeTokensPerSecond == nil && empty.outcome == .rejected,"Empty metric semantics")
        var one = empty
        one.observeCallback(1,uptime:1_000_000_010)
        try require(one.nativeDecodeTokensPerSecond == nil,"Single token throughput")
        one.observeCallback(1.5,uptime:1_500_000_010)
        try require(one.nativeDecodeTokensPerSecond == 2 && one.callbackDecodeTokensPerSecond == 2,"Token throughput formula")
        for _ in 0..<33 { store.append(empty) }
        let (records,dropped) = store.snapshot()
        try require(records.count == 32 && dropped == 1,"Ring overwrite accounting")
        let wire = try NativeRequestSummaryStore.export(records,dropped:dropped)
        try require(wire.count < 1 << 20 && MemoryLayout<NativeRequestSummary>.stride < 16 * 1024,"Wire/collector bounds")
        do { _ = try NativeRequestSummaryStore.export(records+[empty],dropped:0); throw NativeContextFailure("Oversized export accepted") }
        catch let error as NativeContextFailure { try require(error.message == "Summary export exceeds capacity","Wrong export failure") }
        guard CommandLine.arguments.count >= 2 else { throw NativeContextFailure("Expected official model folder") }
        let faults = SummaryFaults()
        let measuring = CommandLine.arguments.count == 3
        let dequeue: @Sendable (NativeCommand) -> Void = { command in faults.dequeue(command) }
        let decode: @Sendable () throws -> Void = { try faults.decode() }
        let allow: @Sendable () -> Bool = { faults.allowSummary() }
        let worker = try NativeWorker(requestSummaryMode:.summary,captureSavedPrefix:true,
            beforeCommandDequeue:measuring ? nil : dequeue,
            beforeChatDecode:measuring ? nil : decode,
            allowSummaryCollection:measuring ? nil : allow)
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        func run(_ command: NativeCommand, initiallyCancelled: Bool = false) throws -> NativeSnapshot {
            try require(worker.submit(command,initiallyCancelled:initiallyCancelled),"Submit rejected")
            guard let s = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Worker timeout") }
            return s
        }
        var settings = NativeSettings(directory:CommandLine.arguments[1])
        settings.loadIdentity = .managedGemmaBF16; settings.loadRequestID = UUID()
        settings.metal = true; settings.chat = true; settings.capacity = 512; settings.chunk = 32
        settings.numericalBudget = 6 * 1024 * 1024 * 1024; settings.textBudget = 0
        let loaded = try run(.load(settings))
        try require(loaded.status == TOKLET_OK,loaded.message)
        let conversation = Data(#"[{"role":"user","content":"Name the capital of France."}]"#.utf8)
        func chat(_ maximum: UInt32 = 12, cancel: Int = 0) -> NativeCommand {
            .chat(id:UUID(),conversation:conversation,maximum:maximum,autoCancelAfter:cancel)
        }
        if CommandLine.arguments.count == 3 {
            // Paired order alternates; first three pairs are warmups. Raw timings
            // include the complete submit/idle acknowledgment in both modes.
            var samples = [[String:Any]]()
            for trial in 0..<13 {
                for enabled in trial % 2 == 0 ? [false,true] : [true,false] {
                    try require(worker.setRequestSummariesEnabled(enabled),"Mode switch failed")
                    let command = chat()
                    // External boundaries include collector setup, finalization,
                    // publication and idle acknowledgment in BOTH modes.
                    let start = DispatchTime.now().uptimeNanoseconds
                    let s = try run(command)
                    let end = DispatchTime.now().uptimeNanoseconds
                    try require(s.status == TOKLET_OK && s.ids == [818,5279,529,7001,563,5213,50429,84750],"Timing output changed")
                    guard case .chat(let requestID,_,_,_) = command else { throw NativeContextFailure("Expected Chat command") }
                    samples.append(["request_id":requestID.uuidString,"trial":trial,"warmup":trial<3,"mode":enabled ? "summary":"disabled",
                        "ttft_ms":Double(s.tokenTimings[0].callbackUptimeNanoseconds-start)/1e6,
                        "total_ms":Double(end-start)/1e6,"submitted_uptime_ns":start,"completed_uptime_ns":end,"tokens":s.ids,
                        "history":s.history,"committed":s.evaluated,"pending":s.pending])
                }
            }
            try JSONSerialization.data(withJSONObject:samples,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
            try worker.exportRequestSummaries().write(to:URL(fileURLWithPath:CommandLine.arguments[2]+".summaries.jsonl"))
        } else {
            let success = try run(chat())
            try require(success.status == TOKLET_OK && success.reason == 2 && success.emitted == 8,"Stock output differs")
            let failed = try run(.chat(id:UUID(),conversation:Data("bad JSON".utf8),maximum:12))
            try require(failed.status != TOKLET_OK && failed.savedPrefix == success.savedPrefix,"Rejected prefix changed")
            let single = try run(chat(1))
            try require(single.reason == 1 && single.emitted == 1,"Output-limit control")
            let cancelled = try run(chat(cancel:1))
            try require(cancelled.reason == 3 && cancelled.pending == 1,"Cancellation control")
            _ = try run(.suspend); let replay = try run(.replay)
            try require(replay.status == TOKLET_OK && replay.savedPrefix == cancelled.savedPrefix,"Replay control")
            let preCancelled = try run(chat(),initiallyCancelled:true)
            try require(preCancelled.reason == 3 && preCancelled.emitted == 0,"Pre-cancel control")
            let result = try rows(worker)
            try require(result.count == 6,"Expected five terminal summaries and manifest")
            try require(result[1]["outcome"] as? String == "stopToken" && result[2]["outcome"] as? String == "rejected" && result[3]["outcome"] as? String == "outputLimit" && result[4]["outcome"] as? String == "cancelled","Terminal outcomes")
            try require(result[1]["promptTokens"] != nil && result[1]["after"] != nil && result[1]["kvOwnedBytes"] != nil && result[1]["callbackTTFTMilliseconds"] != nil,"Missing available metrics")
            guard let phases = result[1]["nativeMetrics"] as? [String:Any] else { throw NativeContextFailure("Native phase report missing") }
            try require(result[1]["nativeMetricStatus"] as? String == "completed" &&
                        phases["prefillTokens"] as? Int == 15 && phases["sampledTokens"] as? Int == 9 &&
                        phases["emittedTokens"] as? Int == 8 && phases["prefillTokensPerSecond"] != nil,
                        "Native phase/count report differs")
            try require(result[2]["nativeMetrics"] == nil && result[2]["nativeMetricStatus"] as? String == "notStarted",
                        "Rejected request inherited native phase data")
            try require(result[3]["nativeDecodeTokensPerSecond"] == nil && result[4]["cancelToTerminalMilliseconds"] != nil,"Single/cancel metrics")
            faults.armDecode()
            let outputFailure = try run(chat())
            try require(outputFailure.status != 0 && outputFailure.emitted == 8 && outputFailure.history > 0,"Output failure lost committed state")
            let lastFailure = try rows(worker).last!
            try require(lastFailure["outcome"] as? String == "failed" && lastFailure["failedPhase"] as? String == "decoding","Output failure classification")
            _ = try run(.suspend)
            let afterSuspend = try run(chat(1))
            let recreatedRows = try rows(worker)
            try require(afterSuspend.reason == 1 && recreatedRows.last?["before"] == nil,"Session recreation summary")
            faults.armBlock()
            try require(worker.submit(chat()),"Queued request rejected")
            try require(faults.entered.wait(timeout:.now()+30) == .success,"Dequeue gate timeout")
            try require(!worker.setRequestSummariesEnabled(false),"Mode changed while busy")
            try require(!worker.submit(chat()),"Busy request accepted")
            do { _ = try worker.exportRequestSummaries(); throw NativeContextFailure("Busy export accepted") }
            catch let error as NativeContextFailure { try require(error.message == "Summaries disabled or worker busy","Wrong busy export failure") }
            worker.interrupt(unload:false); faults.release.signal()
            guard let stopped = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Queued cleanup timeout") }
            try require(!stopped.hasSession,"Queued cleanup failed")
            let queued = try rows(worker).last!
            try require(queued["outcome"] as? String == "cancelled" && queued["evaluationStarted"] as? Bool == false,"Queued cancellation missing")
            let previousBuildID = ProcessInfo.processInfo.environment["TOKLET_BUILD_ID"]
            setenv("TOKLET_BUILD_ID",String(repeating:"x",count:257),1)
            do { _ = try worker.exportRequestSummaries(); throw NativeContextFailure("Oversize manifest accepted") }
            catch let error as NativeContextFailure { try require(error.message == "Build identity exceeds export bound","Wrong manifest failure") }
            if let previousBuildID { setenv("TOKLET_BUILD_ID",previousBuildID,1) } else { unsetenv("TOKLET_BUILD_ID") }
            try require(try rows(worker)[0]["exportErrors"] as? Int == 1,"Export failure counter")
            faults.armBudgetSkip()
            let priorRows = try rows(worker)
            let skipped = try run(chat(1))
            let afterSkip = try rows(worker)
            try require(skipped.status == TOKLET_OK && skipped.ids == [818] && skipped.pending == 1,
                        "Telemetry budget skip changed generation")
            try require(afterSkip.count == priorRows.count && afterSkip[0]["budgetSkippedRecords"] as? Int == 1,
                        "Budget skip was not counted")
            let data = try worker.exportRequestSummaries()
            if CommandLine.arguments.count == 4 && CommandLine.arguments[2] == "--conformance-output" {
                try data.write(to:URL(fileURLWithPath:CommandLine.arguments[3]),options:Data.WritingOptions.withoutOverwriting)
            } else { FileHandle.standardOutput.write(data) }
            print("Summary formulas, bounds, stock Chat, rejection, limit, cancellation and replay passed")
        }
    }
}

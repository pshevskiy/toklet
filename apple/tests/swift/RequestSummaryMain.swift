import Foundation
import CToklet
import CoreFoundation
import CryptoKit

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
    static let warmupPairs = 3
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func parseMeasuredPairs(_ text: String) throws -> Int {
        try require(!text.isEmpty && text.utf8.allSatisfy { $0 >= 48 && $0 <= 57 },"Measured pairs require ASCII digits")
        guard let count = Int(text), (10...100).contains(count) else {
            throw NativeContextFailure("Measured pairs must be from 10 through 100")
        }
        return count
    }
    static func checkMeasuredPairs() throws {
        try require(try parseMeasuredPairs("10") == 10,"Minimum pair count")
        try require(try parseMeasuredPairs("100") == 100,"Maximum pair count")
        for invalid in ["9", "101", "-1", "+10", "1.0", "１０", ""] {
            var rejected = false
            do { _ = try parseMeasuredPairs(invalid) }
            catch is NativeContextFailure { rejected = true }
            try require(rejected,"Invalid measured pair count accepted")
        }
    }
    struct ExpectedOutput: Equatable {
        let tokens: [UInt32]
        let history, committed, pending: UInt64
        let reason, stopToken: UInt32

        func validate(input: UInt64, maximum: UInt32) throws {
            try RequestSummaryMain.require(!tokens.isEmpty && tokens.count <= Int(maximum),"Invalid workload output count")
            try RequestSummaryMain.require(reason == 1 || reason == 2,"Invalid workload terminal reason")
            // app_chat::gemma4::STOP_IDS; ordinary audited Chat stop policy.
            if reason == 1 {
                try RequestSummaryMain.require(tokens.count == Int(maximum) && stopToken == UInt32.max,"Invalid output-limit expectation")
            } else {
                try RequestSummaryMain.require(tokens.count < Int(maximum) && [UInt32(1),106,50].contains(stopToken),"Invalid hidden-stop expectation")
            }
            let sampled = UInt64(tokens.count) + (reason == 2 ? 1 : 0)
            try RequestSummaryMain.require(pending == 1 && history == input + sampled && history == committed + pending,
                        "Invalid workload output state")
        }
    }
    struct Workload {
        let sha256, conversationSHA256, inputSHA256: String
        let conversation: Data
        let input: UInt64
        let maximum: UInt32
        let expected: ExpectedOutput?
    }
    static func digest(_ data: Data) -> String {
        SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined()
    }
    static func unsigned(_ value: Any?, maximum: UInt64, field: String) throws -> UInt64 {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !["f","d"].contains(String(cString:number.objCType)),
              let result = UInt64(number.stringValue), result <= maximum else {
            throw NativeContextFailure("Expected bounded integer: " + field)
        }
        return result
    }
    static func hashField(_ value: Any?, field: String) throws -> String {
        guard let text = value as? String, text.utf8.count == 64,
              text.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw NativeContextFailure("Expected lowercase SHA256: " + field)
        }
        return text
    }
    static func loadWorkload(_ path: String, preflightOnly: Bool) throws -> Workload {
        let handle = try FileHandle(forReadingFrom:URL(fileURLWithPath:path))
        defer { try? handle.close() }
        let raw = try handle.read(upToCount:16 * 1024 + 1) ?? Data()
        try require(!raw.isEmpty && raw.count <= 16 * 1024,"Workload manifest byte bound")
        guard let value = try JSONSerialization.jsonObject(with:raw) as? [String:Any] else {
            throw NativeContextFailure("Expected workload object")
        }
        let keys: Set<String> = ["schema_version","workload_id","conversation_json","conversation_sha256",
            "input_token_count","input_tokens_sha256","maximum_output_tokens","capacity","chunk",
            "cache_policy","expected_output"]
        try require(Set(value.keys) == keys,"Workload fields differ")
        try require(try unsigned(value["schema_version"],maximum:1,field:"schema_version") == 1,"Workload schema differs")
        guard let name = value["workload_id"] as? String, (1...64).contains(name.utf8.count) else {
            throw NativeContextFailure("Invalid workload ID")
        }
        let nameBytes = Array(name.utf8)
        let alphanumeric: (UInt8) -> Bool = { (97...122).contains($0) || (48...57).contains($0) }
        try require(alphanumeric(nameBytes[0]) && nameBytes.allSatisfy({ alphanumeric($0) || $0 == 95 || $0 == 45 }),
                    "Invalid workload ID")
        guard let text = value["conversation_json"] as? String else { throw NativeContextFailure("Missing workload conversation") }
        let conversation = Data(text.utf8)
        try require((1...8192).contains(conversation.count),"Conversation byte bound")
        let conversationHash = try hashField(value["conversation_sha256"],field:"conversation_sha256")
        try require(digest(conversation) == conversationHash,"Conversation hash differs")
        guard let messages = try JSONSerialization.jsonObject(with:conversation) as? [[String:Any]],
              !messages.isEmpty && messages.count <= 63 && messages.count % 2 == 1 else {
            throw NativeContextFailure("Expected alternating Chat conversation")
        }
        for (index,message) in messages.enumerated() {
            try require(Set(message.keys) == ["role","content"] &&
                        message["role"] as? String == (index % 2 == 0 ? "user":"assistant"),"Conversation fields/role differ")
            guard let content = message["content"] as? String,
                  !content.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
                throw NativeContextFailure("Expected nonempty Chat content")
            }
        }
        let input = try unsigned(value["input_token_count"],maximum:512,field:"input_token_count")
        let maximum = UInt32(try unsigned(value["maximum_output_tokens"],maximum:256,field:"maximum_output_tokens"))
        try require(input > 0 && maximum > 0 && input + UInt64(maximum) <= 512,"Workload exceeds shared window")
        try require(try unsigned(value["capacity"],maximum:512,field:"capacity") == 512 &&
                    unsigned(value["chunk"],maximum:32,field:"chunk") == 32 &&
                    value["cache_policy"] as? String == "reset_chat_each_request","Workload admission policy differs")
        let inputHash = try hashField(value["input_tokens_sha256"],field:"input_tokens_sha256")
        var expected: ExpectedOutput?
        if value["expected_output"] is NSNull {
            try require(preflightOnly,"Timing workload requires frozen output")
        } else {
            guard let output = value["expected_output"] as? [String:Any],
                  Set(output.keys) == ["tokens","history","committed","pending","reason","stop_token"],
                  let tokens = output["tokens"] as? [Any], !tokens.isEmpty, tokens.count <= Int(maximum) else {
                throw NativeContextFailure("Invalid frozen output fields")
            }
            expected = ExpectedOutput(tokens:try tokens.map { UInt32(try unsigned($0,maximum:UInt64(UInt32.max),field:"output token")) },
                history:try unsigned(output["history"],maximum:512,field:"history"),
                committed:try unsigned(output["committed"],maximum:512,field:"committed"),
                pending:try unsigned(output["pending"],maximum:1,field:"pending"),
                reason:UInt32(try unsigned(output["reason"],maximum:2,field:"reason")),
                stopToken:UInt32(try unsigned(output["stop_token"],maximum:UInt64(UInt32.max),field:"stop_token")))
            try expected?.validate(input:input,maximum:maximum)
        }
        return Workload(sha256:digest(raw),conversationSHA256:conversationHash,inputSHA256:inputHash,
                        conversation:conversation,input:input,maximum:maximum,expected:expected)
    }
    // Called only after the existing end timestamp. No input IDs reach JSON.
    static func workloadSample(_ snapshot: NativeSnapshot, workload: Workload) throws -> (ExpectedOutput,[String:Any]) {
        guard snapshot.status == TOKLET_OK, let context = snapshot.liveContext,
              let configuration = snapshot.sessionConfiguration, let budget = snapshot.requestBudget,
              let prefix = snapshot.savedPrefix else { throw NativeContextFailure("Missing authoritative workload state") }
        try require(configuration.capacity == 512 && configuration.chunk == 32 && context.capacity == 512 &&
                    budget.context.capacity == 512 && configuration.modelInstance == context.modelInstance &&
                    configuration.sessionInstance == context.sessionInstance &&
                    budget.context.modelInstance == context.modelInstance && budget.context.sessionInstance == context.sessionInstance,
                    "Workload context/owner differs")
        try require(context.history == UInt64(snapshot.history) && context.committed == UInt64(snapshot.evaluated) &&
                    context.pending == UInt64(snapshot.pending) && prefix.count == Int(snapshot.history) &&
                    snapshot.savedHistory == prefix.count && prefix.count >= Int(workload.input),"Workload saved prefix differs")
        let required = workload.input + UInt64(workload.maximum)
        try require(budget.operation == 3 && budget.baseHistory == 0 && budget.input == workload.input &&
                    budget.historyAfterInput == workload.input && budget.output == UInt64(workload.maximum) &&
                    budget.required == required && budget.remaining == 512 - required,"Workload request admission differs")
        var encoded = Data()
        encoded.reserveCapacity(Int(workload.input) * 4)
        for token in prefix.prefix(Int(workload.input)) {
            var little = token.littleEndian
            withUnsafeBytes(of:&little) { encoded.append(contentsOf:$0) }
        }
        try require(digest(encoded) == workload.inputSHA256,"Rendered workload input hash differs")
        let actual = ExpectedOutput(tokens:snapshot.ids,history:UInt64(snapshot.history),committed:UInt64(snapshot.evaluated),
                                  pending:UInt64(snapshot.pending),reason:snapshot.reason,stopToken:snapshot.stopToken)
        try actual.validate(input:workload.input,maximum:workload.maximum)
        try require(snapshot.emitted == UInt32(snapshot.ids.count) && snapshot.tokenTimings.count == snapshot.ids.count,
                    "Workload callback count differs")
        if let expected = workload.expected { try require(actual == expected,"Frozen workload output differs") }
        return (actual,["workload_sha256":workload.sha256,"conversation_sha256":workload.conversationSHA256,
            "input_tokens":workload.input,"input_tokens_sha256":workload.inputSHA256,
            "maximum_output_tokens":workload.maximum,"capacity":512,"chunk":32,"cache_policy":"reset_chat_each_request",
            "admission":["operation":UInt64(budget.operation),"base_history":budget.baseHistory,"input":budget.input,
                "history_after_input":budget.historyAfterInput,"output":budget.output,"required":budget.required,"remaining":budget.remaining],
            "reason":snapshot.reason,"stop_token":snapshot.stopToken,"status":snapshot.status,
            "model_instance":context.modelInstance,"session_instance":context.sessionInstance])
    }
    static func rows(_ worker: NativeWorker) throws -> [[String:Any]] {
        try worker.exportRequestSummaries().split(separator:10).map {
            try JSONSerialization.jsonObject(with:Data($0)) as! [String:Any]
        }
    }
    static func benchmarkRawRow(_ data: Data, requestID: UUID) throws -> Data {
        try require(data.count <= 1 << 20,"Native export byte bound")
        let matches = try data.split(separator:10).dropFirst().filter { line in
            guard let row = try JSONSerialization.jsonObject(with:Data(line)) as? [String:Any] else {
                throw NativeContextFailure("Invalid native record")
            }
            return row["requestID"] as? String == requestID.uuidString
        }
        try require(matches.count == 1,"Benchmark raw record missing or duplicated")
        let raw = Data(matches[0])
        try require(raw.count <= 4096,"Benchmark native record bound")
        return raw
    }
    static func checkBenchmarkRawRows() throws {
        let id = UUID(uuidString:"00000000-0000-0000-0000-000000000001")!
        let header = #"{"kind":"request_summary_manifest"}"# + "\n"
        let selected = #"{"requestID":"00000000-0000-0000-0000-000000000001","spelling":1e0,"zero":-0,"large":9007199254740993}"#
        let other = #"{"requestID":"00000000-0000-0000-0000-000000000002"}"#
        let source = Data((header + selected + "\n" + other + "\n").utf8)
        try require(try benchmarkRawRow(source,requestID:id) == Data(selected.utf8),"Raw summary bytes changed")
        for invalid in [header + other + "\n", header + selected + "\n" + selected + "\n"] {
            do { _ = try benchmarkRawRow(Data(invalid.utf8),requestID:id); throw NativeContextFailure("Invalid raw capture accepted") }
            catch let error as NativeContextFailure {
                try require(error.message == "Benchmark raw record missing or duplicated","Wrong raw capture failure")
            }
        }
    }
    static func main() throws {
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else { throw NativeContextFailure("Expected official model folder") }
        let measuring: Bool
        var measuredPairs = 10, workloadPath: String?
        var preflightOnly = false
        if arguments.count == 2 || (arguments.count == 4 && arguments[2] == "--conformance-output") {
            measuring = false
        } else {
            try require(arguments.count >= 3 && !arguments[2].hasPrefix("--"),"Expected benchmark output path before options")
            measuring = true
            var index = 3, seen = Set<String>()
            while index < arguments.count {
                let option = arguments[index]
                try require(seen.insert(option).inserted,"Duplicate benchmark option")
                if option == "--preflight-only" {
                    preflightOnly = true; index += 1
                } else {
                    try require(option == "--measured-pairs" || option == "--workload","Unknown benchmark option")
                    try require(index + 1 < arguments.count && !arguments[index + 1].hasPrefix("--"),"Missing benchmark option value")
                    if option == "--measured-pairs" { measuredPairs = try parseMeasuredPairs(arguments[index + 1]) }
                    else { workloadPath = arguments[index + 1] }
                    index += 2
                }
            }
        }
        try require(!preflightOnly || workloadPath != nil,"Preflight requires workload")
        let workload = try workloadPath.map { try loadWorkload($0,preflightOnly:preflightOnly) }
        try checkMeasuredPairs()
        try checkBenchmarkRawRows()
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
        let faults = SummaryFaults()
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
        let conversation = workload?.conversation ?? Data(#"[{"role":"user","content":"Name the capital of France."}]"#.utf8)
        func chat(_ maximum: UInt32 = 12, cancel: Int = 0) -> NativeCommand {
            .chat(id:UUID(),conversation:conversation,maximum:maximum,autoCancelAfter:cancel)
        }
        if measuring {
            // Paired order alternates; first three pairs are warmups. Raw timings
            // include the complete submit/idle acknowledgment in both modes.
            var samples = [[String:Any]]()
            let captureByteLimit = 512 * 1024
            var captured = Data(), capturedIDs = Set<UUID>()
            var finalNativeManifest: [String:Any]?
            var preflightExpected: ExpectedOutput?
            let pairs = preflightOnly ? 1 : warmupPairs + measuredPairs
            for trial in 0..<pairs {
                var summaryID: UUID?
                for enabled in trial % 2 == 0 ? [false,true] : [true,false] {
                    try require(worker.setRequestSummariesEnabled(enabled),"Mode switch failed")
                    let command = chat(workload?.maximum ?? 12)
                    // External boundaries include collector setup, finalization,
                    // publication and idle acknowledgment in BOTH modes.
                    let start = DispatchTime.now().uptimeNanoseconds
                    let s = try run(command)
                    let end = DispatchTime.now().uptimeNanoseconds
                    var extra = [String:Any]()
                    if let workload {
                        let (actual,fields) = try workloadSample(s,workload:workload)
                        extra = fields
                        if preflightOnly {
                            if let expected = preflightExpected { try require(actual == expected,"Preflight mode outcomes differ") }
                            else { preflightExpected = actual }
                        }
                    } else {
                        try require(s.status == TOKLET_OK && s.ids == [818,5279,529,7001,563,5213,50429,84750],"Timing output changed")
                    }
                    try require(!s.tokenTimings.isEmpty && s.tokenTimings[0].callbackUptimeNanoseconds >= start &&
                                s.tokenTimings[0].callbackUptimeNanoseconds <= end,"Invalid first callback boundary")
                    guard case .chat(let requestID,_,_,_) = command else { throw NativeContextFailure("Expected Chat command") }
                    if enabled { summaryID = requestID }
                    var sample: [String:Any] = ["request_id":requestID.uuidString,"trial":trial,"warmup":!preflightOnly && trial<warmupPairs,"mode":enabled ? "summary":"disabled",
                        "ttft_ms":Double(s.tokenTimings[0].callbackUptimeNanoseconds-start)/1e6,
                        "total_ms":Double(end-start)/1e6,"submitted_uptime_ns":start,"completed_uptime_ns":end,"tokens":s.ids,
                        "history":s.history,"committed":s.evaluated,"pending":s.pending]
                    for (key,value) in extra { sample[key] = value }
                    samples.append(sample)
                }
                if !preflightOnly && measuredPairs > 10 {
                    // Housekeeping follows BOTH timed intervals, for either pair order.
                    let endsEnabled = trial % 2 == 0
                    if !endsEnabled { try require(worker.setRequestSummariesEnabled(true),"Capture enable failed") }
                    let snapshot: Data
                    do { snapshot = try worker.exportRequestSummaries() }
                    catch {
                        if !endsEnabled { _ = worker.setRequestSummariesEnabled(false) }
                        throw error
                    }
                    if !endsEnabled { try require(worker.setRequestSummariesEnabled(false),"Capture restore failed") }
                    guard let summaryID else { throw NativeContextFailure("Pair summary ID missing") }
                    let raw = try benchmarkRawRow(snapshot,requestID:summaryID)
                    try require(capturedIDs.insert(summaryID).inserted && capturedIDs.count <= 103,"Pair summary duplicated or capture count exceeded")
                    guard let manifestLine = snapshot.split(separator:10).first,
                          let manifest = try JSONSerialization.jsonObject(with:Data(manifestLine)) as? [String:Any] else {
                        throw NativeContextFailure("Native capture manifest missing")
                    }
                    finalNativeManifest = manifest
                    captured.append(raw); captured.append(10)
                    try require(captured.count <= captureByteLimit,"Benchmark capture byte bound")
                }
            }
            try JSONSerialization.data(withJSONObject:samples,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
            let summaryOutput = URL(fileURLWithPath:CommandLine.arguments[2]+".summaries.jsonl")
            if preflightOnly || measuredPairs == 10 {
                try worker.exportRequestSummaries().write(to:summaryOutput)
            } else {
                guard let finalNativeManifest else { throw NativeContextFailure("Final capture manifest missing") }
                try require(capturedIDs.count == warmupPairs + measuredPairs,"Incomplete benchmark capture")
                let header: [String:Any] = ["schemaVersion":1,"kind":"benchmark_request_summary_capture","nativeManifest":finalNativeManifest,
                    "recordCount":capturedIDs.count,"ringCapacity":32,"captureByteLimit":captureByteLimit,"outsideTimedPairs":true]
                var output = try JSONSerialization.data(withJSONObject:header,options:[.sortedKeys])
                output.append(10); output.append(captured)
                try require(output.count <= captureByteLimit,"Benchmark capture byte bound")
                try output.write(to:summaryOutput)
            }
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

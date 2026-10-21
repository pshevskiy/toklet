import Foundation
import Darwin

private func expectTeacher(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw NSError(domain:"TokletTeacherChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func mustReject(_ body: () throws -> Void) throws {
    var rejected = false
    do { try body() } catch { rejected = true }
    try expectTeacher(rejected,"Invalid teacher sequence accepted")
}
private final class SimulatedCommands {
    var prefix: [UInt32] = []
    var requests = 0
    var failAppendAt = 0
    var wrongScorePrefix = false
    var cancelPrompt = false
    var appends = 0
    func perform(_ command: NativeCommand) throws -> NativeSnapshot {
        requests += 1
        var score: NativeDistributionScore?
        var reason: UInt32 = 1
        switch command {
        case .reset: prefix.removeAll(keepingCapacity:true);reason = 0
        case .generate(let operation,let tokens,let maximum,let stops,let cancel):
            try expectTeacher(maximum == 0 && stops.isEmpty && cancel == 0 && tokens.count == 1,"Sampled/combined request")
            if operation == 0 { prefix = tokens;reason = cancelPrompt ? 3 : 1 }
            else {
                try expectTeacher(operation == 1,"Unexpected continue")
                appends += 1
                if appends == failAppendAt { throw NSError(domain:"InjectedAppend",code:1) }
                prefix += tokens
            }
        case .score(let target):
            try expectTeacher(!prefix.isEmpty,"Score before prompt")
            score = NativeDistributionScore(target:target,greedy:47,vocabulary:64,
                evaluated:UInt32(prefix.count + (wrongScorePrefix ? 1 : 0)),
                targetLogit:target == 47 ? 2 : 1,greedyLogit:2,
                negativeLogLikelihood:1.0+Double(target % 3)/10.0)
        default: throw NSError(domain:"UnexpectedCommand",code:1)
        }
        var state = NativeSnapshot();state.hasModel = true;state.hasSession = true
        state.history = UInt32(prefix.count);state.evaluated = state.history;state.savedHistory = prefix.count
        state.reason = reason;state.score = score
        return state
    }
}
private func simulatedChecks() throws -> [String] {
    var names: [String] = []
    do {
        let model = SimulatedCommands();var next = 0;var blocks: [NativeTeacherForcedBlock] = []
        let tokens = (0...10000).map { UInt32($0 % 64) }
        let result = try collectNativeTeacherForcedScores(tokens:tokens,perform:model.perform,onScore: { row in
            try expectTeacher(row.targetIndex == next && row.block == next/1024 && row.position == next%1024+1,"Frozen row order")
            try expectTeacher(row.score.target == tokens[next+1] && row.committedHistory == UInt32(row.position+1),"Frozen row target/commit")
            next += 1
        },onBlock:{ blocks.append($0) })
        try expectTeacher(result.targets == 10000 && result.blocks == 10 && next == 10000,"Frozen target total")
        try expectTeacher(blocks.count == 10 && blocks[0].targetStart == 0 && blocks[0].targetEnd == 1024
            && blocks[9].targetStart == 9216 && blocks[9].targetEnd == 10000 && blocks[9].committedHistory == 785,"Frozen overlap/partial final block")
        try expectTeacher(model.prefix == Array(tokens[9216...10000]) && model.requests == 20020,"Frozen final prefix/command count")
        let expected = (1...10000).reduce(0.0) { $0 + 1.0+Double(UInt32($1%64)%3)/10.0 }
        try expectTeacher(abs(result.negativeLogLikelihoodSum-expected) < 1e-8,"Compensated aggregate")
        names.append("frozen-10000-target-policy")
    }
    do {
        let model = SimulatedCommands();var rows: [NativeTeacherForcedScore] = []
        let result = try collectNativeTeacherForcedScores(tokens:[1,7,13,3,11,9],blockSize:3,perform:model.perform,onScore:{ rows.append($0) })
        try expectTeacher(result.targets == 5 && result.blocks == 2 && rows.map(\.targetIndex) == [0,1,2,3,4],"Partial target coverage")
        try expectTeacher(rows.map(\.position) == [1,2,3,1,2] && model.prefix == [3,11,9],"One-token overlap/reset")
        names.append("partial-block")
    }
    do {
        let model = SimulatedCommands();model.failAppendAt = 2;var emitted = 0
        try mustReject { _ = try collectNativeTeacherForcedScores(tokens:[1,7,13,3],blockSize:3,
            perform:model.perform,onScore:{ _ in emitted += 1 }) }
        try expectTeacher(emitted == 1 && model.prefix == [1,7] && model.requests == 6,"Failed append advanced completed rows")
        names.append("append-failure-prefix")
    }
    do {
        let model = SimulatedCommands();model.wrongScorePrefix = true;var emitted = 0
        try mustReject { _ = try collectNativeTeacherForcedScores(tokens:[1,7],perform:model.perform,onScore:{ _ in emitted += 1 }) }
        try expectTeacher(emitted == 0 && model.prefix == [1] && model.requests == 3,"Wrong prefix score appended")
        names.append("score-prefix-rejection")
    }
    do {
        let model = SimulatedCommands();model.cancelPrompt = true
        try mustReject { _ = try collectNativeTeacherForcedScores(tokens:[1,7],perform:model.perform,onScore:{ _ in }) }
        try expectTeacher(model.requests == 2 && model.appends == 0,"Cancelled prompt resumed")
        names.append("cancelled-prompt")
    }
    do {
        let model = SimulatedCommands();var rows = 0
        try mustReject { _ = try collectNativeTeacherForcedScores(tokens:[1,7,13,3],blockSize:3,perform:model.perform,onScore:{ row in
            if row.targetIndex == 1 { throw NSError(domain:"InjectedOutput",code:1) };rows += 1
        }) }
        try expectTeacher(rows == 1 && model.prefix == [1,7,13] && model.requests == 6,"Sink failure lost valid committed prefix")
        names.append("output-failure-prefix")
    }
    do {
        let model = SimulatedCommands()
        let invalidTokenCases: [[UInt32]] = [[],[1],Array(repeating:1,count:10002)]
        for tokens in invalidTokenCases {
            try mustReject { _ = try collectNativeTeacherForcedScores(tokens:tokens,perform:model.perform,onScore:{ _ in }) }
        }
        for block in [0,1025] {
            try mustReject { _ = try collectNativeTeacherForcedScores(tokens:[1,7],blockSize:block,perform:model.perform,onScore:{ _ in }) }
        }
        try expectTeacher(model.requests == 0,"Invalid policy issued commands")
        names.append("preflight-bounds")
    }
    return names
}
private struct TeacherChecks: Encodable {
    let passed: Bool
    let backend: String
    let simulated: [String]
    let rows: [NativeTeacherForcedScore]
    let summary: NativeTeacherForcedSummary
    let closed: Bool
    let released: Bool
    let physicalDeviceQualified = false
}
@main private struct TeacherForcingMain {
    static func main() {
        do {
            guard CommandLine.arguments.count == 2 || (CommandLine.arguments.count == 3 && CommandLine.arguments[2] == "metal") else { throw NSError(domain:"TeacherArguments",code:1) }
            let metal = CommandLine.arguments.count == 3
            let simulated = try simulatedChecks()
            var owner: NativeWorker? = try NativeWorker();weak let weakOwner = owner
            var rows: [NativeTeacherForcedScore] = []
            let summary: NativeTeacherForcedSummary
            do {
                let worker = owner!;defer { worker.close() }
                func perform(_ command: NativeCommand) throws -> NativeSnapshot {
                    try expectTeacher(worker.submit(command),"Actual command rejected")
                    guard let state = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TeacherDeadline",code:1) }
                    return state
                }
                var settings = NativeSettings(directory:CommandLine.arguments[1]);settings.metal = metal;settings.capacity = 4;settings.chunk = 1
                let loaded = try perform(.load(settings));try expectTeacher(loaded.status == 0,"Actual load failed")
                summary = try collectNativeTeacherForcedScores(tokens:[1,7,13,3],blockSize:3,perform:perform,onScore:{ rows.append($0) })
                // Independent frozen tiny-v1 prompt logits, rows0..2. These
                // losses were reduced with Python FP64 math.fsum, not the C API.
                let oracle = [4.389499721216301,5.483734151464388,5.5483128169001885]
                if !metal {
                    try expectTeacher(rows.count == 3 && zip(rows,oracle).allSatisfy { pair in abs(pair.0.score.negativeLogLikelihood-pair.1) < 1e-5 },"Independent teacher-forced CPU loss")
                }
                let state = worker.snapshot()
                try expectTeacher(state.history == 4 && state.evaluated == 4 && state.pending == 0 && state.savedHistory == 4,"Final target not committed")
                worker.close();try expectTeacher(worker.waitUntilClosed(seconds:120) && worker.snapshot().workerFinished,"Actual close deadline")
            }
            owner = nil
            let deadline = Date(timeIntervalSinceNow:5)
            while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
            try expectTeacher(weakOwner == nil,"Teacher worker retained")
            let result = TeacherChecks(passed:true,backend:metal ? "metal":"cpu",simulated:simulated,rows:rows,summary:summary,closed:true,released:true)
            print(String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
        } catch { fputs("Teacher-forcing check: \(error)\n",stderr);exit(1) }
    }
}

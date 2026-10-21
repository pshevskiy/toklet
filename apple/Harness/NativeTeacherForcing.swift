import Foundation

struct NativeTeacherForcedScore: Codable, Sendable {
    // Target zero scores tokens[1]; each block starts at position one.
    let targetIndex: Int
    let block: Int
    let position: Int
    let committedHistory: UInt32
    let score: NativeDistributionScore
}
struct NativeTeacherForcedBlock: Codable, Sendable {
    let index: Int
    let targetStart: Int
    let targetEnd: Int
    let committedHistory: UInt32
}
struct NativeTeacherForcedSummary: Encodable, Sendable {
    let targets: Int
    let blocks: Int
    let blockSize: Int
    let negativeLogLikelihoodSum: Double
    let meanNegativeLogLikelihood: Double
    let policy = "independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once"
}
private func teacherForcingError(_ message: String) -> NSError {
    NSError(domain:"TokletTeacherForcing",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}

// Cold diagnostic coordinator. The caller owns the admitted worker, immutable
// input IDs and output sink. perform must provide a bounded synchronous command
// and enforce its foreground/cancellation policy before and after completion.
// No owner or full logits escape. Ordinary closures keep both the native worker
// integration and deterministic failure checks at this coarse command boundary.
// On failure there is no complete summary: emitted rows cover only successfully
// appended targets, and the caller must close/replay according to native state.
func collectNativeTeacherForcedScores(tokens: [UInt32], blockSize: Int = 1024,
    perform: (NativeCommand) throws -> NativeSnapshot,
    onScore: (NativeTeacherForcedScore) throws -> Void,
    onBlock: (NativeTeacherForcedBlock) throws -> Void = { _ in }) throws -> NativeTeacherForcedSummary {
    guard (2...10001).contains(tokens.count), (1...1024).contains(blockSize) else {
        throw teacherForcingError("Invalid bounded teacher-forcing policy")
    }
    var total = 0.0; var correction = 0.0; var block = 0; var start = 0
    let targets = tokens.count - 1
    func command(_ request: NativeCommand) throws -> NativeSnapshot {
        let state = try perform(request)
        guard state.status == 0, !state.busy, !state.closed, state.hasModel, state.hasSession,
              state.ids.isEmpty, state.tokenTimings.isEmpty else {
            throw teacherForcingError("Teacher-forcing command failed or emitted sampled tokens")
        }
        return state
    }
    func committed(_ state: NativeSnapshot, _ count: Int, reason: UInt32) throws {
        guard state.history == UInt32(count), state.evaluated == UInt32(count), state.pending == 0,
              state.savedHistory == count, state.reason == reason else {
            throw teacherForcingError("Exact committed teacher-forcing prefix differs")
        }
    }
    while start < targets {
        let end = min(start + blockSize,targets)
        let reset = try command(.reset)
        try committed(reset,0,reason:0)
        guard reset.score == nil else { throw teacherForcingError("Reset retained an earlier score") }
        let initial = try command(.generate(operation:0,tokens:[tokens[start]],maximum:0,stops:[]))
        try committed(initial,1,reason:1)
        guard initial.score == nil else { throw teacherForcingError("Prompt retained an earlier score") }
        for index in start..<end {
            let position = index - start + 1
            let target = tokens[index + 1]
            let scored = try command(.score(target:target))
            try committed(scored,position,reason:1)
            guard let score = scored.score, score.target == target, score.evaluated == UInt32(position),
                  score.vocabulary > 0, target < score.vocabulary, score.greedy < score.vocabulary,
                  score.targetLogit.isFinite, score.greedyLogit.isFinite,
                  score.greedyLogit >= score.targetLogit,
                  score.negativeLogLikelihood.isFinite, score.negativeLogLikelihood >= 0 else {
                throw teacherForcingError("Score does not describe the exact next target")
            }
            // Commit the teacher target, including the last target of a block.
            // Never count it as appended or emit a completed row before success.
            let appended = try command(.generate(operation:1,tokens:[target],maximum:0,stops:[]))
            try committed(appended,position+1,reason:1)
            guard appended.score == nil else { throw teacherForcingError("Append retained an earlier score") }
            try onScore(NativeTeacherForcedScore(targetIndex:index,block:block,position:position,
                committedHistory:appended.history,score:score))
            let adjusted = score.negativeLogLikelihood - correction
            let next = total + adjusted
            correction = (next - total) - adjusted;total = next
        }
        try onBlock(NativeTeacherForcedBlock(index:block,targetStart:start,targetEnd:end,
            committedHistory:UInt32(end-start+1)))
        block += 1;start = end
    }
    guard total.isFinite, total >= 0 else { throw teacherForcingError("Invalid total teacher-forced loss") }
    return NativeTeacherForcedSummary(targets:targets,blocks:block,blockSize:blockSize,negativeLogLikelihoodSum:total,
        meanNegativeLogLikelihood:total/Double(targets))
}

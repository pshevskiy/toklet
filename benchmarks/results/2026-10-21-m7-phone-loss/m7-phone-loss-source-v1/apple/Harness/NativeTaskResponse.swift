import Foundation

struct NativeTaskChunk: Codable, Sendable {
    let index: Int
    let requested: UInt32
    let ids: [UInt32]
    let tokenTimings: [NativeTokenTiming]
    let reason: UInt32
    let stopToken: UInt32?
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
}
struct NativeTaskResponse: Codable, Sendable {
    let ids: [UInt32]
    let sampled: Int
    let reason: UInt32
    let stopToken: UInt32?
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
    let chunks: Int
}
private func taskResponseError(_ message: String) -> NSError {
    NSError(domain:"TokletTaskResponse",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}

// Cold raw-ID coordinator for one admitted task. A caller owns the model,
// capacity, command deadlines and a persistent cancellation/foreground latch.
// checkActive must reject a latched abort even between successful commands;
// NativeWorker.submit clears its per-command cancellation flag.
// Each worker command stays <=128 sampled tokens; the complete visible response
// stays <=256. A hidden stop consumes one sample and remains pending in history.
// No formatting/tokenizer or native owner crosses this boundary. On any failure
// there is no completed response; onChunk contains validated successful chunks
// only. The caller must close or reset the worker before another task.
func collectNativeTaskResponse(prompt: [UInt32], maximum: Int, capacity: UInt32, vocabulary: UInt32,
    stops: [UInt32], commandLimit: Int = 128,
    checkActive: () throws -> Void,
    perform: (NativeCommand) throws -> NativeSnapshot,
    onChunk: (NativeTaskChunk) throws -> Void = { _ in }) throws -> NativeTaskResponse {
    guard (1...306).contains(prompt.count), (1...256).contains(maximum),
          (1...128).contains(commandLimit), capacity <= 32768,
          UInt32(prompt.count + maximum) <= capacity, vocabulary > 0,
          prompt.allSatisfy({ $0 < vocabulary }), !stops.isEmpty, stops.count <= 2,
          stops.allSatisfy({ $0 < vocabulary }), Set(stops).count == stops.count else {
        throw taskResponseError("Invalid bounded raw task policy")
    }
    func command(_ request: NativeCommand) throws -> NativeSnapshot {
        try checkActive()
        let state = try perform(request)
        try checkActive()
        guard state.status == 0, !state.busy, !state.closed, state.hasModel, state.hasSession,
              state.score == nil else { throw taskResponseError("Task command incomplete or failed") }
        return state
    }
    let reset = try command(.reset)
    guard reset.history == 0, reset.evaluated == 0, reset.pending == 0,
          reset.savedHistory == 0, reset.reason == 0, reset.stopToken == .max,
          reset.ids.isEmpty, reset.tokenTimings.isEmpty else {
        throw taskResponseError("Task reset retained an earlier prefix or output")
    }
    var ids: [UInt32] = [];ids.reserveCapacity(maximum)
    var sampled = 0;var chunks = 0;var lastCallback: UInt64 = 0
    while sampled < maximum {
        let requested = min(commandLimit,maximum-sampled)
        let state = try command(.generate(operation:chunks == 0 ? 0 : 2,
            tokens:chunks == 0 ? prompt : [],maximum:UInt32(requested),stops:stops))
        guard state.ids.count == state.tokenTimings.count,
              state.ids.count <= requested,
              state.ids.allSatisfy({ $0 < vocabulary && !stops.contains($0) }) else {
            throw taskResponseError("Task emitted invalid IDs or callback coverage")
        }
        var elapsed = 0.0;var callback = lastCallback
        for (index,timing) in state.tokenTimings.enumerated() {
            guard timing.index == UInt32(index), timing.nativeElapsedSeconds.isFinite,
                  timing.nativeElapsedSeconds >= elapsed,
                  timing.callbackUptimeNanoseconds > 0, timing.callbackUptimeNanoseconds >= callback else {
                throw taskResponseError("Task callback order or clocks differ")
            }
            elapsed = timing.nativeElapsedSeconds;callback = timing.callbackUptimeNanoseconds
        }
        let hidden: UInt32?
        if state.reason == 1 {
            guard state.ids.count == requested, state.stopToken == .max else {
                throw taskResponseError("Task output limit truncated or retained a hidden stop")
            }
            hidden = nil
        } else if state.reason == 2 {
            guard stops.contains(state.stopToken), state.ids.count < requested else {
                throw taskResponseError("Task hidden stop or sample budget differs")
            }
            hidden = state.stopToken
        } else { throw taskResponseError("Task cancelled or stop reason differs") }
        let nextSampled = sampled + state.ids.count + (hidden == nil ? 0 : 1)
        let history = prompt.count + nextSampled
        guard nextSampled > sampled, nextSampled <= maximum,
              state.history == UInt32(history), state.evaluated == UInt32(history-1),
              state.pending == 1, state.savedHistory == history else {
            throw taskResponseError("Task sampled and committed prefix counts differ")
        }
        // A failing sink cannot turn partial output into a completed response.
        try onChunk(NativeTaskChunk(index:chunks,requested:UInt32(requested),ids:state.ids,
            tokenTimings:state.tokenTimings,reason:state.reason,stopToken:hidden,
            history:state.history,evaluated:state.evaluated,pending:state.pending))
        try checkActive()
        ids.append(contentsOf:state.ids);sampled = nextSampled;chunks += 1;lastCallback = callback
        if hidden != nil || sampled == maximum {
            return NativeTaskResponse(ids:ids,sampled:sampled,reason:state.reason,stopToken:hidden,
                history:state.history,evaluated:state.evaluated,pending:state.pending,chunks:chunks)
        }
    }
    throw taskResponseError("Task lacks a completed response")
}

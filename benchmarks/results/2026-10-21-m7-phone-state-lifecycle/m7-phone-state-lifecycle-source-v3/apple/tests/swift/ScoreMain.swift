import Foundation
import Darwin

private struct ScoreCheck: Codable {
    let name: String
    let status: UInt32
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
    let saved: Int
    let score: NativeDistributionScore?
}
private struct ScoreChecks: Codable {
    let passed: Bool
    let backend: String
    let checks: [ScoreCheck]
    let closed: Bool
    let released: Bool
    let physicalDeviceQualified: Bool
    let publicationRaces: [String]
}
private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain:"TokletScoreChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}

// The observer pauses a completed score before publication. The caller can
// deterministically enqueue cleanup/close while the score remains in flight.
private final class PublicationGate: @unchecked Sendable {
    let arrived = DispatchSemaphore(value:0)
    let resume = DispatchSemaphore(value:0)
    private let lock = NSLock()
    private var resumed = false
    private var published: Bool?
    func pause() {
        arrived.signal()
        let success = resume.wait(timeout:.now()+120) == .success
        lock.lock();resumed = success;lock.unlock()
    }
    func publication(_ value: Bool) { lock.lock();published = value;lock.unlock() }
    func completed() -> Bool { lock.lock();defer { lock.unlock() };return resumed && published == false }
}
private func runPublicationRace(directory: String, metal: Bool, close: Bool) throws {
    let gate = PublicationGate()
    var owner: NativeWorker? = try NativeWorker(beforeScorePublication:{ gate.pause() },afterScorePublication:{ gate.publication($0) })
    weak let weakOwner = owner
    do {
        let worker = owner!
        defer { gate.resume.signal();worker.close() }
        var settings = NativeSettings(directory:directory);settings.metal = metal
        for command in [NativeCommand.load(settings),.generate(operation:0,tokens:[1,7,13,3],maximum:0,stops:[])] {
            try require(worker.submit(command),"Race setup rejected")
            try require(worker.waitUntilIdle(seconds:120)?.status == 0,"Race setup failed")
        }
        try require(worker.submit(.score(target:47)),"Race score rejected")
        try require(gate.arrived.wait(timeout:.now()+120) == .success,"Race completion barrier")
        try require(worker.snapshot().busy && worker.snapshot().score == nil,"Score escaped completion barrier")
        if close { worker.close() } else { worker.interrupt(unload:true) }
        try require(worker.snapshot().score == nil,"Superseded score visible")
        gate.resume.signal()
        if close {
            try require(worker.waitUntilClosed(seconds:120),"Race close deadline")
        } else {
            guard let state = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletScoreChecks",code:6) }
            try require(state.status == 0 && !state.hasModel && !state.hasSession && state.score == nil,"Race unload failed")
            worker.close()
            try require(worker.waitUntilClosed(seconds:120),"Race unload close deadline")
        }
        let state = worker.snapshot()
        try require(gate.completed() && state.workerFinished && state.closed && state.score == nil,"Race final publication/release")
    }
    owner = nil
    let deadline = Date(timeIntervalSinceNow:5)
    while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
    try require(weakOwner == nil,"Race worker retain cycle")
}

@main private struct ScoreMain {
    static func main() {
        do {
            guard CommandLine.arguments.count == 2 || (CommandLine.arguments.count == 3 && CommandLine.arguments[2] == "metal") else {
                throw NSError(domain:"TokletScoreChecks",code:2)
            }
            let metal = CommandLine.arguments.count == 3
            var owner: NativeWorker? = try NativeWorker()
            weak let weakOwner = owner
            var rows: [ScoreCheck] = []
            do {
                let worker = owner!
                defer { worker.close() }
                func command(_ command: NativeCommand, _ name: String, status: UInt32 = 0) throws -> NativeSnapshot {
                    try require(worker.submit(command),"Rejected \(name)")
                    // Accepted work invalidates an earlier score immediately,
                    // including before the owning thread processes failures.
                    let immediate = worker.snapshot()
                    if immediate.busy { try require(immediate.score == nil,"Stale queued score: \(name)") }
                    guard let value = worker.waitUntilIdle(seconds:120) else {
                        throw NSError(domain:"TokletScoreChecks",code:3)
                    }
                    try require(value.status == status,"\(name): \(value.message)")
                    if status != 0 { try require(value.score == nil,"Failure published stale score") }
                    rows.append(ScoreCheck(name:name,status:value.status,history:value.history,
                        evaluated:value.evaluated,pending:value.pending,saved:value.savedHistory,score:value.score))
                    return value
                }
                _ = try command(.score(target:47),"unloaded-reject",status:1)
                var settings = NativeSettings(directory:CommandLine.arguments[1]);settings.metal = metal
                _ = try command(.load(settings),"load")
                _ = try command(.score(target:47),"empty-reject",status:1)
                var state = try command(.generate(operation:0,tokens:[1,7,13,3],maximum:0,stops:[]),"commit")
                try require(state.score == nil && state.ids.isEmpty && state.history == 4 && state.evaluated == 4 && state.pending == 0,"Zero-output prefix")
                state = try command(.score(target:47),"target")
                guard let first = state.score else { throw NSError(domain:"TokletScoreChecks",code:4) }
                try require(first.target == 47 && first.greedy == 47 && first.vocabulary == 64 && first.evaluated == 4,"Score identity")
                try require(first.targetLogit == first.greedyLogit && first.negativeLogLikelihood > 0,"Target/greedy loss")
                if !metal { try require(abs(first.negativeLogLikelihood - 1.6877391136919981) < 1e-5,"Independent tiny CPU loss") }
                state = try command(.score(target:47),"repeat")
                try require(state.score == first && state.history == 4 && state.evaluated == 4 && state.savedHistory == 4,"Read-only repeat")
                state = try command(.score(target:3),"alternative")
                guard let alternative = state.score else { throw NSError(domain:"TokletScoreChecks",code:5) }
                try require(alternative.greedy == 47 && abs((alternative.negativeLogLikelihood-first.negativeLogLikelihood)-(first.targetLogit-alternative.targetLogit)) < 1e-12,"Alternative target identity")
                state = try command(.score(target:64),"range-reject",status:1)
                try require(state.history == 4 && state.evaluated == 4 && state.pending == 0,"Rejected score changed prefix")
                state = try command(.score(target:47),"recover")
                try require(state.score == first,"Score recovery")
                state = try command(.generate(operation:1,tokens:[47],maximum:0,stops:[]),"teacher-append")
                try require(state.score == nil && state.history == 5 && state.evaluated == 5 && state.pending == 0,"Teacher append state")
                state = try command(.score(target:47),"after-append")
                try require(state.score?.evaluated == 5 && state.score?.greedy == 47,"Score after append")
                state = try command(.generate(operation:2,tokens:[],maximum:1,stops:[]),"pending")
                try require(state.score == nil && state.ids == [47] && state.history == 6 && state.evaluated == 5 && state.pending == 1,"Pending state")
                _ = try command(.score(target:47),"pending-reject",status:1)
                state = try command(.suspend,"suspend")
                try require(state.score == nil && !state.hasSession && state.savedHistory == 6,"Suspended score")
                _ = try command(.score(target:47),"suspend-reject",status:1)
                _ = try command(.replay,"replay")
                state = try command(.score(target:47),"after-replay")
                try require(state.score?.evaluated == 6 && state.history == 6 && state.evaluated == 6 && state.pending == 0,"Replayed committed score")
                state = try command(.reset,"reset")
                try require(state.score == nil && state.savedHistory == 0,"Reset score")
                _ = try command(.score(target:47),"reset-reject",status:1)
                state = try command(.unload,"unload")
                try require(state.score == nil && !state.hasSession && !state.hasModel,"Unloaded score")
                worker.close()
                try require(worker.snapshot().score == nil && worker.waitUntilClosed(seconds:120),"Close score/deadline")
                try require(worker.snapshot().workerFinished && worker.snapshot().score == nil,"Final score release")
            }
            owner = nil
            let deadline = Date(timeIntervalSinceNow:5)
            while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
            try require(weakOwner == nil,"Score worker retain cycle")
            try runPublicationRace(directory:CommandLine.arguments[1],metal:metal,close:false)
            try runPublicationRace(directory:CommandLine.arguments[1],metal:metal,close:true)
            let result = ScoreChecks(passed:true,backend:metal ? "metal":"cpu",checks:rows,closed:true,released:true,physicalDeviceQualified:false,publicationRaces:["interrupt-unload","close"])
            print(String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
        } catch {
            fputs("Swift score check: \(error)\n",stderr);exit(1)
        }
    }
}

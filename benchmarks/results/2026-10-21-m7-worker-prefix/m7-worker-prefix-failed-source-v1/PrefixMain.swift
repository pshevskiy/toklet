import Foundation

private func require(_ ok: @autoclosure () -> Bool, _ message: String) throws {
    if !ok() { throw NSError(domain:"TokletPrefix",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private struct Row: Codable {
    let name: String
    let prefix: [UInt32]?
    let saved: Int
    let status: UInt32
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
    let reason: UInt32
}
private struct Result: Encodable {
    let passed: Bool
    let backend: String
    let rows: [Row]
    let races: [String]
    let publicationWitnesses: [Publication]
    let closed: Bool
    let released: Bool
    let physicalDeviceQualified = false
}
private struct Publication: Encodable {
    let name: String
    let ownerAlive: Bool
    let prefixAbsent: Bool
    let scorePublished: Bool
}
private final class PrefixWitness: @unchecked Sendable {
    // The lock protects the weak owner and the actual post-publication witness.
    private let lock = NSLock()
    private weak var worker: NativeWorker?
    private var value: Publication?
    func attach(_ owner: NativeWorker) { lock.lock();worker = owner;lock.unlock() }
    func observe(_ published: Bool, name: String) {
        lock.lock();let owner = worker;lock.unlock()
        let state = owner?.snapshot()
        lock.lock();value = Publication(name:name,ownerAlive:owner != nil,
            prefixAbsent:state?.savedPrefix == nil,scorePublished:published);lock.unlock()
    }
    func snapshot() -> Publication? { lock.lock();defer { lock.unlock() };return value }
}
@main struct PrefixChecks {
    static func main() throws {
        let args = ProcessInfo.processInfo.arguments
        guard args.count == 2 || args.count == 3 && args[2] == "metal" else { throw NSError(domain:"TokletPrefix",code:2) }
        let metal = args.count == 3
        var settings = NativeSettings(directory:args[1]);settings.metal = metal
        var rows: [Row] = []
        func record(_ name: String, _ state: NativeSnapshot, _ expected: [UInt32]?, status: UInt32 = 0) throws {
            try require(state.savedPrefix == expected,"Exact prefix differs: \(name)")
            try require(state.status == status && !state.busy,"Status/completion differs: \(name)")
            if let prefix = state.savedPrefix { try require(prefix.count == state.savedHistory && prefix.count <= 32,"Prefix bound/count") }
            rows.append(Row(name:name,prefix:state.savedPrefix,saved:state.savedHistory,status:state.status,
                            history:state.history,evaluated:state.evaluated,pending:state.pending,reason:state.reason))
        }
        func perform(_ worker: NativeWorker, _ command: NativeCommand) throws -> NativeSnapshot {
            try require(worker.submit(command),"Submit rejected")
            guard let state = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletPrefix",code:3) }
            return state
        }
        var owner: NativeWorker? = try NativeWorker(captureSavedPrefix:true)
        weak let weakOwner = owner
        do {
            let worker = owner!;defer { worker.close() }
            try record("load",perform(worker,.load(settings)),[])
            let prompt: [UInt32] = [1,7,13,3]
            try record("prompt",perform(worker,.generate(operation:0,tokens:prompt,maximum:2,stops:[])),prompt+[47,47])
            try record("recoverable-failure",perform(worker,.score(target:47)),prompt+[47,47],status:1)
            try record("continue",perform(worker,.generate(operation:2,tokens:[],maximum:2,stops:[])),prompt+[47,47,47,47])
            try record("suspend",perform(worker,.suspend),prompt+[47,47,47,47])
            try require(worker.submit(.replay,initiallyCancelled:true),"Cancelled replay rejected")
            guard let cancelled = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletPrefix",code:3) }
            try require(!cancelled.hasSession && cancelled.reason == 3,"Cancelled replay created session")
            try record("cancelled-replay",cancelled,prompt+[47,47,47,47])
            let replayed = try perform(worker,.replay)
            try require(replayed.evaluated == 8 && replayed.pending == 0,"Replay commit semantics")
            try record("replay",replayed,prompt+[47,47,47,47])
            try record("after-replay",perform(worker,.generate(operation:2,tokens:[],maximum:1,stops:[])),prompt+[47,47,47,47,47])
            try record("reset",perform(worker,.reset),[])
            let interrupted = try perform(worker,.generate(operation:0,tokens:prompt,maximum:4,stops:[],autoCancelAfter:1))
            try require(interrupted.reason == 3 && interrupted.evaluated == 4 && interrupted.pending == 1,"Cancellation pending prefix")
            try record("cancel",interrupted,prompt+[47])
            worker.interrupt(unload:false)
            guard let suspended = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletPrefix",code:3) }
            try require(!suspended.hasSession && suspended.hasModel,"Lifecycle suspension ownership")
            try record("interrupt-suspend",suspended,prompt+[47])
            try record("cancel-prefix-replay",perform(worker,.replay),prompt+[47])
            try record("resume",perform(worker,.generate(operation:2,tokens:[],maximum:1,stops:[])),prompt+[47,47])
            try record("reset-hidden",perform(worker,.reset),[])
            let stopped = try perform(worker,.generate(operation:0,tokens:prompt,maximum:4,stops:[47]))
            try require(stopped.reason == 2 && stopped.ids.isEmpty && stopped.stopToken == 47,"Hidden EOS state")
            try record("hidden-stop",stopped,prompt+[47])
            try record("unload",perform(worker,.unload),[])
            worker.close();try require(worker.waitUntilClosed(seconds:120),"Close timeout")
            try record("close",worker.snapshot(),nil)
        }
        owner = nil
        let deadline = Date(timeIntervalSinceNow:5)
        while weakOwner != nil && Date() < deadline { Thread.sleep(forTimeInterval:0.002) }
        try require(weakOwner == nil,"Diagnostic snapshot retained native owner")
        do {
            let worker = try NativeWorker();defer { worker.close() }
            for (name,command) in [("default-load",NativeCommand.load(settings)),("default-prompt",.generate(operation:0,tokens:[1,7,13,3],maximum:1,stops:[])),
                                   ("default-suspend",.suspend),("default-replay",.replay),("default-reset",.reset),("default-unload",.unload)] {
                try record(name,perform(worker,command),nil)
            }
            worker.close();try require(worker.waitUntilClosed(seconds:120),"Default close timeout")
            try record("default-close",worker.snapshot(),nil)
        }
        var races: [String] = []
        var publications: [Publication] = []
        for close in [false,true] {
            let entered = DispatchSemaphore(value:0);let release = DispatchSemaphore(value:0)
            let witness = PrefixWitness();let name = close ? "close":"unload"
            let worker = try NativeWorker(captureSavedPrefix:true,
                beforeScorePublication:{ entered.signal();_ = release.wait(timeout:.now()+30) },
                afterScorePublication:{ published in witness.observe(published,name:name) })
            witness.attach(worker)
            defer { release.signal();worker.close() }
            _ = try perform(worker,.load(settings))
            _ = try perform(worker,.generate(operation:0,tokens:[1,7,13,3],maximum:0,stops:[]))
            try require(worker.submit(.score(target:47)),"Race score rejected")
            try require(entered.wait(timeout:.now()+30) == .success,"Race gate timeout")
            try require(worker.snapshot().savedPrefix == nil,"Busy command retained prefix")
            if close { worker.close() } else { worker.interrupt(unload:true) }
            try require(worker.snapshot().savedPrefix == nil,"Queued cleanup retained prefix")
            release.signal()
            if close {
                try require(worker.waitUntilClosed(seconds:120),"Race close timeout")
                try record("race-close",worker.snapshot(),nil)
            } else {
                guard let state = worker.waitUntilIdle(seconds:120) else { throw NSError(domain:"TokletPrefix",code:3) }
                try require(!state.hasModel && !state.hasSession && state.savedHistory == 0,"Race unload ownership")
                try record("race-unload",state,[])
                worker.close();try require(worker.waitUntilClosed(seconds:120),"Race unload close timeout")
            }
            guard let publication = witness.snapshot() else { throw NSError(domain:"TokletPrefix",code:4) }
            try require(publication.ownerAlive && publication.prefixAbsent && !publication.scorePublished,
                        "Queued cleanup republished prefix/score before cleanup")
            publications.append(publication);races.append(name)
        }
        let result = Result(passed:true,backend:metal ? "metal":"cpu",rows:rows,races:races,publicationWitnesses:publications,closed:true,released:true)
        print(String(decoding:try JSONEncoder().encode(result),as:UTF8.self))
    }
}

import Foundation
import CToklet

// Explicit Mac worker witness for the pinned offline official artifact.
@main
struct OfficialGemmaWorkerMain {
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw NativeContextFailure(message) }
    }

    static func main() throws {
        try require(CommandLine.arguments.count == 2, "Expected exact official model folder")
        let worker = try NativeWorker(captureSavedPrefix:true)
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        func run(_ command: NativeCommand) throws -> NativeSnapshot {
            try require(worker.submit(command), "Worker rejected command")
            guard let result = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Worker timed out") }
            return result
        }

        var settings = NativeSettings(directory:CommandLine.arguments[1])
        settings.loadIdentity = .managedGemmaBF16
        settings.loadRequestID = UUID()
        settings.metal = true; settings.chat = true
        settings.capacity = 512; settings.chunk = 32
        settings.numericalBudget = 6 * 1024 * 1024 * 1024; settings.textBudget = 0
        let loaded = try run(.load(settings))
        try require(loaded.status == TOKLET_OK && loaded.chatEnabled && loaded.acceptedLoadRequestID == settings.loadRequestID, loaded.message)
        guard let identity = loaded.acceptedIdentity, let context = loaded.modelContext,
              let memory = loaded.sessionMemory,
              case .officialGemma(let load,let session,let kv,let codec) = memory.reading else {
            throw NativeContextFailure("Official model owner facts were not published")
        }
        try require(identity.profile == .gemmaBF16 && identity.sourceRanges == nil && identity.sourceReads == nil &&
                    identity.matches(context) && context.declaredTotal == 131_072 && context.loadedLimit == 512,
                    "Official identity or context mismatch")
        try require(load > 5_700_000_000 && load <= settings.numericalBudget &&
                    session > 5_800_000_000 && session <= settings.numericalBudget &&
                    kv == 20_054_016 && codec == 1 << 30,
                    "Official owner accounting differs: \(load), \(session), \(kv), \(codec)")
        let request = Data(#"[{"role":"user","content":"Name the capital of France."}]"#.utf8)
        let id = UUID()
        let generated = try run(.chat(id:id,conversation:request,maximum:12))
        let expected: [UInt32] = [818,5279,529,7001,563,5213,50429,84750]
        try require(generated.status == TOKLET_OK && generated.ids == expected &&
                    generated.chatResult?.text == "The capital of France is **Paris**." &&
                    generated.reason == 2 && [1,106,50].contains(generated.stopToken), "Official Swift Chat output differs")
        let suspended = try run(.suspend)
        try require(!suspended.hasSession && suspended.sessionMemory == nil &&
                    suspended.acceptedIdentity == identity, "Suspend did not preserve model-only identity")
        let replayed = try run(.replay)
        try require(replayed.status == TOKLET_OK && replayed.hasSession && replayed.savedPrefix == generated.savedPrefix &&
                    replayed.sessionConfiguration?.sessionInstance != loaded.sessionConfiguration?.sessionInstance &&
                    replayed.acceptedIdentity == identity, "Replay did not retain exact token prefix")
        let cancelled = try run(.chat(id:UUID(),conversation:request,maximum:12,autoCancelAfter:1))
        try require(cancelled.status == TOKLET_OK && cancelled.reason == 3 && cancelled.ids == [818],
                    "Official Chat did not stop after the first token")
        let unloaded = try run(.unload)
        try require(!unloaded.hasModel && !unloaded.hasSession && unloaded.acceptedIdentity == nil &&
                    unloaded.sessionMemory == nil, "Unload retained official owner facts")
        print("Official Mac worker: exact Chat, memory, stop, suspend/replay, cancellation and unload passed")
        print("Owner bytes: \(load), \(session), \(kv), \(codec); hidden stop: \(generated.stopToken)")
    }
}

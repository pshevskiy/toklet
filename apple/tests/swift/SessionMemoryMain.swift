import Foundation
import CToklet

@main
struct SessionMemoryMain {
    static func check(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func controls() throws {
        var raw = TokletContextInfo()
        raw.abi_version = TOKLET_ABI_VERSION; raw.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        raw.backend = TOKLET_CONTEXT_CPU; raw.model_instance = 1; raw.session_instance = 2
        raw.engine_global_position_ceiling = 32768; raw.loaded_position_limit = 512; raw.session_capacity = 512
        raw.numerical_and_bridge_admitted_bytes = 1024 * 1024
        let context = try NativeContextFacts(raw)
        let configuration = try NativeSessionConfiguration(context:context,capacity:512,chunk:32,wholeSessionBudget:128 * 1024 * 1024)
        let snapshot = NativeSessionMemorySnapshot.capture(configuration:configuration) {
            throw NativeContextFailure(String(repeating:"x",count:1000))
        }
        guard case .unavailable(let reason) = snapshot.reading else { throw NativeContextFailure("Failed inspection became an allocation claim") }
        try check(reason.count == 256 && snapshot.matches(context,configuration:configuration),"Unavailable report lost binding or message bound")
        try check(!snapshot.matches(nil,configuration:configuration) && !snapshot.matches(context,configuration:nil),"Missing session retained memory report")
        raw.session_instance = 3
        try check(!snapshot.matches(try NativeContextFacts(raw),configuration:configuration),"Replacement session accepted old report")
        raw.session_instance = 2; raw.model_instance = 3
        try check(!snapshot.matches(try NativeContextFacts(raw),configuration:configuration),"Different model accepted old report")
        raw.model_instance = 1; raw.session_capacity = 256
        try check(!snapshot.matches(try NativeContextFacts(raw),configuration:configuration),"Changed capacity accepted old report")
        raw.session_capacity = 512; raw.history_tokens = 1; raw.committed_tokens = 1
        try check(snapshot.matches(try NativeContextFacts(raw),configuration:configuration),"History advancement invalidated explicitly historical snapshot")
        let other = try NativeSessionConfiguration(context:context,capacity:512,chunk:16,wholeSessionBudget:128 * 1024 * 1024)
        try check(!snapshot.matches(context,configuration:other),"Changed chunk accepted old report")
        print("SESSION_MEMORY_CONTROLS passed")
    }
    static func cpu(_ directory: String) throws {
        // Exclusive synthetic copy for an intentionally large CPU chunk.
        // Generation stays within the original fixture's32-position range.
        let source = URL(fileURLWithPath:directory)
        let fixture = FileManager.default.temporaryDirectory.appendingPathComponent("toklet-memory-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:fixture,withIntermediateDirectories:false)
        defer { try? FileManager.default.removeItem(at:fixture) }
        guard var config = try JSONSerialization.jsonObject(with:Data(contentsOf:source.appendingPathComponent("config.json"))) as? [String:Any] else {
            throw NativeContextFailure("Invalid tiny fixture configuration")
        }
        config["max_position_embeddings"] = 2048
        try JSONSerialization.data(withJSONObject:config,options:[.sortedKeys]).write(to:fixture.appendingPathComponent("config.json"))
        try FileManager.default.copyItem(at:source.appendingPathComponent("model.safetensors"),to:fixture.appendingPathComponent("model.safetensors"))
        let worker = try NativeWorker()
        var failure: Error?
        do {
            var settings = NativeSettings(directory:fixture.path)
            // Legacy CPU supports this chunk; bounded D1 inspection does not.
            settings.capacity = 2048; settings.chunk = 1025
            try check(worker.submit(.load(settings)),"CPU load submission failed")
            guard let loaded = worker.waitUntilIdle(seconds:30) else { throw NativeContextFailure("CPU load timeout") }
            try check(loaded.status == 0 && loaded.hasSession,"Optional inspection broke valid CPU load: \(loaded.message)")
            guard let memory = loaded.sessionMemory else { throw NativeContextFailure("Missing CPU snapshot") }
            guard case .unavailable = memory.reading else { throw NativeContextFailure("Unsupported report was presented as available") }
            try check(worker.submit(.generate(operation:0,tokens:[1,7],maximum:1,stops:[])),"CPU generation submission failed")
            guard let generated = worker.waitUntilIdle(seconds:30) else { throw NativeContextFailure("CPU generation timeout") }
            try check(generated.status == 0 && generated.emitted == 1
                && generated.sessionConfiguration == loaded.sessionConfiguration
                && generated.sessionMemory?.matches(generated.liveContext,configuration:generated.sessionConfiguration) == true,
                "Unavailable memory report broke valid generation")
        } catch { failure = error }
        worker.close(); try check(worker.waitUntilClosed(seconds:30),"CPU close timeout")
        try check(worker.snapshot().sessionMemory == nil,"Closed worker retained memory snapshot")
        if let failure { throw failure }
        print("SESSION_MEMORY_CPU unavailable_report_preserves_load_and_generation=true")
    }
    static func main() throws {
        try controls()
        if CommandLine.arguments.count == 2 { try cpu(CommandLine.arguments[1]) }
    }
}

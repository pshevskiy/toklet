import Foundation
import CToklet

private func require(_ value: @autoclosure () -> Bool,_ message: String) throws {
    if !value() { throw NSError(domain:"LongChatChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
@main
struct LongChatMain {
    static func main() throws {
        let root = URL(fileURLWithPath:CommandLine.arguments[1])
        let reference = try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("tests/fixtures/chat-long-v1/reference.json"))) as! [String:Any]
        let input = try JSONSerialization.data(withJSONObject:reference["messages"]!)
        let prompt = UInt32(reference["prompt_tokens"] as! Int)
        let inputIDs = (reference["ids"] as! [NSNumber]).map { $0.uint32Value }
        try require(prompt == 65 && NativeSettings.maximumChatOutputTokens == 256,"Independent input or output limit")
        let worker = try NativeWorker(captureSavedPrefix:true)
        defer { worker.close();_ = worker.waitUntilClosed(seconds:60) }
        func execute(_ command: NativeCommand,_ allowError: Bool = false) throws -> NativeSnapshot {
            try require(worker.submit(command),"Command rejected")
            guard let state = worker.waitUntilIdle(seconds:180) else { throw NSError(domain:"LongChatChecks",code:2) }
            try require(allowError || state.status == 0,"Native operation failed: " + state.message)
            return state
        }
        var settings = NativeSettings(directory:root.appendingPathComponent("models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c").path)
        settings.metal = true;settings.mlx = true;settings.chat = true
        settings.capacity = 512;settings.chunk = 32;settings.numericalBudget = 1024*1024*1024;settings.textBudget = 128*1024*1024
        let loaded = try execute(.load(settings))
        try require(loaded.hasSession && loaded.chatEnabled,"Missing chat owners")
        let firstID = UUID()
        let full = try execute(.chat(id:firstID,conversation:input,maximum:256))
        try require(full.ids.count > 129 && full.ids.count <= 256 && full.emitted == UInt32(full.ids.count),"No authoritative output beyond old bound")
        try require(full.reason == 1 || full.reason == 2,"Unexpected terminal reason")
        guard let originalPrefix = full.savedPrefix else { throw NSError(domain:"LongChatChecks",code:3) }
        try require(full.savedHistory == originalPrefix.count && originalPrefix.count == Int(full.history)
            && Array(originalPrefix.prefix(Int(prompt))) == inputIDs,"Consumed prompt differs from independent IDs")
        let sampled = full.emitted + (full.reason == 2 ? 1:0)
        try require(full.history == prompt + sampled && full.evaluated + 1 == full.history && full.pending == 1,"Exact sampled/committed/pending counts")
        try require(full.reason != 1 || sampled == 256,"Output-limit count")
        try require(full.reason != 2 || [UInt32(151643),151645].contains(full.stopToken),"Hidden stop ID")
        try require(full.tokenTimings.count == full.ids.count && full.chatResult?.id == firstID && full.chatResult?.error.isEmpty == true && full.chatResult?.text.isEmpty == false,"Visible completion")
        for (index,event) in full.tokenTimings.enumerated() { try require(event.index == UInt32(index),"Callback order") }
        func preserved(_ value: NativeSnapshot) throws {
            try require(value.status != 0 && value.history == full.history && value.evaluated == full.evaluated && value.pending == full.pending,
                "Rejected request changed committed prefix")
            guard let actualPrefix = value.savedPrefix else { throw NSError(domain:"LongChatChecks",code:3) }
            try require(value.savedHistory == full.savedHistory && actualPrefix == originalPrefix,"Rejected request changed sampled history")
            try require(value.ids.isEmpty && value.emitted == 0,"Rejected request retained output")
        }
        let tooLarge = try execute(.chat(id:UUID(),conversation:input,maximum:257),true)
        try preserved(tooLarge);try require(tooLarge.chatResult?.rejected == true,"Oversized output did not reject before generation")
        let overflow = try JSONSerialization.data(withJSONObject:[["role":"user","content":String(repeating:" word",count:400)]])
        let tooLong = try execute(.chat(id:UUID(),conversation:overflow,maximum:256),true)
        try preserved(tooLong);try require(tooLong.chatResult?.rejected == true,"Context did not reject before reset")
        try preserved(execute(.generate(operation:3,tokens:[1],maximum:129,stops:[]),true))
        try preserved(execute(.generate(operation:3,tokens:[1],maximum:128,stops:[],autoCancelAfter:129),true))
        var cancellations: [[String:UInt32]] = []
        for count in [128,129] {
            let id = UUID()
            let cancelled = try execute(.chat(id:id,conversation:input,maximum:256,autoCancelAfter:count))
            try require(cancelled.reason == 3 && cancelled.emitted == UInt32(count) && cancelled.ids.count == count,"Exact callback cancellation")
            try require(cancelled.ids == Array(full.ids.prefix(count)) && cancelled.history == prompt + UInt32(count)
                && cancelled.evaluated + 1 == cancelled.history && cancelled.pending == 1,"Cancelled deterministic prefix/counts")
            guard let cancelledPrefix = cancelled.savedPrefix else { throw NSError(domain:"LongChatChecks",code:3) }
            try require(cancelled.savedHistory == cancelledPrefix.count && Array(cancelledPrefix.prefix(Int(prompt))) == inputIDs
                && Array(cancelledPrefix.dropFirst(Int(prompt))) == cancelled.ids,"Cancelled sampled history")
            try require(cancelled.chatResult?.id == id && cancelled.chatResult?.reason == 3 && cancelled.chatResult?.error.isEmpty == true,"Cancelled visible response")
            cancellations.append(["emitted":cancelled.emitted,"history":cancelled.history,"evaluated":cancelled.evaluated,"pending":cancelled.pending])
        }
        let replay = try execute(.chat(id:UUID(),conversation:input,maximum:256))
        try require(replay.ids == full.ids && replay.emitted == full.emitted && replay.history == full.history
            && replay.evaluated == full.evaluated && replay.pending == full.pending && replay.savedPrefix == originalPrefix && replay.chatResult?.text == full.chatResult?.text,"Fresh replay after cancellation")
        _ = try execute(.unload);worker.close();try require(worker.waitUntilClosed(seconds:60),"Close deadline")
        let final = worker.snapshot()
        try require(!final.hasModel && !final.hasSession && final.emitted == 0 && final.ids.isEmpty,"Closed owners/output")
        let result: [String:Any] = ["passed":true,"prompt_tokens":prompt,"maximum_output":256,"emitted":full.emitted,"sampled":sampled,
            "history":full.history,"evaluated":full.evaluated,"pending":full.pending,"reason":full.reason,"ids":full.ids,
            "cancellations":cancellations,"exact_fresh_replay":true,"raw_diagnostic_bound_unchanged":true,"rejected_requests_preserve_prefix":true,"physical_device_qualified":false]
        print(String(decoding:try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]),as:UTF8.self))
    }
}

import Foundation
import AppKit
import SwiftUI
import CToklet

@MainActor
@main
struct ContextIntegrationMain {
    static func require(_ condition: @autoclosure () -> Bool,_ label: String) throws {
        if !condition() { throw NativeContextFailure("Context integration: " + label) }
    }
    static func settled(_ worker: NativeWorker) throws -> NativeSnapshot {
        guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Context worker deadline") }
        return state
    }
    static func execute(_ worker: NativeWorker,_ command: NativeCommand) throws -> NativeSnapshot {
        try require(worker.submit(command),"command accepted")
        let state = try settled(worker)
        try require(state.status == 0,"command completed: " + state.message)
        return state
    }
    static func fixture(_ directory: String) throws {
        let baseline = try runWorkerChecks(directory:directory,metal:false)
        try require(baseline.passed && baseline.closed,"existing worker lifecycle regression")
        let worker = try NativeWorker(captureSavedPrefix:true)
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let loaded = try execute(worker,.load(NativeSettings(directory:directory)))
        guard let retained = loaded.modelContext, let active = loaded.liveContext else { throw NativeContextFailure("Missing loaded facts") }
        try require(retained.declaredTotal == nil && retained.configuredCapacity == 32,"legacy declared window unknown")
        let exact = try execute(worker,.generate(operation:3,tokens:[1,7,13,3],maximum:28,stops:[]))
        try require(exact.requestBudget?.required == 32 && exact.requestBudget?.remaining == 0 && exact.history == 32 && exact.pending == 1,"raw exact boundary includes pending sample")
        let old = exact.liveContext
        try require(worker.submit(.generate(operation:3,tokens:[1,7,13,3],maximum:29,stops:[])),"raw overflow accepted for validation")
        let overflow = try settled(worker)
        try require(overflow.status == TOKLET_INVALID && overflow.liveContext == old && overflow.savedPrefix == exact.savedPrefix && overflow.requestBudget == nil && overflow.ids.isEmpty,"raw maximum+1 is atomic")
        let suspended = try execute(worker,.suspend)
        try require(suspended.modelContext == retained && suspended.liveContext == nil && !suspended.hasSession,"suspend retains no session facts")
        try require(worker.submit(.generate(operation:2,tokens:[],maximum:1,stops:[])),"suspended Continue command accepted")
        let missing = try settled(worker)
        try require(missing.status == TOKLET_INVALID && !missing.hasSession && missing.liveContext == nil,"Expert never silently recreates empty history")
        try require(worker.submit(.replay,initiallyCancelled:true),"cancelled replay accepted")
        let cancelled = try settled(worker)
        try require(cancelled.status == 0 && !cancelled.hasSession && cancelled.liveContext == nil && cancelled.modelContext == retained && cancelled.savedHistory == 32,"cancelled replay retains model-only facts/prefix")
        let replayed = try execute(worker,.replay)
        try require(replayed.liveContext?.sessionInstance != active.sessionInstance && replayed.modelContext == retained && replayed.pending == 0 && replayed.evaluated == 32,"replay gets fresh identity and complete prefix")
        let unloaded = try execute(worker,.unload)
        try require(unloaded.modelContext == nil && unloaded.liveContext == nil && unloaded.requestBudget == nil,"unload clears all facts")
        let reloaded = try execute(worker,.load(NativeSettings(directory:directory)))
        try require(reloaded.modelContext?.modelInstance != retained.modelInstance,"same-folder reload has new model identity")
        worker.close()
        let closing = worker.snapshot()
        try require(closing.modelContext == nil && closing.liveContext == nil && closing.requestBudget == nil,"close invalidates facts immediately")
        try require(worker.waitUntilClosed(seconds:30),"close completes")
        for unload in [false,true] { try publicationRace(directory,unload:unload) }
        print("TOKLET_CONTEXT_WORKER_JSON={\"passed\":true,\"raw_exact_and_plus_one\":true,\"suspend_replay_identity\":true,\"cancelled_replay\":true,\"stale_publication_cases\":2,\"legacy_declaration_unknown\":true}")
    }
    static func publicationRace(_ directory: String, unload: Bool) throws {
        let entered = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
        let worker = try NativeWorker(beforeScorePublication:{ entered.signal(); _ = proceed.wait(timeout:.now()+30) })
        defer { proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        _ = try execute(worker,.load(NativeSettings(directory:directory)))
        _ = try execute(worker,.generate(operation:3,tokens:[1,7,13,3],maximum:0,stops:[]))
        try require(worker.submit(.score(target:47)),"score reaches publication gate")
        try require(entered.wait(timeout:.now()+30) == .success,"publication gate entered")
        if unload { worker.interrupt(unload:true) } else { worker.close() }
        let invalidated = worker.snapshot()
        try require(invalidated.modelContext == nil && invalidated.liveContext == nil && invalidated.requestBudget == nil,"queued cleanup clears facts")
        proceed.signal()
        if unload {
            let final = try settled(worker)
            try require(!final.hasModel && final.modelContext == nil && final.liveContext == nil && final.score == nil,"finishing command cannot restore facts after unload")
        } else {
            try require(worker.waitUntilClosed(seconds:30),"closed publication completes")
            let final = worker.snapshot()
            try require(final.modelContext == nil && final.liveContext == nil && final.score == nil,"finishing command cannot restore facts after close")
        }
    }

    struct RenderedProbe {
        let draft: String
        let conversation: Data
        let ids: [UInt32]
    }
    static let prior = [["role":"user","content":"Earlier question"],["role":"assistant","content":"Earlier answer"]]
    static func probe(_ directory: String) throws -> RenderedProbe {
        try autoreleasepool {
            let profile = try HubProfile.identify(URL(fileURLWithPath:directory))
            try require(profile == .qwen3,"pinned Qwen3 probe")
            var options = TokletOptions(abi_version:TOKLET_ABI_VERSION,struct_size:UInt32(MemoryLayout<TokletOptions>.size),capacity:512,chunk:32,memory_limit_bytes:profile.numericalBudget)
            var model: OpaquePointer?, chat: OpaquePointer?; var error = TokletError()
            let path = Array(directory.utf8)
            let code = path.withUnsafeBufferPointer { toklet_model_load_mlx($0.baseAddress,$0.count,&options,&model,&error) }
            try require(code == TOKLET_OK,"load tokenizer-count probe")
            defer { if let chat { _ = toklet_chat_release(chat,nil) }; if let model { _ = toklet_model_release(model,nil) } }
            try require(toklet_chat_create(model,profile.textBudget,&chat,&error) == TOKLET_OK,"load pinned codec probe")
            for n in 250...430 {
                let draft = String(repeating:"Hello ",count:n)
                let data = try JSONSerialization.data(withJSONObject:prior+[["role":"user","content":draft]],options:[.sortedKeys])
                var count = 0
                let query = data.withUnsafeBytes { bytes in
                    toklet_chat_encode(chat,bytes.bindMemory(to:UInt8.self).baseAddress,bytes.count,nil,0,&count,&error)
                }
                if query == TOKLET_OK && count == 385 {
                    var ids = [UInt32](repeating:0,count:count)
                    let copied = data.withUnsafeBytes { bytes in ids.withUnsafeMutableBufferPointer { output in
                        toklet_chat_encode(chat,bytes.bindMemory(to:UInt8.self).baseAddress,bytes.count,output.baseAddress,output.count,&count,&error)
                    } }
                    try require(copied == TOKLET_OK && count == 385,"exact independent rendered count")
                    let codecOwner = chat; chat = nil
                    try require(toklet_chat_release(codecOwner,&error) == TOKLET_OK,"probe codec released before app load")
                    let modelOwner = model; model = nil
                    try require(toklet_model_release(modelOwner,&error) == TOKLET_OK,"probe model released before app load")
                    return RenderedProbe(draft:draft,conversation:data,ids:ids)
                }
            }
            throw NativeContextFailure("No bounded385-token rendered Qwen3 probe")
        }
    }
    static func real(_ directory: String,_ output: URL) throws {
        let counted = try probe(directory) // Release probe owners before the app loads weights.
        let entered = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
        let worker = try NativeWorker(captureSavedPrefix:true,beforeCommandDequeue:{ command in
            if case .chat = command { entered.signal(); _ = proceed.wait(timeout:.now()+30) }
        })
        defer { proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        var settings = NativeSettings(directory:directory)
        settings.metal = true; settings.mlx = true; settings.chat = true; settings.capacity = 512; settings.chunk = 32
        settings.numericalBudget = HubProfile.qwen3.numericalBudget; settings.textBudget = HubProfile.qwen3.textBudget
        let loaded = try execute(worker,.load(settings))
        guard let initial = loaded.liveContext, let model = loaded.modelContext else { throw NativeContextFailure("Missing real context facts") }
        try require(model.declaredTotal == 40960 && model.loadedLimit == 512 && model.configuredCapacity == 512,"real declared/loaded/selected separation")
        let rawExact = try execute(worker,.generate(operation:3,tokens:counted.ids,maximum:127,stops:[151645],autoCancelAfter:1))
        try require(rawExact.requestBudget?.required == 512 && rawExact.requestBudget?.input == 385,"real raw exact rendered-token boundary")
        try require(worker.submit(.generate(operation:3,tokens:counted.ids,maximum:128,stops:[151645])),"real raw overflow submitted")
        let rawOverflow = try settled(worker)
        try require(rawOverflow.status == TOKLET_INVALID && rawOverflow.liveContext == rawExact.liveContext && rawOverflow.savedPrefix == rawExact.savedPrefix && rawOverflow.requestBudget == nil,"real raw+1 preserves prefix")
        func send(_ maximum: UInt32) throws -> NativeSnapshot {
            try require(worker.submit(.chat(id:UUID(),conversation:counted.conversation,maximum:maximum,autoCancelAfter:1)),"chat boundary submitted")
            try require(entered.wait(timeout:.now()+30) == .success,"chat boundary dequeue held")
            proceed.signal(); return try settled(worker)
        }
        let chatExact = try send(127)
        try require(chatExact.status == 0 && chatExact.requestBudget?.required == 512 && chatExact.requestBudget?.input == 385 && chatExact.chatResult?.rejected == false,"real Chat exact shared boundary")
        let chatOverflow = try send(128)
        try require(chatOverflow.status == TOKLET_INVALID && chatOverflow.liveContext == chatExact.liveContext && chatOverflow.savedPrefix == chatExact.savedPrefix && chatOverflow.chatResult?.rejected == true,"real Chat+1 rejected before generation")
        let suspended = try execute(worker,.suspend)
        try require(suspended.modelContext == model && suspended.liveContext == nil,"real suspend retains only configuration")
        let recreated = try send(127)
        try require(recreated.status == 0 && recreated.modelContext == model && recreated.liveContext?.sessionInstance != initial.sessionInstance,"Chat Send recreates and preflights fresh session")
        // Friendly store uses the same worker and an isolated history file.
        let harness = HarnessStore(enabled:false); harness.worker = worker; harness.poll()
        let chat = FriendlyChatStore(harness:harness,fileURL:output.appendingPathComponent("history.json"))
        chat.open(URL(fileURLWithPath:directory))
        for _ in 0..<4 {
            _ = try settled(worker); harness.poll(); chat.observe(harness.state)
            if chat.ready { break }
        }
        try require(chat.ready,"friendly store completes its explicit unload/load handoff")
        let currentModel = harness.state.modelContext
        try require(currentModel?.modelInstance != model.modelInstance,"same-folder handoff receives new model identity")
        harness.state = loaded
        chat.open(URL(fileURLWithPath:directory))
        try require(chat.ready && harness.state.modelContext == currentModel && !worker.snapshot().busy,"stale same-folder snapshot cannot relabel/reload the current model")
        chat.workspace.conversations[chat.workspace.selectedIndex].messages = [ChatMessage(role:"user",text:"Earlier question"),ChatMessage(role:"assistant",text:"Earlier answer")]
        chat.workspace.outputLimit = 128
        let previous = chat.workspace.current.messages.map { ($0.id,$0.role,$0.text) }
        for newer in [Optional<String>.none,Optional("Newer draft"),Optional("")] {
            chat.edit(counted.draft); chat.sendOrStop()
            try require(entered.wait(timeout:.now()+30) == .success,"friendly rejected-turn gate")
            if let newer { if newer.isEmpty { chat.edit("New edit") }; chat.edit(newer) }
            proceed.signal(); _ = try settled(worker); harness.poll(); chat.observe(harness.state)
            let messages = chat.workspace.current.messages
            try require(messages.count == previous.count && zip(messages,previous).allSatisfy { $0.0.id == $0.1.0 && $0.0.role == $0.1.1 && $0.0.text == $0.1.2 },"rejected rendered+1 preserves prior turns")
            try require(chat.workspace.current.draft == (newer ?? counted.draft),"rejected rendered+1 preserves original/newer/cleared draft")
            try require(harness.state.chatResult?.rejected == true && !chat.notice.isEmpty,"rejection classification/notice")
        }
        chat.workspace.outputLimit = 131072; chat.save()
        try require(!chat.outputLimitAvailable,"unsupported saved preference remains unavailable")
        let restored = FriendlyChatStore(harness:harness,fileURL:output.appendingPathComponent("history.json"))
        try require(restored.workspace.outputLimit == 131072,"unsupported preference preserved on offline restore")
        chat.workspace.outputLimit = 128
        guard let displayed = harness.state.modelContext else { throw NativeContextFailure("Missing displayed model context") }
        // Render the actual shared product component; embedded native text
        // editors are unsupported by ImageRenderer and are checked separately.
        let view = VStack(alignment:.leading,spacing:12) {
            Text("Chat context").font(.headline)
            ContextWindowSummary(model:displayed,detailed:true).font(.callout)
            Text(chat.notice).font(.callout)
        }.padding(20).frame(width:393,alignment:.leading).background(Color.white)
        let renderer = ImageRenderer(content:view); renderer.scale = 2
        guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
              let png = NSBitmapImageRep(data:tiff)?.representation(using:.png,properties:[:]) else { throw NativeContextFailure("Context UI render failed") }
        try png.write(to:output.appendingPathComponent("context-window.png"))
        print("TOKLET_CONTEXT_REAL_JSON={\"passed\":true,\"declared\":40960,\"loaded\":512,\"selected\":512,\"rendered_input\":385,\"exact_reply\":127,\"overflow_reply\":128,\"raw_and_chat_boundary\":true,\"token_prefix_preserved\":true,\"cancel_after_tokens\":1,\"completed_512_positions\":false,\"suspend_send_new_identity\":true,\"rejected_draft_cases\":3,\"offline_preference_preserved\":true,\"phone_qualified\":false}")
    }
    static func main() throws {
        guard CommandLine.arguments.count >= 3 else { throw NativeContextFailure("Expected --fixture PATH or --model PATH OUTPUT") }
        if CommandLine.arguments[1] == "--fixture" { try fixture(CommandLine.arguments[2]) }
        else if CommandLine.arguments[1] == "--model", CommandLine.arguments.count == 4 {
            let output = URL(fileURLWithPath:CommandLine.arguments[3]); try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
            try real(CommandLine.arguments[2],output)
        } else { throw NativeContextFailure("Unknown context integration mode") }
    }
}

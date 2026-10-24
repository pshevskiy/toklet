import Foundation
import SwiftUI
import AppKit
import Combine

private actor VerificationGate {
    private var entered = false
    private var started: CheckedContinuation<Void,Never>?
    private var paused: CheckedContinuation<Void,Never>?
    func pause() async {
        entered = true; started?.resume(); started = nil
        await withCheckedContinuation { paused = $0 }
    }
    func waitForStart() async {
        if !entered { await withCheckedContinuation { started = $0 } }
    }
    func release() { paused?.resume(); paused = nil }
}

@main
@MainActor
struct MobileMain {
    static func require(_ value: @autoclosure () -> Bool,_ label: String) throws {
        if !value() { throw HubFailure("Mobile interface regression: " + label) }
    }
    static func content() throws {
        try require(ChatBlock.parse("Hello **world** 🌍") == [.prose("Hello **world** 🌍")],"plain and inline Markdown preserved")
        try require(ChatBlock.parse("Explain:\n```swift\nlet x = 42\n```\nDone") == [
            .prose("Explain:"),.code(language:"swift",text:"let x = 42"),.prose("Done")],"fenced Swift block")
        try require(ChatBlock.parse("~~~python\nprint('🌍')\n~~~") == [.code(language:"python",text:"print('🌍')")],"tilde fence and Unicode")
        try require(ChatBlock.parse("````rust\n```\nfn main() {}") == [.code(language:"rust",text:"```\nfn main() {}")],"unfinished long fence stays code")
        try require(ChatBlock.parse("    ```literal") == [.prose("    ```literal")],"indented prose remains literal")
        for (language,code) in [("swift","let text = \"🌍\" // 42\nreturn 7"),("python","# note\nprint('hello')"),("json","{\"n\":42,\"ok\":true}"),("rust","fn main() { /* comment */ }"),("","plain <html> & text")] {
            let spans = CodeHighlight.spans(code,language:language)
            try require(spans.map(\.text).joined() == code,"highlight preserves exact code for " + language)
            try require(String(CodeHighlight.attributed(code,language:language).characters) == code,"attributed code preserves Unicode")
        }
        let spans = CodeHighlight.spans("let x = 42 // note",language:"swift")
        try require(spans.contains { $0.text == "let" && $0.kind == .keyword },"keyword highlight")
        try require(spans.contains { $0.text == "42" && $0.kind == .number },"number highlight")
        try require(spans.contains { $0.text == "// note" && $0.kind == .comment },"comment highlight")
        for text in ["",String(repeating:"🌍👨‍👩‍👧‍👦e\u{301}",count:6000),String(repeating:"let value = 42 // code\n",count:32768),String(repeating:"x",count:32769)] {
            let pages = ChatPages.split(text)
            try require(pages.joined() == text,"long text pages preserve exact Unicode and newlines")
            try require(pages.allSatisfy { $0.count <= ChatPages.maximumCharacters },"individual layout/highlighting work is bounded")
            for page in pages {
                try require(CodeHighlight.spans(page,language:"swift").map(\.text).joined() == page,"long code pages preserve exact text")
            }
        }
    }
    static func polling(_ directory: String) throws {
        let entered = DispatchSemaphore(value:0), proceed = DispatchSemaphore(value:0)
        let worker = try NativeWorker(beforeCommandDequeue: { command in
            if case .generate = command { entered.signal(); _ = proceed.wait(timeout:.now() + 30) }
        })
        defer { proceed.signal(); worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let harness = HarnessStore(enabled:false); harness.worker = worker
        var publications = 0
        let observer = harness.$state.sink { _ in publications += 1 }
        defer { observer.cancel() }
        for _ in 0..<50 { harness.poll() }
        try require(publications == 1,"idle polling does not rebuild the interface")
        let settings = NativeSettings(directory:directory)
        func execute(_ command: NativeCommand) throws -> NativeSnapshot {
            let before = worker.snapshot().revision
            try require(worker.submit(command),"revision command accepted")
            guard let result = worker.waitUntilIdle(seconds:30) else { throw HubFailure("Revision check timed out") }
            try require(result.status == 0 && result.revision > before,"completed command advances revision")
            harness.poll(); let count = publications
            for _ in 0..<50 { harness.poll() }
            try require(publications == count,"unchanged command result is published once")
            return result
        }
        _ = try execute(.load(settings))
        let loaded = worker.snapshot().revision
        try require(worker.submit(.generate(operation:3,tokens:[1,7,13,3],maximum:2,stops:[])),"callback revision command accepted")
        try require(entered.wait(timeout:.now() + 30) == .success,"generation dequeue held")
        let queued = worker.snapshot().revision
        try require(queued > loaded,"submit advances revision before dequeue")
        proceed.signal()
        guard let generated = worker.waitUntilIdle(seconds:30) else { throw HubFailure("Callback revision check timed out") }
        try require(generated.status == 0 && generated.ids == [47,47] && generated.revision > queued + 2,"callback member mutations advance without recursive observer")
        _ = try execute(.suspend); _ = try execute(.replay); _ = try execute(.unload)
        let unloaded = worker.snapshot().revision
        worker.close(); try require(worker.waitUntilClosed(seconds:30),"revision close completes")
        try require(worker.snapshot().closed && worker.snapshot().revision > unloaded,"cleanup and terminal close advance revision")
    }
    static func workspace() throws {
        try checkChatWorkspace()
        var value = ChatWorkspace()
        try require(value.outputLimit == 128,"longer initial reply limit")
        value.edit("Explain a function")
        let pending = try value.begin()
        value.finish(NativeChatResult(id:pending.0,text:"An incomplete explanation",reason:1,error:""))
        try require(value.current.messages.last?.notice.contains("Response limit reached") == true,"length stop is visible")
        for limit in ChatWorkspace.outputLimits {
            value.outputLimit = limit
            let restored = try ChatWorkspace.restore(JSONEncoder().encode(value))
            try require(restored.outputLimit == limit,"old saved setting preserved")
        }
        var large = ChatWorkspace()
        large.edit("Write a long explanation")
        let largeRequest = try large.begin()
        // Text storage/rendering probe only: no model or token-count claim.
        let longReply = String(repeating:"word ",count:131072)
        large.finish(NativeChatResult(id:largeRequest.0,text:longReply,reason:1,error:""))
        let restored = try ChatWorkspace.restore(JSONEncoder().encode(large))
        try require(restored.current.messages.last?.text == longReply,"long reply saves without truncation")
        let escapedReply = String(repeating:"\u{0000}",count:ChatWorkspace.maximumMessageBytes)
        var escaped = ChatWorkspace(); escaped.edit("Test escaped storage")
        let escapedRequest = try escaped.begin()
        escaped.finish(NativeChatResult(id:escapedRequest.0,text:escapedReply,reason:1,error:""))
        let escapedBytes = try JSONEncoder().encode(escaped)
        try require(escapedBytes.count <= ChatWorkspace.maximumBytes,"completion reservation includes worst-case JSON escaping")
        let harness = HarnessStore(enabled:false)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("TokletReplyLimit-" + UUID().uuidString + ".json")
        let chat = FriendlyChatStore(harness:harness,fileURL:file)
        chat.workspace.outputLimit = 131072
        try require(chat.availableOutputLimits.allSatisfy { $0 <= NativeSettings.maximumChatOutputTokens },"settings only offers runnable native output limits")
        try require(chat.outputLimitAvailable == (NativeSettings.maximumChatOutputTokens >= 131072),"large saved preference follows native capability")
        try require(chat.workspace.outputLimit == 131072,"unsupported saved preference is not silently shortened")
        chat.workspace.outputLimit = 128
        try require(chat.outputLimitAvailable,"existing reply preference stays available")
    }
    static func hub() async throws {
        let gate = VerificationGate()
        let store = HubStore(admission:{ HubAdmission(unifiedGPU:true,physicalMemory:8 << 30,availableMemory:6 << 30) },verify:{ profile in
            if profile == .smol { await gate.pause() }
            // Intentionally ignore cancellation to check stale-result isolation.
            return URL(fileURLWithPath:"/verified-" + profile.key)
        })
        store.refresh(); await gate.waitForStart()
        try require(store.busy && !store.blocksSelection,"verification leaves selection enabled")
        store.select(.qwen)
        try require(store.profile == .qwen,"model changes while previous hash is pending")
        await gate.release(); await store.waitForIdle()
        try require(!store.busy && store.localURL?.path == "/verified-qwen","stale verification cannot replace new selection")

        let cancellation = VerificationGate()
        let cancelled = HubStore(admission:{ HubAdmission(unifiedGPU:true,physicalMemory:8 << 30,availableMemory:6 << 30) },verify:{ _ in
            await cancellation.pause(); return nil
        })
        cancelled.refresh(); await cancellation.waitForStart(); cancelled.cancel()
        await cancellation.release(); await cancelled.waitForIdle()
        try require(!cancelled.busy && !cancelled.blocksSelection,"Cancel clears operation ownership")

        let opening = VerificationGate()
        let exclusive = HubStore(profile:.qwen,admission:{ HubAdmission(unifiedGPU:true,physicalMemory:8 << 30,availableMemory:6 << 30) },verify:{ _ in
            await opening.pause(); return URL(fileURLWithPath:"/verified-qwen")
        })
        var didOpen = false
        exclusive.open { _ in didOpen = true }; await opening.waitForStart()
        exclusive.select(.mistral)
        try require(exclusive.blocksSelection && exclusive.profile == .qwen,"open protects captured model choice")
        exclusive.cancel(); await opening.release(); await exclusive.waitForIdle()
        try require(!didOpen && !exclusive.busy,"cancelled open cannot invoke callback")
        let replacementGate = VerificationGate()
        let replacement = HubStore(admission:{ HubAdmission(unifiedGPU:true,physicalMemory:8 << 30,availableMemory:6 << 30) },verify:{ profile in
            if profile == .smol { await replacementGate.pause() }
            return URL(fileURLWithPath:"/verified-" + profile.key)
        })
        var opened: [String] = []
        replacement.use(.smol) { opened.append($0.lastPathComponent) }
        await replacementGate.waitForStart()
        try require(!replacement.blocksSelection,"pending Use allows a newer model choice")
        replacement.use(.qwen) { opened.append($0.lastPathComponent) }
        await replacementGate.release(); await replacement.waitForIdle()
        try require(opened == ["verified-qwen"] && replacement.profile == .qwen,"one-tap Use opens only the latest model")
        try require(replacement.savedProfiles.contains(HubProfile.qwen.repository),"used saved model is shown as saved")
        try require(HubProfile.matching("qwen 0.5b") == [.qwen],"qualified matches ignore case")
        try require(HubProfile.matching("not-a-profile").isEmpty,"unsupported search stays explicit")
    }
    static func loading(_ directory: String) throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("TokletMobile-" + UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at:file) }
        let harness = HarnessStore(enabled:true)
        guard let worker = harness.worker else { throw HubFailure("Missing model-check worker") }
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let chat = FriendlyChatStore(harness:harness,fileURL:file)
        let folder = URL(fileURLWithPath:directory)
        chat.open(folder)
        try require(chat.loadingModel,"load exposes progress immediately")
        guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Model load timed out") }
        harness.poll(); chat.observe(harness.state)
        try require(chat.ready && !chat.loadingModel,"load completes into ready state")
        chat.workspace.outputLimit = 32; chat.edit("Hello!"); chat.sendOrStop()
        guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Model reply timed out") }
        harness.poll(); chat.observe(harness.state)
        try require(!chat.running && chat.workspace.current.messages.last?.text.isEmpty == false,"real response returns")
        let history = worker.snapshot().history
        let conversation = chat.workspace.selected
        chat.open(folder)
        try require(chat.ready && !chat.loadingModel && !worker.snapshot().busy,"same model open is idempotent")
        try require(worker.snapshot().history == history && chat.workspace.selected == conversation,"same model open preserves token prefix and conversation")
        chat.open(file.deletingLastPathComponent().appendingPathComponent("not-a-model-" + UUID().uuidString))
        try require(chat.ready && !chat.notice.isEmpty && worker.snapshot().history == history,"invalid model preserves current session")
        print("TOKLET_MOBILE_LOCAL_MODEL_PASS")
    }
    static func switching(_ first: String,_ second: String) async throws {
        let harness = HarnessStore(enabled:true)
        guard let worker = harness.worker else { throw HubFailure("Missing switch worker") }
        defer { worker.close(); _ = worker.waitUntilClosed(seconds:30) }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("TokletSwitch-" + UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at:file) }
        var freshAdmissions = 0
        let chat = FriendlyChatStore(harness:harness,fileURL:file,admission:{
            try require(!worker.snapshot().hasModel,"fresh admission runs after the old model is released")
            freshAdmissions += 1
            return HubAdmission(unifiedGPU:true,physicalMemory:8 << 30,availableMemory:6 << 30)
        })
        func finishLoad(_ name: String) throws {
            for _ in 0..<8 {
                guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Model switch timed out") }
                harness.poll(); chat.observe(harness.state)
                if chat.ready && chat.modelName == name { return }
            }
            throw HubFailure("Switch never became ready: " + chat.notice + harness.state.message)
        }
        chat.open(URL(fileURLWithPath:first)); try finishLoad(HubProfile.smol.name)
        let hub = HubStore(admission:{
            HubAdmission(unifiedGPU:true,physicalMemory:8 << 30,availableMemory:worker.snapshot().hasModel ? 0 : 6 << 30)
        },verify:{ profile in URL(fileURLWithPath:profile == .smol ? first : second) })
        var rejectedHandoff = false
        hub.use(.qwen) { _ in rejectedHandoff = true }; await hub.waitForIdle()
        try require(!rejectedHandoff && chat.ready,"resident-memory rejection preserves active model")
        for profile: HubProfile in [.qwen,.smol] {
            hub.use(profile,replacingLoadedModel:true) { chat.open($0) }; await hub.waitForIdle()
            try finishLoad(profile.name)
        }
        try require(freshAdmissions == 3,"every replacement receives fresh post-release admission")
        print("TOKLET_MOBILE_REPLACEMENT_ADMISSION_PASS")
    }
    static func render(_ directory: String) throws {
        _ = NSApplication.shared
        let folder = URL(fileURLWithPath:directory)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        let code = "func greet(name: String) -> String {\n    // Build a greeting.\n    return \"Hello, \\(name)! 🌍\"\n}"
        for (name,scheme) in [("light",ColorScheme.light),("dark",ColorScheme.dark)] {
            // ImageRenderer omits native scroll/button surfaces. Export the
            // actual highlighted text at phone width, not a simulated app view.
            let content = VStack(alignment:.leading,spacing:16) {
                Text("Highlighted Swift at phone width").font(.headline)
                Text(CodeHighlight.attributed(code,language:"swift"))
                    .font(.system(.callout,design:.monospaced)).padding(12)
                    .background(.quaternary,in:RoundedRectangle(cornerRadius:12))
            }.padding(24).frame(width:393)
                .background(scheme == .light ? Color.white : Color.black).environment(\.colorScheme,scheme)
            let renderer = ImageRenderer(content:content); renderer.scale = 2
            guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
                  let bytes = NSBitmapImageRep(data:tiff)?.representation(using:.png,properties:[:]) else { throw HubFailure("Could not render code preview") }
            try bytes.write(to:folder.appendingPathComponent(name + ".png"))
        }
        print("TOKLET_MOBILE_HIGHLIGHT_PREVIEW_EXPORTED")
    }
    static func main() async throws {
        try content(); try workspace(); try await hub()
        if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--switching" { try await switching(CommandLine.arguments[2],CommandLine.arguments[3]) }
        else if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--polling" { try polling(CommandLine.arguments[2]) }
        else if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--model" { try loading(CommandLine.arguments[2]) }
        else if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--render" { try render(CommandLine.arguments[2]) }
        else if CommandLine.arguments.count != 1 { throw HubFailure("Expected --model LOCAL_FOLDER or no arguments") }
        print("TOKLET_MOBILE_INTERFACE_PASS")
    }
}

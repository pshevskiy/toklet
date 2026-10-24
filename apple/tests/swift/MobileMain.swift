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
    static func dashboard() async throws {
        var state = NativeSnapshot()
        try require(GenerationMetrics(state).firstTokenSeconds == nil && GenerationMetrics(state).decodeTokensPerSecond == nil,"idle has no fabricated speed")
        state.requestStartedUptimeNanoseconds = 1_000_000_000
        state.tokenTimings = [NativeTokenTiming(index:0,nativeElapsedSeconds:0.1,callbackUptimeNanoseconds:1_250_000_000)]
        try require(GenerationMetrics(state).firstTokenSeconds == 0.25 && GenerationMetrics(state).decodeTokensPerSecond == nil,"TTFT includes queue/preparation; one token has no decode rate")
        state.tokenTimings.append(NativeTokenTiming(index:1,nativeElapsedSeconds:0.15,callbackUptimeNanoseconds:1_300_000_000))
        state.tokenTimings.append(NativeTokenTiming(index:2,nativeElapsedSeconds:0.2,callbackUptimeNanoseconds:1_350_000_000))
        state.requestCompletedUptimeNanoseconds = 1_400_000_000
        let metrics = GenerationMetrics(state)
        try require(metrics.tokens == 3 && metrics.decodeTokensPerSecond == 20 && metrics.totalSeconds == 0.4,"decode rate counts intervals after first callback and total ends at completion")
        state.requestStartedUptimeNanoseconds = 2_000_000_000
        try require(GenerationMetrics(state).firstTokenSeconds == nil && GenerationMetrics(state).totalSeconds == nil,"reversed clock observations never underflow")
        let generous = HubAdmission(unifiedGPU:true,physicalMemory:8 << 30,availableMemory:6 << 30)
        try require(HubRecommendations.ranked(generous).map(\.profile) == [.mistral,.qwen,.smol],"balanced ranking favors capacity when memory allows")
        let phone = HubAdmission(unifiedGPU:true,physicalMemory:6 << 30,availableMemory:2 << 30)
        try require(HubRecommendations.ranked(phone).map(\.profile) == [.qwen,.smol],"phone allowance excludes over-budget Mistral")
        try require(HubRecommendations.ranked(phone,task:.fast).map(\.profile) == [.smol,.qwen],"fast-reply preference changes ordering among admitted models")
        try require(HubRecommendations.ranked(phone,task:.coding).first?.profile == .qwen,"coding preference favors capacity without admitting an over-budget model")
        let low = HubAdmission(unifiedGPU:true,physicalMemory:6 << 30,availableMemory:0)
        let resident = HubRecommendations.ranked(low,active:.qwen)
        try require(HubRecommendations.ranked(low).isEmpty && resident.map(\.profile) == [.qwen],"resident model need not fit twice; unknown replacements are excluded")
        try require(resident.first?.active == true && resident.first?.headroom == 0,"resident exemption cannot fabricate free memory")
        for task in HubRankingTask.allCases {
            for allowance: HubAdmission in [low,phone,generous,HubAdmission(unifiedGPU:true,physicalMemory:UInt64.max,availableMemory:UInt64.max)] {
                let rows = HubRecommendations.ranked(allowance,task:task)
                try require(rows == HubRecommendations.ranked(allowance,task:task),"ranking is deterministic")
                for row in rows {
                    try require(HubProfile.catalog.contains(row.profile),"ranking only contains supported pinned profiles")
                    for value in [row.capacity,row.speed,row.headroom,row.score] {
                        try require(value.isFinite && (0...100).contains(value),"all score components are bounded and finite")
                    }
                }
            }
        }
        let moreRoom = HubRecommendations.ranked(generous).first(where:{ $0.profile == .qwen })!
        let lessRoom = HubRecommendations.ranked(phone).first(where:{ $0.profile == .qwen })!
        try require(moreRoom.headroom > lessRoom.headroom,"fresh memory allowance affects the score")
        try require(HubRecommendations.ranked(HubAdmission(unifiedGPU:false,physicalMemory:8 << 30,availableMemory:6 << 30)).isEmpty,"unsupported GPU gets no recommendation")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletRanking-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let offline: HubFetch = { _,_,_,_ in throw HubFailure("Injected offline connection") }
        let hub = HubStore(hub:ModelHub(root:root,fetch:offline),admission:{ phone }); hub.findBest()
        await hub.waitForIdle()
        try require(hub.recommendations.map(\.profile) == [.qwen,.smol] && hub.recommendationMessage.contains("3 downloadable"),"shortlist explains why fewer than five are available")
        hub.findBest(task:.fast)
        try require(hub.recommendations.first?.profile == .smol && hub.recommendationTask == .fast,"controller publishes the selected preference and reranks")
        hub.findBest()
        await hub.waitForIdle()
        try require(hub.recommendationTask == .fast,"refresh preserves task preference")
        let unavailable = HubStore(hub:ModelHub(root:root,fetch:offline),admission:{ throw HubFailure("Memory reading failed") }); unavailable.findBest(task:.coding)
        await unavailable.waitForIdle()
        try require(unavailable.recommendations.isEmpty && unavailable.recommendationMessage.contains("Memory reading failed"),"missing memory cannot produce recommendations")
        try require(NativeSettings.chatContextPositions == 512,"ranking displays the actual current Chat capacity")
        var workspace = ChatWorkspace(); workspace.outputLimit = 256
        workspace.edit("delete this draft"); let pending = try workspace.begin()
        let original = workspace.selected
        try workspace.newChat(modelKey:"other-model"); let second = workspace.selected
        workspace.remove(original)
        try require(workspace.selected == second && workspace.conversations.count == 1,"deleting a different dialog preserves selection")
        workspace.removeAll(modelKey:"loaded-model")
        try require(workspace.conversations.count == 1 && workspace.current.messages.isEmpty && workspace.current.draft.isEmpty && workspace.current.modelKey == "loaded-model" && workspace.outputLimit == 256,"delete all resets messages/drafts but keeps active model and reply preference")
        try require(!workspace.finish(NativeChatResult(id:pending.0,text:"late",reason:1,error:"")),"late reply cannot resurrect a deleted dialog")
        _ = try ChatWorkspace.restore(JSONEncoder().encode(workspace))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TokletDelete-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:directory) }
        let file = directory.appendingPathComponent("history.json")
        let harness = HarnessStore(enabled:false)
        let chat = FriendlyChatStore(harness:harness,fileURL:file)
        chat.edit("private draft"); chat.newChat(); chat.edit("another draft"); chat.workspace.outputLimit = 256
        harness.state.busy = true; chat.removeAll()
        try require(chat.workspace.conversations.count == 2,"delete all is blocked during active work")
        harness.state.busy = false; chat.removeAll()
        let reopened = try ChatWorkspace.restore(Data(contentsOf:file))
        try require(reopened.conversations.count == 1 && reopened.current.draft.isEmpty && reopened.outputLimit == 256,"delete all persists without a stale draft-save restoring content")
        chat.startBenchmark(); try require(!chat.benchmarkRunning,"benchmark requires an admitted chat model")
        let failureFile = directory.appendingPathComponent("failure-history.json")
        var failWrites = false
        let failure = FriendlyChatStore(harness:harness,fileURL:failureFile,persist:{ data,url in
            if failWrites { throw HubFailure("Injected full-disk failure") }
            try data.write(to:url,options:.atomic)
        })
        failure.edit("first draft"); failure.newChat(); failure.edit("second draft"); try require(failure.save(),"write-failure fixture saved")
        let saved = try Data(contentsOf:failureFile)
        let selected = failure.workspace.selected
        failWrites = true; failure.remove(selected)
        try require(failure.workspace.conversations.count == 2 && failure.workspace.selected == selected && !failure.storageError.isEmpty,"failed single deletion restores selection and dialogs with visible error")
        let afterSingleFailure = try Data(contentsOf:failureFile)
        try require(afterSingleFailure == saved,"failed deletion leaves the old saved file intact")
        failure.removeAll()
        try require(failure.workspace.conversations.count == 2 && failure.workspace.current.draft == "second draft" && failure.notice.contains("were kept"),"failed delete-all restores all drafts and dialogs")
        failWrites = false; failure.removeAll()
        let afterRetry = try ChatWorkspace.restore(Data(contentsOf:failureFile))
        try require(afterRetry.conversations.count == 1 && afterRetry.current.draft.isEmpty && failure.storageError.isEmpty,"delete-all retry persists after storage recovers")
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
    static func discovery() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletDiscovery-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let rows = Data(("[" + (HubProfile.catalog.map { "{\"id\":\"" + $0.repository + "\"}" } + ["{\"id\":\"mlx-community/gemma-4-e2b-it-4bit\"}"]).joined(separator:",") + "]").utf8)
        let allowance = HubAdmission(unifiedGPU:true,physicalMemory:6 << 30,availableMemory:2 << 30)
        actor Requests {
            var urls: [URL] = []
            func record(_ url: URL) -> Int { urls.append(url); return urls.count }
            func count() -> Int { urls.count }
        }
        let requests = Requests(), gate = VerificationGate()
        let service = ModelHub(root:root,fetch:{ url,file,limit,_ in
            if await requests.record(url) == 1 { await gate.pause() }
            guard limit == HubAPI.metadataLimit, url.host == "huggingface.co", url.path == "/api/models" else { throw HubFailure("Unbounded or non-HF discovery") }
            try rows.write(to:file)
        })
        let store = HubStore(hub:service,admission:{ allowance })
        store.findBest(); await gate.waitForStart()
        try require(store.findingBest && store.discoveryMessage.contains("Searching Hugging Face"),"first tap starts visible live discovery")
        store.inventory(); store.findBest(task:.fast)
        await gate.release(); await store.waitForIdle()
        let requestCount = await requests.count()
        try require(requestCount == 4 && !store.findingBest && store.discoveryMessage.hasPrefix("Hugging Face: 4"),"sheet inventory and preference changes cannot cancel discovery")
        try require(store.discoveredModels.count == 4 && store.discoveredModels.last?.profile == nil,"online results are deduplicated; Gemma discovery never grants compatibility")
        try require(store.recommendations.first?.profile == .smol,"latest preference is used when online search completes")
        try require(HubAPI.discoveryURLs.count == 4 && HubAPI.discoveryURLs.allSatisfy { URLComponents(url:$0,resolvingAgainstBaseURL:false)?.queryItems?.contains(URLQueryItem(name:"limit",value:"20")) == true },"discovery requests are bounded to four twenty-row pages")
        let noMemory = HubStore(hub:service,admission:{ HubAdmission(unifiedGPU:true,physicalMemory:6 << 30,availableMemory:0) })
        noMemory.findBest(); await noMemory.waitForIdle()
        try require(noMemory.recommendations.isEmpty && noMemory.discoveredModels.count == 4 && noMemory.recommendationMessage.contains("memory"),"low memory still shows online discoveries with no false fit")
        let offline = HubStore(hub:ModelHub(root:root,fetch:{ _,_,_,_ in throw HubFailure("Injected offline") }),admission:{ allowance })
        offline.findBest(); await offline.waitForIdle()
        try require(offline.discoveryMessage.contains("failed") && offline.discoveryMessage.contains("offline catalog") && offline.recommendations.count == 2,"failed network keeps an explicit offline fallback")
        let partial = HubStore(hub:ModelHub(root:root,fetch:{ url,file,_,_ in
            if URLComponents(url:url,resolvingAgainstBaseURL:false)?.queryItems?.contains(URLQueryItem(name:"sort",value:"downloads")) == true { throw HubFailure("Injected one failed page") }
            try rows.write(to:file)
        }),admission:{ allowance })
        partial.findBest(); await partial.waitForIdle()
        try require(partial.discoveredModels.count == 4 && partial.discoveryMessage.contains("Some searches failed"),"partial failure retains valid online results with retry status")
        let cleanupRequests = Requests()
        let cleanup = HubStore(hub:ModelHub(root:root,fetch:{ url,file,_,_ in
            _ = await cleanupRequests.record(url)
            if URLComponents(url:url,resolvingAgainstBaseURL:false)?.queryItems?.contains(URLQueryItem(name:"sort",value:"downloads")) == true {
                throw HubTransfer.metadataCleanupFailure(HubFailure("Injected page failure"),"Injected sink close failure")
            }
            try rows.write(to:file)
        }),admission:{ allowance })
        cleanup.findBest(); await cleanup.waitForIdle()
        let joined = await cleanupRequests.count()
        try require(joined == 4 && cleanup.discoveredModels.isEmpty && cleanup.discoveryMessage.contains("Injected sink close failure"),"a partial-page cleanup failure joins siblings and cannot become successful discovery")
        let removal = HubStore(hub:ModelHub(root:root,fetch:{ _,file,_,_ in try rows.write(to:file) },removeMetadata:{ file in
            if FileManager.default.fileExists(atPath:file.path) { try FileManager.default.removeItem(at:file) }
            throw HubFailure("Injected metadata removal failure")
        }),admission:{ allowance })
        removal.findBest(); await removal.waitForIdle()
        try require(removal.discoveredModels.isEmpty && removal.message.components(separatedBy:"Injected metadata removal failure").count == 5,"all four temporary-removal failures survive draining siblings")
        let closeGate = VerificationGate(), closeRequests = Requests()
        let cancelledCleanup = HubStore(hub:ModelHub(root:root,fetch:{ url,file,_,_ in
            if await closeRequests.record(url) == 1 {
                await closeGate.pause()
                throw HubTransfer.metadataCleanupFailure(CancellationError(),"Injected cancellation cleanup failure")
            }
            try rows.write(to:file)
        }),admission:{ allowance })
        cancelledCleanup.findBest(); await closeGate.waitForStart(); cancelledCleanup.cancel()
        await closeGate.release(); await cancelledCleanup.waitForIdle()
        try require(cancelledCleanup.discoveredModels.isEmpty && cancelledCleanup.discoveryMessage.contains("Injected cancellation cleanup failure"),"cancellation does not replace a sibling cleanup cause")
        let cancelledGate = VerificationGate(), cancelRequests = Requests()
        let cancelled = HubStore(hub:ModelHub(root:root,fetch:{ url,file,_,_ in
            if await cancelRequests.record(url) == 1 { await cancelledGate.pause() }
            try rows.write(to:file)
        }),admission:{ allowance })
        cancelled.findBest(); await cancelledGate.waitForStart(); cancelled.cancel()
        await cancelledGate.release(); await cancelled.waitForIdle()
        try require(cancelled.discoveredModels.isEmpty && !cancelled.findingBest && cancelled.discoveryMessage.contains("stopped"),"cancelled discovery cannot publish stale rows")
        let files = try FileManager.default.contentsOfDirectory(atPath:root.path)
        try require(files.isEmpty,"metadata temporary files are removed on success, failure and cancellation")
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
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let ownerHistory = try encoder.encode(chat.workspace)
        chat.startBenchmark()
        try require(chat.benchmarkRunning && !chat.ready,"benchmark owns serial worker and blocks Chat controls")
        for _ in 0..<4 {
            guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Local benchmark timed out") }
            harness.poll(); chat.observe(harness.state)
        }
        try require(!chat.benchmarkRunning && chat.benchmarkSamples.count == 3 && chat.benchmarkStatus.hasPrefix("Complete"),"warm-up plus three real runs complete")
        try require(chat.benchmarkSamples.allSatisfy { $0.metrics.tokens > 0 && $0.metrics.firstTokenSeconds != nil && $0.metrics.totalSeconds != nil },"real benchmark reports actual timing and token counts")
        let afterBenchmark = try encoder.encode(chat.workspace)
        try require(afterBenchmark == ownerHistory,"benchmark never modifies owner conversations/drafts/preferences")
        chat.startBenchmark(); chat.stopBenchmark()
        guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Benchmark stop timed out") }
        harness.poll(); chat.observe(harness.state)
        try require(!chat.benchmarkRunning && chat.benchmarkSamples.isEmpty && chat.benchmarkStatus.contains("stopped"),"cancelled warm-up cannot publish a benchmark sample")
        let afterStop = try encoder.encode(chat.workspace)
        try require(afterStop == ownerHistory,"stopping a benchmark preserves dialog bytes")
        chat.startBenchmark()
        guard worker.waitUntilIdle(seconds:120) != nil else { throw HubFailure("Benchmark warm-up timed out") }
        worker.interrupt(unload:false)
        guard worker.waitUntilIdle(seconds:30) != nil else { throw HubFailure("Benchmark lifecycle cleanup timed out") }
        harness.poll(); chat.observe(harness.state)
        try require(!chat.benchmarkRunning && chat.benchmarkSamples.isEmpty && !worker.snapshot().busy && !worker.snapshot().hasSession,"background cleanup between runs cannot automatically restart the benchmark")
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
            let store = HarnessStore(enabled:false); store.poll()
            store.state.requestStartedUptimeNanoseconds = 1_000_000_000
            store.state.tokenTimings = [
                NativeTokenTiming(index:0,nativeElapsedSeconds:0.1,callbackUptimeNanoseconds:1_250_000_000),
                NativeTokenTiming(index:1,nativeElapsedSeconds:0.15,callbackUptimeNanoseconds:1_300_000_000)]
            let monitor = VStack(alignment:.leading,spacing:12) {
                Text("Monitoring preview · Mac RAM sample").font(.headline)
                Text("Illustrative generation timing").font(.caption).foregroundStyle(.secondary)
                RuntimeMonitorView(store:store)
            }.padding(20).frame(width:393).background(scheme == .light ? Color.white : Color.black)
                .environment(\.colorScheme,scheme)
            let monitorRenderer = ImageRenderer(content:monitor); monitorRenderer.scale = 2
            guard let monitorImage = monitorRenderer.nsImage, let monitorTIFF = monitorImage.tiffRepresentation,
                  let monitorBytes = NSBitmapImageRep(data:monitorTIFF)?.representation(using:.png,properties:[:]) else { throw HubFailure("Could not render monitor preview") }
            try monitorBytes.write(to:folder.appendingPathComponent("monitor-" + name + ".png"))
        }
        print("TOKLET_MOBILE_HIGHLIGHT_PREVIEW_EXPORTED")
    }
    static func main() async throws {
        try content(); try workspace(); try await dashboard(); try await hub(); try await discovery()
        if CommandLine.arguments.count == 2 && CommandLine.arguments[1] == "--discovery" {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletLiveDiscovery-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at:root) }
            let store = HubStore(hub:ModelHub(root:root))
            store.findBest(); await store.waitForIdle()
            try require(store.discoveryMessage.hasPrefix("Hugging Face:") && !store.discoveredModels.isEmpty,"live Hugging Face discovery returns models")
            let result: [String:Any] = ["status":store.discoveryMessage,"models":store.discoveredModels.map(\.id),"ranked":store.recommendations.map { $0.profile.repository }]
            print("TOKLET_LIVE_DISCOVERY_JSON=" + String(decoding:try JSONSerialization.data(withJSONObject:result,options:.sortedKeys),as:UTF8.self))
        }
        else if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--switching" { try await switching(CommandLine.arguments[2],CommandLine.arguments[3]) }
        else if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--polling" { try polling(CommandLine.arguments[2]) }
        else if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--model" { try loading(CommandLine.arguments[2]) }
        else if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--render" { try render(CommandLine.arguments[2]) }
        else if CommandLine.arguments.count != 1 { throw HubFailure("Expected --model LOCAL_FOLDER or no arguments") }
        print("TOKLET_MOBILE_INTERFACE_PASS")
    }
}

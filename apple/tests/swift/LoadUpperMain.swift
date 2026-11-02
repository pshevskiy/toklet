import Foundation
import CToklet

// Cold policy controls plus an explicitly invoked cached-model integration.
@main
@MainActor
struct LoadUpperMain {
    static func check(_ condition: Bool, _ message: String) throws {
        if !condition { throw NativeContextFailure(message) }
    }
    static func encoded(_ workspace: ChatWorkspace) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(workspace)
    }
    static func controls() throws {
        var settings = NativeSettings(directory:"/never-opened")
        try check(try settings.loadOptions().capacity == settings.options().capacity,"Ordinary defaults changed")
        settings.loadIdentity = .managedQwen2; settings.metal = true; settings.mlx = true; settings.chat = true
        settings.capacity = 512; settings.loadUpper = 1024
        try check(try settings.loadOptions().capacity == 1024 && settings.options().capacity == 512,"Model/session limits conflated")
        func rejects(_ value: NativeSettings) throws {
            var rejected = false
            do { _ = try value.admittedLoadUpper() } catch { rejected = true }
            try check(rejected,"Invalid separate upper accepted")
        }
        for upper: UInt32 in [0,511,513,1025,32768,.max] {
            var value = settings; value.loadUpper = upper; try rejects(value)
        }
        for mode: NativeLoadIdentity in [.ordinary,.managedSmol,.managedQwen3,.managedMistral] {
            var value = settings; value.loadIdentity = mode; try rejects(value)
        }
        var value = settings; value.capacity = 1025; try rejects(value)
        value.capacity = 0; try rejects(value)
        value = settings; value.metal = false; try rejects(value)
        value = settings; value.mlx = false; try rejects(value)
        value = settings; value.chat = false; try rejects(value)
        value = settings; value.packedDirectory = "/packed"; try rejects(value)
        let harness = HarnessStore(enabled:false)
        let chat = FriendlyChatStore(harness:harness,fileURL:URL(fileURLWithPath:"/never-opened/load-upper-controls/chat.json"))
        let before = chat.workspace
        chat.setQwen2LoadUpper(1024)
        let request = FriendlyLoadRequest(folder:URL(fileURLWithPath:"/never-opened"),mode:.managedQwen2,
            conversation:before.selected,loadUpper:chat.qwen2LoadUpper)
        chat.setQwen2LoadUpper(512); chat.setQwen2LoadUpper(2048)
        try check(request.loadUpper == 1024 && chat.qwen2LoadUpper == 512,"Captured intent changed or invalid choice accepted")
        try check(try encoded(before) == encoded(chat.workspace),"Choice edited conversation")
        print("LOAD_UPPER_CONTROLS passed")
    }
    static func main() throws {
        try controls()
        guard CommandLine.arguments.count > 1 else { return }
        try check(CommandLine.arguments.count == 3,"Expected cached model and fresh output directory")
        let folder = URL(fileURLWithPath:CommandLine.arguments[1])
        let output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let preferences = FriendlyPreferenceStore(directory:output.appendingPathComponent("preferences"))
        let harness = HarnessStore(enabled:true)
        guard let worker = harness.worker else { throw NativeContextFailure("Missing worker") }
        let chat = FriendlyChatStore(harness:harness,fileURL:output.appendingPathComponent("chat.json"),preferences:preferences)
        var failure: Error?
        func wait() throws -> NativeSnapshot {
            guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Worker timeout") }
            try check(state.status == 0,"Worker failed: \(state.message)")
            harness.poll(); chat.observe(harness.state)
            return state
        }
        do {
            chat.setQwen2LoadUpper(1024)
            chat.openManaged(folder,profile:.qwen,ifSelected:chat.workspace.selected)
            let loaded = try wait()
            guard let identity = loaded.acceptedIdentity, let initial = loaded.sessionConfiguration,
                  let runtime = loaded.runtimePlan else { throw NativeContextFailure("Missing load facts") }
            try check(identity.loadedLimit == 1024 && initial.capacity == 512 && initial.chunk == 32,
                "Load upper did not remain separate from initial KV")
            try check(runtime.requestedLoadContext == 1024 && runtime.requestedCapacity == 512,
                "Runtime plan conflated model and session")
            chat.setQwen2LoadUpper(512)
            try check(worker.snapshot().sessionConfiguration == initial && worker.snapshot().acceptedIdentity == identity,
                "Editing next-load choice changed live owner")
            try check(chat.qwen2ReloadPending,"Active model lacks explicit reopen action")
            chat.setQwen2LoadUpper(2048)
            try check(chat.qwen2LoadUpper == 512 && worker.snapshot().sessionConfiguration == initial,
                "Invalid choice changed current owner or next-load limit")
            chat.showPlanning(); chat.editPlanning(.upper,"1024"); chat.previewPlanning()
            _ = try wait()
            try check(chat.planning.result?.capacity?.capacity.selected?.capacity == 1024
                && chat.planningApplyUnavailableReason == nil,"Growth preview unavailable")
            chat.applyPlanning()
            let grown = try wait()
            guard let configuration = grown.sessionConfiguration, let transition = grown.transition,
                  let plan = transition.plan, let candidate = transition.candidate else {
                throw NativeContextFailure("Missing growth evidence")
            }
            try check(configuration.capacity == 1024 && configuration.modelInstance == initial.modelInstance
                && configuration.sessionInstance != initial.sessionInstance && grown.acceptedIdentity == identity
                && grown.modelContext?.loadedLimit == 1024 && transition.committed
                && transition.headroom?.fits == true && plan.fitsTransitionBudget && plan.fitsCandidateIndividualBudget,
                "Growth changed model identity or failed checked admission")
            try check(plan.candidateConstructorRequiredBytes == candidate.numericalAdmission,
                "Candidate admission differs from plan")
            // Pinned Qwen2: 24 layers, two KV heads, dimension64, separate
            // F16 K/V arrays with seven fixed guard elements per array.
            func kvBytes(_ positions: UInt64) -> UInt64 { 24 * 2 * (2 * positions * 64 + 7) * 2 }
            try check(plan.currentAllocated.kvBytes == runtime.sessionMemory.kvBytes
                && plan.currentAllocated.kvBytes == kvBytes(512) && plan.candidatePlanned.kvBytes == kvBytes(1024),
                "Initial KV allocation did not remain at 512 positions")
            let record = try preferences.url(identity.namespace), saved = try Data(contentsOf:record)
            let workspace = try encoded(chat.workspace)
            let selection = NativeTransitionSelection(id:UUID(),modelInstance:configuration.modelInstance,
                sessionInstance:configuration.sessionInstance)
            let invalid = NativeTransitionRequest(id:UUID(),selection:selection,capacity:1025,chunk:32,
                wholeSessionBudget:configuration.wholeSessionBudget,coexistenceBudget:HubProfile.qwen.processAllowance,
                replayReserve:0,textReserve:HubProfile.qwen.textBudget,applicationReserve:NativeTransitionHeadroomCheck.applicationReserve,
                memoryPolicy:.friendly)
            worker.selectTransition(selection)
            try check(worker.submit(.replaceWithEmptySession(invalid)),"Rejection request not submitted")
            guard let rejected = worker.waitUntilIdle(seconds:30) else { throw NativeContextFailure("Rejection timeout") }
            harness.poll(); chat.observe(harness.state)
            try check(rejected.transition?.status != 0 && rejected.transition?.outcome == .rejected && rejected.transition?.committed == false
                && rejected.sessionConfiguration == configuration && rejected.acceptedIdentity == identity
                && rejected.history == grown.history && rejected.evaluated == grown.evaluated && rejected.pending == grown.pending
                && (try Data(contentsOf:record)) == saved && (try encoded(chat.workspace)) == workspace,
                "Beyond-upper rejection changed owner/history/preferences")
            // Exercise the real active-model shortcut and deferred unload/load.
            chat.openManaged(folder,profile:.qwen,ifSelected:chat.workspace.selected)
            var reopened = try wait()
            for _ in 0..<3 where reopened.acceptedIdentity == nil { reopened = try wait() }
            try check(reopened.acceptedIdentity?.modelInstance != identity.modelInstance
                && reopened.acceptedIdentity?.namespace == identity.namespace && reopened.acceptedIdentity?.loadedLimit == 512
                && reopened.sessionConfiguration?.capacity == 512 && reopened.sessionConfiguration?.chunk == 32
                && !chat.qwen2ReloadPending && !chat.loadingModel && !chat.restoringPreference
                && (try Data(contentsOf:record)) == saved && (try encoded(chat.workspace)) == workspace,
                "Deferred reopen changed captured upper, conversation or incompatible saved record")
            print("LOAD_UPPER_MAC passed model=\(identity.modelInstance) initial=\(initial.capacity) grown=\(configuration.capacity) loaded=\(identity.loadedLimit) initial_kv=\(plan.currentAllocated.kvBytes) grown_kv=\(plan.candidatePlanned.kvBytes) aggregate=\(plan.aggregateRequiredBytes) candidate_admission=\(candidate.numericalAdmission)")
        } catch { failure = error }
        worker.close()
        try check(worker.waitUntilClosed(seconds:60),"Worker close timeout")
        if let failure { throw failure }
    }
}

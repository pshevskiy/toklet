import Foundation
import CToklet

struct NativeSettings: Sendable {
    static let maximumChatOutputTokens: UInt32 = TOKLET_MAX_CHAT_OUTPUT_TOKENS
    static let chatContextPositions: UInt32 = 512
    var directory: String
    var loadIdentity: NativeLoadIdentity = .ordinary
    var loadRequestID: UUID? // Correlation only; never content identity.
    var packedDirectory = ""
    var metal = false
    var mlx = false
    var chat = false
    var measureLoadingMemory = false
    var textBudget: UInt64 = 64 * 1024 * 1024
    var capacity: UInt32 = 32
    // Cold, immutable model limit; capacity remains the current session size.
    // Explicitly supported only for managed Qwen2 in this bounded integration.
    var loadUpper: UInt32? = nil
    var chunk: UInt32 = 3
    var numericalBudget: UInt64 = 128 * 1024 * 1024

    var planningAttention: NativeCapacityPlan.AttentionPolicy {
        loadIdentity == .managedQwen15 ? .reference : .nativeFactory
    }

    func admittedLoadUpper() throws -> UInt32 {
        guard let loadUpper else { return capacity }
        guard loadIdentity == .managedQwen2, metal, mlx, chat, packedDirectory.isEmpty,
              (loadUpper == 512 || loadUpper == 1024), capacity > 0, capacity <= loadUpper else {
            throw NativeContextFailure("The separate model limit requires managed Qwen2, 512 or 1024 positions, and an initial window within that limit.")
        }
        return loadUpper
    }

    func loadOptions() throws -> TokletOptions {
        var result = options()
        result.capacity = try admittedLoadUpper()
        return result
    }

    func options() -> TokletOptions {
        TokletOptions(abi_version: TOKLET_ABI_VERSION,
                      struct_size: UInt32(MemoryLayout<TokletOptions>.size),
                      capacity: capacity, chunk: chunk, memory_limit_bytes: numericalBudget)
    }
}

enum NativeCommand: Sendable {
    case preview(NativePreviewRequest)
    // Explicit adoption: successful acceptance resets only native session cache.
    case replaceWithEmptySession(NativeTransitionRequest)
    case load(NativeSettings)
    case chat(id: UUID, conversation: Data, maximum: UInt32, autoCancelAfter: Int = 0)
    // Raw IDs only. autoCancelAfter is a deterministic diagnostic interruption.
    case generate(operation: UInt32, tokens: [UInt32], maximum: UInt32,
                  stops: [UInt32], autoCancelAfter: Int = 0)
    case score(target: UInt32)
    case reset, suspend, replay, unload
}

// The native elapsed clock starts inside generation; callback uptime is the
// monotonic process clock observed at the Swift callback. Neither uses UI polls.
struct NativeTokenTiming: Codable, Sendable {
    let index: UInt32
    let nativeElapsedSeconds: Double
    let callbackUptimeNanoseconds: UInt64
}

// Scalar diagnostic only; the owning engine keeps all full logits resident.
struct NativeDistributionScore: Codable, Equatable, Sendable {
    let target: UInt32
    let greedy: UInt32
    let vocabulary: UInt32
    let evaluated: UInt32
    let targetLogit: Double
    let greedyLogit: Double
    let negativeLogLikelihood: Double
}

struct NativeSnapshot: Sendable {
    var revision: UInt64 = 0
    var busy = false
    var closed = false
    var hasModel = false
    var hasSession = false
    var phase = "Idle"
    var status: UInt32 = 0
    var message = ""
    var ids: [UInt32] = [] // At most maximumChatOutputTokens; raw diagnostics stay <=128.
    var tokenTimings: [NativeTokenTiming] = [] // Same bounded emitted-token order.
    // Submit-to-callback latency includes queueing and Chat tokenization.
    var requestStartedUptimeNanoseconds: UInt64?
    var requestCompletedUptimeNanoseconds: UInt64?
    var reason: UInt32 = 0
    // Authoritative C report count for this command, zero if generation never ran.
    var emitted: UInt32 = 0
    // Meaningful only for reason 2; hidden EOS remains in sampled history.
    var stopToken: UInt32 = .max
    var history: UInt32 = 0
    var evaluated: UInt32 = 0
    var pending: UInt32 = 0
    var savedHistory = 0
    // Opt-in cold diagnostic only, published after a completed command.
    var savedPrefix: [UInt32]?
    var admittedBytes: UInt64 = 0
    var score: NativeDistributionScore?
    // Last attempted MLX artifact, never a runtime/text qualification label.
    var compatibility: NativeCompatibility?
    var runtimePlan: NativeResourcePlan?
    // Independent historical preview; does not replace the last load plan.
    var preview: NativePreviewResult?
    var transition: NativeTransitionResult?
    var restartRequired = false
    var compatibilityDirectory = ""
    var workerFinished = false
    var modelContext: NativeModelContext?
    var liveContext: NativeContextFacts?
    var sessionConfiguration: NativeSessionConfiguration?
    var loadMemory: NativeLoadMemorySnapshot?
    var sessionMemory: NativeSessionMemorySnapshot?
    var acceptedIdentity: NativeAcceptedIdentity?
    var acceptedLoadRequestID: UUID?
    var requestBudget: NativeRequestBudget?
    var chatEnabled = false
    // Retained across queued lifecycle cleanup until the next text request.
    var chatResult: NativeChatResult?
}

struct NativeChatResult: Sendable {
    let id: UUID
    let text: String
    let reason: UInt32
    let error: String
    // Only known-unstarted turns (validation/setup rejection or queued lifecycle
    // cancellation) roll back. Once generation starts, keep its prefix.
    var rejected = false
}

private func nativeError(_ code: UInt32, _ error: inout TokletError) -> NSError {
    let text = withUnsafeBytes(of: &error.message) { bytes in
        String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
    return NSError(domain: "TokletNative", code: Int(code),
                   userInfo: [NSLocalizedDescriptionKey: text.isEmpty ? "Native status \(code)" : text])
}
private func requireOK(_ code: UInt32, _ error: inout TokletError) throws {
    if code != TOKLET_OK { throw nativeError(code, &error) }
}
private func invalid(_ text: String) -> NSError {
    NSError(domain: "TokletHarness", code: Int(TOKLET_INVALID),
            userInfo: [NSLocalizedDescriptionKey: text])
}

// @unchecked Sendable is confined to this synchronization boundary. Condition
// protects every shared field, including the independent atomic cancel handle's
// lifetime. Model/session pointers live only in Engine on the dedicated Thread.
// Call close() explicitly. waitUntilClosed() is for non-UI checks, never the UI.
final class NativeWorker: @unchecked Sendable {
    private let condition = NSCondition()
    private var queued: NativeCommand?
    private var closing = false
    private var previewSelection: NativePreviewSelection?
    private var previewRequest: NativePreviewRequest?
    private var previewPriorPhase = "Idle"
    private var transitionSelection: NativeTransitionSelection?
    private var transitionRequest: NativeTransitionRequest?
    private var transitionCancelled = false
    private var snapshotRevision: UInt64 = 0
    // All view mutations hold condition. Avoid retaining oldValue here: it
    // would force callback arrays to copy on every append. Observer self-writes
    // do not recurse; value-type member mutations invoke the observer too.
    private var view = NativeSnapshot() {
        didSet { snapshotRevision &+= 1; view.revision = snapshotRevision }
    }
    private var cancelFlag: OpaquePointer?
    private var thread: Thread?
    // Cold diagnostic completion observer for deterministic non-UI checks.
    // Runs after FFI returns, outside the condition lock, around publication;
    // the after observer receives only whether a score was published. Neither
    // receives native owners. Production callers leave both nil.
    private let captureSavedPrefix: Bool
    private let beforeScorePublication: (@Sendable () -> Void)?
    private let afterScorePublication: (@Sendable (Bool) -> Void)?
    // Cold diagnostic gate, outside the condition lock; ordinary callers nil.
    private let beforeCommandDequeue: (@Sendable (NativeCommand) -> Void)?
    // Cold test gate after C inspection ownership was released, outside the lock.
    private let afterRuntimeInspection: (@Sendable (NativeResourcePlan) -> Void)?

    private let afterPreviewLimits: (@Sendable (NativePreviewRequest, NativeContextLimits) -> Void)?
    private let beforePreviewPublication: (@Sendable (NativePreviewResult) -> Void)?

    private let beforeTransitionAcceptance: (@Sendable (NativeTransitionRequest, NativeContextFacts) -> Void)?
    private let beforeTransitionRejection: (@Sendable (NativeTransitionRequest, UInt32) -> Void)?
    private let afterTransitionAcceptance: (@Sendable (NativeTransitionResult) -> Void)?
    private let transitionRetirementFault: (@Sendable () -> NativeTransitionRetirementFault?)?
    private let readTransitionHeadroom: @Sendable () throws -> (availableBytes: UInt64, instrument: String)

    init(captureSavedPrefix: Bool = false, beforeScorePublication: (@Sendable () -> Void)? = nil,
         afterScorePublication: (@Sendable (Bool) -> Void)? = nil,
         beforeCommandDequeue: (@Sendable (NativeCommand) -> Void)? = nil,
         afterRuntimeInspection: (@Sendable (NativeResourcePlan) -> Void)? = nil,
         afterPreviewLimits: (@Sendable (NativePreviewRequest, NativeContextLimits) -> Void)? = nil,
         beforePreviewPublication: (@Sendable (NativePreviewResult) -> Void)? = nil,
         beforeTransitionAcceptance: (@Sendable (NativeTransitionRequest, NativeContextFacts) -> Void)? = nil,
         beforeTransitionRejection: (@Sendable (NativeTransitionRequest, UInt32) -> Void)? = nil,
         afterTransitionAcceptance: (@Sendable (NativeTransitionResult) -> Void)? = nil,
         transitionRetirementFault: (@Sendable () -> NativeTransitionRetirementFault?)? = nil,
         readTransitionHeadroom: @escaping @Sendable () throws -> (availableBytes: UInt64, instrument: String) = { try NativeFootprint.readHeadroom() }) throws {
        self.captureSavedPrefix = captureSavedPrefix
        self.beforeScorePublication = beforeScorePublication
        self.afterScorePublication = afterScorePublication
        self.beforeCommandDequeue = beforeCommandDequeue
        self.afterRuntimeInspection = afterRuntimeInspection
        self.afterPreviewLimits = afterPreviewLimits
        self.beforePreviewPublication = beforePreviewPublication
        self.beforeTransitionAcceptance = beforeTransitionAcceptance
        self.beforeTransitionRejection = beforeTransitionRejection
        self.afterTransitionAcceptance = afterTransitionAcceptance
        self.transitionRetirementFault = transitionRetirementFault
        self.readTransitionHeadroom = readTransitionHeadroom
        var error = TokletError()
        try requireOK(toklet_cancel_create(&cancelFlag, &error), &error)
        view.ids.reserveCapacity(Int(NativeSettings.maximumChatOutputTokens))
        view.tokenTimings.reserveCapacity(Int(NativeSettings.maximumChatOutputTokens))
        let thread = Thread { [weak self] in self?.run() }
        thread.name = "Toklet inference"
        self.thread = thread
        thread.start()
    }

    func snapshot() -> NativeSnapshot {
        condition.lock(); defer { condition.unlock() }
        return view
    }

    @discardableResult
    func submit(_ command: NativeCommand, initiallyCancelled: Bool = false) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard !closing, !view.closed, !view.busy, !view.restartRequired, let flag = cancelFlag else { return false }
        if case .replaceWithEmptySession(let request) = command {
            guard transitionSelection == request.selection,
                  view.liveContext?.modelInstance == request.selection.modelInstance,
                  view.liveContext?.sessionInstance == request.selection.sessionInstance else { return false }
            do { try request.validate() } catch { return false }
            transitionRequest = request; transitionCancelled = initiallyCancelled; view.transition = nil
            _ = toklet_cancel_set(flag,initiallyCancelled ? 1 : 0,nil)
            queued = command; view.busy = true
            condition.signal()
            return true
        }
        if case .preview(let request) = command {
            guard !initiallyCancelled, previewSelection == request.selection, request.matchesResident(view) else { return false }
            do { try request.validate() } catch { return false }
            previewRequest = request; view.preview = nil; previewPriorPhase = view.phase
            queued = command; view.busy = true; view.phase = "Planning preview"
            condition.signal()
            return true
        }
        invalidatePreview()
        invalidateTransition()
        // Clear only while idle, under the same lock as cancel/enqueue. A cancel
        // arriving after this point survives until the request observes it.
        _ = toklet_cancel_set(flag, initiallyCancelled ? 1 : 0, nil)
        // A previous command's failure is not the status of this accepted work.
        view.status = 0; view.message = ""
        queued = command
        switch command {
        case .chat, .generate: view.requestStartedUptimeNanoseconds = DispatchTime.now().uptimeNanoseconds
        default: view.requestStartedUptimeNanoseconds = nil
        }
        view.requestCompletedUptimeNanoseconds = nil
        if case .chat = command { view.chatResult = nil }
        if case .load = command { view.acceptedIdentity = nil; view.acceptedLoadRequestID = nil; view.compatibility = nil; view.runtimePlan = nil; view.compatibilityDirectory = "" }
        if case .unload = command { view.acceptedIdentity = nil; view.acceptedLoadRequestID = nil; view.compatibility = nil; view.runtimePlan = nil; view.compatibilityDirectory = "" }
        if case .suspend = command { view.compatibility = nil; view.runtimePlan = nil; view.compatibilityDirectory = "" }
        switch command {
        case .load, .unload, .suspend:
            view.modelContext = nil; view.liveContext = nil; view.sessionConfiguration = nil; view.sessionMemory = nil; view.loadMemory = nil
        case .chat, .generate, .reset, .replay:
            view.liveContext = nil
        default: break
        }
        view.requestBudget = nil
        view.busy = true
        view.phase = "Queued"
        view.score = nil
        view.savedPrefix = nil
        view.stopToken = .max
        view.emitted = 0
        view.ids.removeAll(keepingCapacity: true)
        view.tokenTimings.removeAll(keepingCapacity: true)
        condition.signal()
        return true
    }

    // Selection changes can invalidate a synchronous inspection while it runs.
    // They never mutate/cancel Engine. A fresh UUID for the same path is a new
    // selection; UUID/path equality is not payload authentication.
    @discardableResult
    func selectPreview(_ selection: NativePreviewSelection?) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard !closing, !view.closed else { return false }
        if let selection {
            guard !selection.directory.isEmpty, selection.directory.utf8.count <= 16384,
                  !selection.directory.utf8.contains(0) else { return false }
        }
        previewSelection = selection
        invalidatePreview()
        return true
    }

    // Caller holds condition. Suppress both late success and late error.
    private func invalidatePreview() {
        previewRequest = nil; view.preview = nil
    }

    private func previewCurrent(_ request: NativePreviewRequest) -> Bool {
        !closing && queued == nil && previewRequest == request && previewSelection == request.selection
            && request.matchesResident(view)
    }

    private func previewStillCurrent(_ request: NativePreviewRequest) -> Bool {
        condition.lock(); defer { condition.unlock() }
        return previewCurrent(request)
    }

    // Entirely separate from Engine.perform and its ordinary state publisher.
    // All C handles are copied/released on this same thread by the inspectors.
    private func performPreview(_ request: NativePreviewRequest) {
        var result: NativePreviewResult?
        var limits: NativeContextLimits?
        var derived: NativeCapacityPlan.Request?
        if previewStillCurrent(request) {
            do { try autoreleasepool {
                try request.validate()
                let metadata = try NativeContextLimits.inspect(directory:request.selection.directory,profile:request.selection.profile)
                limits = metadata
                afterPreviewLimits?(request,metadata) // after C handle release; no lock
                guard previewStillCurrent(request) else { return }
                if request.selection.profile == .smolAudited && request.estimator == .sourceProfile {
                    result = NativePreviewResult(request:request,outcome:.metadataOnly,limits:metadata,
                        capacityRequest:nil,capacity:nil,limitation:.auditedProfileHasNoCapacityAPI,status:0,message:"")
                } else if let upper = metadata.planningUpper {
                    let planRequest = try request.capacityRequest(planningUpper:upper)
                    derived = planRequest
                    guard previewStillCurrent(request) else { return }
                    let plan = try NativeCapacityPlan.inspect(directory:request.selection.directory,request:planRequest)
                    if request.estimator == .residentAdvisory {
                        guard plan.resources.attention == .reference else { throw NativeContextFailure("Advisory estimator mode differs from the resident factory") }
                    }
                    result = NativePreviewResult(request:request,outcome:.planned,limits:metadata,
                        capacityRequest:planRequest,capacity:plan,limitation:nil,status:0,message:"")
                } else {
                    let limitation: NativePreviewResult.Limitation = metadata.status == .compatibleUntested
                        ? .backendUnavailable : .metadataUnsupported
                    result = NativePreviewResult(request:request,outcome:.metadataOnly,limits:metadata,
                        capacityRequest:nil,capacity:nil,limitation:limitation,status:0,message:"")
                }
            } } catch {
                let error = error as NSError
                result = NativePreviewResult(request:request,outcome:.failed,limits:limits,
                    capacityRequest:derived,capacity:nil,limitation:nil,status:UInt32(clamping:error.code),
                    message:String(error.localizedDescription.prefix(512)))
            }
        }
        if let result { beforePreviewPublication?(result) } // no handles or lock
        condition.lock()
        if previewCurrent(request) { view.preview = result }
        previewRequest = nil
        view.busy = closing || queued != nil
        view.phase = view.busy ? "Cleanup queued" : previewPriorPhase
        condition.broadcast(); condition.unlock()
    }

    // A model/session-bound selection, independent of path-based previews.
    // Changing it invalidates pending success AND error without touching Engine.
    @discardableResult
    func selectTransition(_ selection: NativeTransitionSelection?) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard !closing, !view.closed, !view.restartRequired else { return false }
        if let selection {
            guard selection.modelInstance > 0, selection.sessionInstance > 0,
                  selection.modelInstance != selection.sessionInstance else { return false }
        }
        transitionSelection = selection; invalidateTransition()
        return true
    }
    private func invalidateTransition() {
        transitionRequest = nil; transitionCancelled = true; view.transition = nil
    }
    private func transitionCurrent(_ request: NativeTransitionRequest) -> Bool {
        !closing && queued == nil && !view.restartRequired && !transitionCancelled
            && transitionRequest == request && transitionSelection == request.selection
            && view.liveContext?.modelInstance == request.selection.modelInstance
            && view.liveContext?.sessionInstance == request.selection.sessionInstance
    }

    // All candidate construction and validation runs offside, on Engine's thread.
    // No native calls, throwing work, allocation, or callbacks in the swap lock.
    private func performTransition(_ request: NativeTransitionRequest, engine: Engine, flag: OpaquePointer) {
        var prepared: PreparedTransition?
        var failure: NSError?
        var plan: NativeTransitionPlan?
        var headroom: NativeTransitionHeadroomCheck?
        condition.lock(); let start = transitionCurrent(request); condition.unlock()
        if start {
            do { try autoreleasepool {
                let value = try engine.prepareEmpty(request,flag:flag,captureSavedPrefix:captureSavedPrefix,
                    readHeadroom:readTransitionHeadroom,headroom:&headroom)
                prepared = value; plan = value.plan
            } } catch { failure = error as NSError }
        }
        if let prepared { beforeTransitionAcceptance?(request,prepared.context) }
        else if let failure { beforeTransitionRejection?(request,UInt32(clamping:failure.code)) }
        var retired: OpaquePointer?
        var result: NativeTransitionResult?
        let acceptedResult = prepared.map {
            NativeTransitionResult(request:request,outcome:.committed,committed:true,
                plan:$0.plan,candidate:$0.context,status:0,message:"",headroom:headroom)
        }
        let rejectedResult = NativeTransitionResult(request:request,outcome:.rejected,committed:false,
            plan:plan,candidate:nil,status:UInt32(clamping:failure?.code ?? Int(TOKLET_CANCELLED)),
            message:String((failure?.localizedDescription ?? "Transition cancelled").prefix(512)),headroom:headroom)
        condition.lock()
        if transitionCurrent(request), let value = prepared, let handle = value.handle {
            retired = engine.session
            engine.session = handle; value.handle = nil
            engine.settings = value.settings; engine.saved = value.saved; engine.savedCount = 0
            engine.report = value.report; engine.lastEmitted = 0
            engine.modelContext = value.modelContext; engine.liveContext = value.context
            engine.sessionConfiguration = value.configuration; engine.sessionMemory = value.memory
            engine.requestBudget = nil; engine.score = nil
            engine.admissionUpperBound = value.context.numericalAdmission
            engine.phase = "Ready; native cache reset"
            view.modelContext = value.modelContext; view.liveContext = value.context
            view.sessionConfiguration = value.configuration; view.sessionMemory = value.memory
            view.requestBudget = nil; view.score = nil; view.savedPrefix = value.emptyPrefix
            view.preview = nil
            view.savedHistory = 0; view.history = 0; view.evaluated = 0; view.pending = 0
            view.reason = 0; view.emitted = 0; view.stopToken = .max
            view.ids = value.emptyIDs; view.tokenTimings = value.emptyTimings
            view.requestStartedUptimeNanoseconds = nil; view.requestCompletedUptimeNanoseconds = nil
            view.admittedBytes = value.context.numericalAdmission; view.status = 0; view.message = ""
            view.phase = engine.phase
            result = acceptedResult
            view.transition = result; transitionRequest = nil; transitionSelection = nil
        } else if transitionCurrent(request) {
            result = rejectedResult
            view.transition = result
        }
        condition.unlock()
        // Cancellation after the acceptance point does not undo a committed swap.
        if let result, result.committed { afterTransitionAcceptance?(result) }
        var cleanup = engine.retirementFailure
        if let retired { cleanup = engine.retire(retired,fault:transitionRetirementFault?()) }
        if let pending = prepared?.handle {
            prepared?.handle = nil
            cleanup = engine.retire(pending,fault:nil)
        }
        let cleanupResult = cleanup.map {
            NativeTransitionResult(request:request,outcome:.cleanupFailed,committed:retired != nil,
                plan:plan,candidate:retired == nil ? nil : prepared?.context,
                status:UInt32(clamping:$0.code),message:$0.localizedDescription,headroom:headroom)
        }
        condition.lock()
        if let cleanupResult {
            view.restartRequired = true; view.transition = cleanupResult
            view.phase = "Session cleanup failed; restart required"
        }
        transitionRequest = nil
        view.busy = closing || queued != nil
        condition.broadcast(); condition.unlock()
    }

    // Non-UI diagnostics can wait for command/queued-cleanup completion without
    // retaining repeated snapshots or estimating completion from a polling tick.
    // NSCondition uses a wall-clock deadline; event timestamps stay monotonic.
    func waitUntilIdle(seconds: TimeInterval) -> NativeSnapshot? {
        guard seconds.isFinite, seconds >= 0, seconds <= 3600 else { return nil }
        condition.lock(); defer { condition.unlock() }
        let end = Date(timeIntervalSinceNow: seconds)
        while view.busy {
            if !condition.wait(until: end) { return view.busy ? nil : view }
        }
        return view
    }

    func cancel() {
        condition.lock(); defer { condition.unlock() }
        invalidatePreview()
        invalidateTransition()
        if let flag = cancelFlag { _ = toklet_cancel_set(flag, 1, nil) }
    }

    // Lifecycle cleanup supersedes a queued request and runs after active work
    // returns. Active native resources are never released concurrently.
    func interrupt(unload: Bool) {
        condition.lock(); defer { condition.unlock() }
        guard !closing, !view.closed else { return }
        invalidatePreview()
        invalidateTransition()
        if let flag = cancelFlag { _ = toklet_cancel_set(flag, 1, nil) }
        cancelQueuedChat()
        view.compatibility = nil; view.runtimePlan = nil; view.compatibilityDirectory = ""
        // Once a full unload is queued, a later background event cannot weaken it.
        if case .unload? = queued {} else { queued = unload ? .unload : .suspend }
        view.busy = true
        view.modelContext = nil; view.liveContext = nil; view.requestBudget = nil
        view.sessionConfiguration = nil; view.sessionMemory = nil; view.loadMemory = nil
        view.acceptedIdentity = nil; view.acceptedLoadRequestID = nil // Retained Engine facts republish only after safe completion.
        view.phase = "Cancelling for lifecycle cleanup"
        view.score = nil
        view.savedPrefix = nil
        view.stopToken = .max
        condition.signal()
    }

    func close() {
        condition.lock(); defer { condition.unlock() }
        guard !view.closed else { return }
        invalidatePreview()
        invalidateTransition()
        cancelQueuedChat()
        closing = true; queued = nil
        view.acceptedIdentity = nil; view.acceptedLoadRequestID = nil
        view.modelContext = nil; view.liveContext = nil; view.requestBudget = nil
        view.sessionConfiguration = nil; view.sessionMemory = nil; view.loadMemory = nil
        view.compatibility = nil; view.runtimePlan = nil; view.compatibilityDirectory = ""
        view.score = nil
        view.savedPrefix = nil
        view.stopToken = .max
        if let flag = cancelFlag { _ = toklet_cancel_set(flag, 1, nil) }
        view.busy = true
        condition.signal()
    }

    // Caller holds condition. An accepted request always receives a terminal
    // response, even when lifecycle cleanup supersedes it before dequeue.
    private func cancelQueuedChat() {
        if case .chat(let id,_,_,_)? = queued {
            view.chatResult = NativeChatResult(id:id,text:"",reason:3,error:"Response stopped before generation.",rejected:true)
        }
    }

    func waitUntilClosed(seconds: TimeInterval) -> Bool {
        condition.lock(); defer { condition.unlock() }
        let end = Date(timeIntervalSinceNow: seconds)
        while !view.closed {
            if !condition.wait(until: end) { return view.closed }
        }
        return true
    }

    fileprivate func checkLoadAfterRuntimeInspection(_ plan: NativeResourcePlan) throws {
        afterRuntimeInspection?(plan)
        condition.lock(); defer { condition.unlock() }
        guard !closing && queued == nil else { throw invalid("Load superseded by lifecycle cleanup") }
    }

    fileprivate func record(_ event: TokletTokenEvent, autoCancelAfter: Int) -> UInt32 {
        condition.lock(); defer { condition.unlock() }
        guard view.ids.count < Int(NativeSettings.maximumChatOutputTokens), event.index == UInt32(view.ids.count),
              event.elapsed_seconds.isFinite, event.elapsed_seconds >= 0,
              event.elapsed_seconds >= (view.tokenTimings.last?.nativeElapsedSeconds ?? 0) else { return 1 }
        let timing = NativeTokenTiming(index:event.index,nativeElapsedSeconds:event.elapsed_seconds,
                                       callbackUptimeNanoseconds:DispatchTime.now().uptimeNanoseconds)
        view.ids.append(event.token_id)
        view.tokenTimings.append(timing)
        if autoCancelAfter > 0, view.ids.count == autoCancelAfter, let flag = cancelFlag {
            _ = toklet_cancel_set(flag, 1, nil)
        }
        return 0
    }

    private func run() {
        precondition(!Thread.isMainThread)
        // This object's model/session fields never leave this OS thread.
        let engine = Engine()
        while true {
            condition.lock()
            while queued == nil && !closing { condition.wait() }
            if closing { condition.unlock(); break }
            if let beforeCommandDequeue, let pending = queued {
                condition.unlock(); beforeCommandDequeue(pending); condition.lock()
                if closing { condition.unlock(); break }
            }
            if view.restartRequired {
                closing = true; queued = nil; condition.unlock(); break
            }
            let command = queued!
            queued = nil
            if case .replaceWithEmptySession(let request) = command {
                let flag = cancelFlag!
                condition.unlock()
                performTransition(request,engine:engine,flag:flag)
                continue
            }
            if case .preview(let request) = command {
                condition.unlock()
                performPreview(request)
                continue
            }
            let flag = cancelFlag!
            view.phase = "Running"
            condition.unlock()
            var loadSettings: NativeSettings?
            if case .load(let requested) = command { loadSettings = requested }
            let memoryProbe = NativeLoadMemoryProbe.begin(enabled:loadSettings?.measureLoadingMemory == true)
            var failure: NSError?
            do { try autoreleasepool {
                try engine.perform(command, flag: flag, worker: self)
                try engine.refreshContext()
            } }
            catch {
                failure = error as NSError
                // Preserve the original command error while suppressing any
                // invalid live facts. No old session snapshot authorizes replay.
                try? engine.refreshContext()
            }
            if let requested = loadSettings {
                let completed = failure == nil && engine.model != nil && engine.session != nil
                engine.loadMemory = memoryProbe?.finish(loadRequestID:requested.loadRequestID,capacity:requested.capacity,
                    upper:requested.loadUpper ?? requested.capacity,completed:completed,
                    model:engine.modelContext?.modelInstance,session:engine.liveContext?.sessionInstance)
            }
            if case .score = command { beforeScorePublication?() }
            condition.lock()
            switch command {
            case .chat, .generate:
                view.requestCompletedUptimeNanoseconds = DispatchTime.now().uptimeNanoseconds
            default: break
            }
            view.hasModel = engine.model != nil
            view.hasSession = engine.session != nil
            view.modelContext = !closing && queued == nil ? engine.modelContext : nil
            view.liveContext = !closing && queued == nil ? engine.liveContext : nil
            view.sessionConfiguration = !closing && queued == nil ? engine.sessionConfiguration : nil
            view.loadMemory = !closing && queued == nil ? engine.loadMemory : nil
            view.sessionMemory = !closing && queued == nil && engine.sessionMemory?.matches(engine.liveContext,configuration:engine.sessionConfiguration) == true ? engine.sessionMemory : nil
            view.acceptedIdentity = !closing && queued == nil && engine.model != nil && engine.chat != nil
                ? engine.acceptedIdentity : nil
            view.acceptedLoadRequestID = !closing && queued == nil && engine.model != nil && engine.chat != nil
                ? engine.acceptedLoadRequestID : nil
            view.requestBudget = !closing && queued == nil ? engine.requestBudget : nil
            view.chatEnabled = engine.chat != nil
            if case .chat = command { view.chatResult = engine.chatResult }
            view.savedHistory = engine.savedCount
            // Only this owning thread reads the engine prefix. The copy is
            // bounded by admitted capacity and absent from ordinary callers.
            view.savedPrefix = captureSavedPrefix && !closing && queued == nil
                ? Array(engine.saved.prefix(engine.savedCount)) : nil
            view.history = engine.report.history_tokens
            view.evaluated = engine.report.evaluated_tokens
            view.pending = engine.report.pending_tokens
            view.reason = engine.report.reason
            view.emitted = engine.lastEmitted
            view.stopToken = failure == nil && !closing && queued == nil && engine.report.reason == 2
                ? engine.report.stop_token : .max
            view.admittedBytes = engine.admissionUpperBound
            view.compatibility = !closing && queued == nil ? engine.compatibility : nil
            view.compatibilityDirectory = view.compatibility == nil ? "" : engine.compatibilityDirectory
            view.runtimePlan = !closing && queued == nil ? engine.runtimePlan : nil
            view.score = failure == nil && !closing && queued == nil ? engine.score : nil
            view.status = UInt32(clamping: failure?.code ?? 0)
            view.message = String((failure?.localizedDescription ?? "").prefix(512))
            view.busy = closing || queued != nil
            view.phase = view.busy ? "Cleanup queued" : engine.phase
            let publishedScore = view.score != nil
            condition.broadcast()
            condition.unlock()
            if case .score = command { afterScorePublication?(publishedScore) }
        }
        var releaseError: NSError?
        do { try autoreleasepool { try engine.unload() } } catch { releaseError = error as NSError }
        condition.lock()
        // All native calls have returned; setters cannot overlap flag release.
        if let flag = cancelFlag {
            var error = TokletError()
            let code = toklet_cancel_release(flag, &error)
            if code != TOKLET_OK { releaseError = nativeError(code, &error) }
            cancelFlag = nil
        }
        view.hasModel = false; view.hasSession = false
        view.chatEnabled = false; view.acceptedIdentity = nil; view.acceptedLoadRequestID = nil
        view.modelContext = nil; view.liveContext = nil; view.requestBudget = nil
        view.sessionConfiguration = nil; view.sessionMemory = nil; view.loadMemory = nil
        view.compatibility = nil; view.runtimePlan = nil; view.compatibilityDirectory = ""
        // Closed means there is no active numerical prefix. Do not retain the
        // last command's counts when close supersedes an in-flight operation.
        view.history = 0; view.evaluated = 0; view.pending = 0; view.reason = 0; view.emitted = 0
        view.savedHistory = 0; view.busy = false
        view.closed = true; view.workerFinished = true
        view.score = nil
        view.savedPrefix = nil
        view.stopToken = .max
        view.phase = releaseError == nil ? "Closed" : "Release failed; restart required"
        if let error = releaseError {
            view.status = UInt32(clamping: error.code); view.message = error.localizedDescription
        } else {
            // Completed close describes the released owner, not the previous
            // request. Earlier operation errors remain in caller diagnostics.
            view.status = 0; view.message = ""; view.admittedBytes = 0
            view.ids.removeAll(keepingCapacity:true);view.tokenTimings.removeAll(keepingCapacity:true)
        }
        thread = nil
        condition.broadcast(); condition.unlock()
    }
}

private final class Sink {
    let worker: NativeWorker
    let autoCancelAfter: Int
    init(_ worker: NativeWorker, _ count: Int) { self.worker = worker; autoCancelAfter = count }
}
private let nativeSink: @convention(c) (UnsafeMutableRawPointer?, TokletTokenEvent) -> UInt32 = { context, event in
    guard let context else { return 1 }
    let sink = Unmanaged<Sink>.fromOpaque(context).takeUnretainedValue()
    return sink.worker.record(event, autoCancelAfter: sink.autoCancelAfter)
}

// Private thread-local native owners. All FFI callbacks are synchronous and the
// Sink remains strongly owned until generate returns. No actor/queue is used as
// a substitute for the C boundary's OS-thread affinity.
private final class PreparedTransition {
    var handle: OpaquePointer?
    let plan: NativeTransitionPlan
    let settings: NativeSettings
    let context: NativeContextFacts
    let configuration: NativeSessionConfiguration
    let modelContext: NativeModelContext
    let report: TokletReport
    let memory: NativeSessionMemorySnapshot
    let saved: [UInt32]
    let emptyPrefix: [UInt32]?
    var emptyIDs: [UInt32] = []
    var emptyTimings: [NativeTokenTiming] = []
    init(handle: OpaquePointer, plan: NativeTransitionPlan, settings: NativeSettings,
         context: NativeContextFacts, configuration: NativeSessionConfiguration,
         memory: NativeSessionMemorySnapshot, report: TokletReport, captureSavedPrefix: Bool) {
        self.handle = handle; self.plan = plan; self.settings = settings
        self.context = context; self.configuration = configuration
        modelContext = NativeModelContext(context); self.report = report; self.memory = memory
        saved = Array(repeating:0,count:Int(settings.capacity))
        emptyPrefix = captureSavedPrefix ? [] : nil
        emptyIDs.reserveCapacity(Int(NativeSettings.maximumChatOutputTokens))
        emptyTimings.reserveCapacity(Int(NativeSettings.maximumChatOutputTokens))
    }
}

private final class Engine {
    var model: OpaquePointer?
    var session: OpaquePointer?
    var settings: NativeSettings?
    var saved: [UInt32] = []
    var savedCount = 0
    var admissionUpperBound: UInt64 = 0
    var report = TokletReport()
    var lastEmitted: UInt32 = 0
    var phase = "Idle"
    var score: NativeDistributionScore?
    var chat: OpaquePointer?
    var chatStops: [UInt32] = []
    var chatResult: NativeChatResult?
    var compatibility: NativeCompatibility?
    var runtimePlan: NativeResourcePlan?
    var compatibilityDirectory = ""
    var modelContext: NativeModelContext?
    var liveContext: NativeContextFacts?
    var sessionConfiguration: NativeSessionConfiguration?
    var loadMemory: NativeLoadMemorySnapshot?
    var sessionMemory: NativeSessionMemorySnapshot?
    var acceptedIdentity: NativeAcceptedIdentity?
    var acceptedLoadRequestID: UUID?
    var requestBudget: NativeRequestBudget?
    // Only BUSY/WRONG_THREAD prove a failed release did not consume the pointer.
    // At most one retained owner; no automatic retry loop or reuse for inference.
    var retiredSession: OpaquePointer?
    var retirementFailure: NSError?

    func retire(_ pointer: OpaquePointer, fault: NativeTransitionRetirementFault?) -> NSError? {
        var error = TokletError()
        let code: UInt32
        if fault == .busy { code = TOKLET_BUSY }
        else {
            let actual = toklet_session_release(pointer,&error)
            code = actual == TOKLET_OK && fault == .consumedPanic ? TOKLET_PANIC : actual
        }
        guard code != TOKLET_OK else { return nil }
        if code == TOKLET_BUSY || code == TOKLET_WRONG_THREAD { retiredSession = pointer }
        let failure = nativeError(code,&error)
        retirementFailure = failure
        return failure
    }

    func prepareEmpty(_ request: NativeTransitionRequest, flag: OpaquePointer,
                      captureSavedPrefix: Bool,
                      readHeadroom: () throws -> (availableBytes: UInt64, instrument: String),
                      headroom: inout NativeTransitionHeadroomCheck?) throws -> PreparedTransition {
        try request.validate()
        guard let model, let session, let settings, let old = liveContext,
              old.modelInstance == request.selection.modelInstance,
              old.sessionInstance == request.selection.sessionInstance else { throw invalid("Stale session transition") }
        // Conservative caller allowance for the two Swift history buffers,
        // report decoding/retained values and transaction metadata. This is a
        // requirement, not measured Swift allocator or whole-process storage.
        let app = try NativeTransitionPlan.sum([1024 * 1024,UInt64(saved.count) * 4,UInt64(request.capacity) * 4])
        guard request.applicationReserve >= app,
              request.textReserve >= (chat == nil ? 0 : settings.textBudget) else {
            throw invalid("Transition reserves omit retained Swift/text owners")
        }
        let plan = try NativeTransitionPlan.inspect(model:model,session:session,request:request,
            current:old,chunk:settings.chunk,wholeBudget:settings.numericalBudget)
        guard plan.fitsTransitionBudget, plan.fitsCandidateIndividualBudget else { throw invalid("Transition memory budget does not fit") }
        try request.validateKV(bytes:plan.candidatePlanned.kvBytes)
        if request.memoryPolicy == .friendly {
            let required = try NativeTransitionHeadroomCheck.requiredIncrementalBytes(
                candidateOwned:plan.candidatePlanned.totalBytes,
                candidateAllowance:plan.reserves.candidateMetadataDriverBytes,
                bridge:plan.reserves.candidateBridgeBytes,json:plan.reserves.jsonSnapshotBytes)
            let observed = try readHeadroom()
            guard !observed.instrument.isEmpty, observed.instrument.utf8.count <= 512 else {
                throw invalid("Process-headroom observation is invalid")
            }
            let check = NativeTransitionHeadroomCheck(requiredBytes:required,
                availableBytes:observed.availableBytes,instrument:observed.instrument)
            headroom = check
            guard check.fits else {
                throw invalid("Replacement needs \(required) additional accounted bytes; process headroom is \(observed.availableBytes) bytes")
            }
        }
        var options = request.options(), accounting = request.accounting(), error = TokletError()
        var pointer: OpaquePointer?
        try requireOK(toklet_session_prepare_empty(model,session,&options,&accounting,flag,&pointer,&error),&error)
        guard let pointer else { throw invalid("Native preparation returned no candidate") }
        // Native ownership remains scoped here until validation and all Swift
        // staging finish; every throwing failure releases on this thread.
        do {
            var raw = TokletContextInfo(), report = TokletReport()
            try requireOK(toklet_session_context(pointer,&raw,&error),&error)
            let next = try NativeContextFacts(raw)
            try NativeTransitionPlan.validateCandidate(next,old:old,request:request)
            try requireOK(toklet_session_state(pointer,&report,&error),&error)
            guard report.abi_version == TOKLET_ABI_VERSION,
                  report.struct_size == UInt32(MemoryLayout<TokletReport>.size),
                  report.history_tokens == 0, report.evaluated_tokens == 0, report.pending_tokens == 0,
                  report.emitted_tokens == 0, report.reason == 0, report.stop_token == .max,
                  report.admitted_bytes == next.numericalAdmission else {
                throw invalid("Candidate is not an empty validated session")
            }
            var nextSettings = settings
            nextSettings.capacity = request.capacity; nextSettings.chunk = request.chunk
            nextSettings.numericalBudget = request.wholeSessionBudget
            let configuration = try NativeSessionConfiguration(context:next,capacity:nextSettings.capacity,
                chunk:nextSettings.chunk,wholeSessionBudget:nextSettings.numericalBudget,acceptedRequestID:request.id)
            let memory = NativeSessionMemorySnapshot.inspect(model:model,session:pointer,context:next,configuration:configuration)
            try request.validateCandidateKV(memory)
            return PreparedTransition(handle:pointer,plan:plan,settings:nextSettings,
                context:next,configuration:configuration,memory:memory,
                report:report,captureSavedPrefix:captureSavedPrefix)
        } catch {
            if let cleanup = retire(pointer,fault:nil) { throw cleanup }
            throw error
        }
    }

    func refreshContext() throws {
        let prior = sessionConfiguration
        liveContext = nil; sessionConfiguration = nil
        guard let session else { return }
        do {
            var value = TokletContextInfo(); var error = TokletError()
            try requireOK(toklet_session_context(session,&value,&error),&error)
            let fresh = try NativeContextFacts(value)
            let retained = NativeModelContext(fresh)
            guard let settings, fresh.capacity == UInt64(settings.capacity),
                  fresh.offset == 0, fresh.backend == (settings.metal ? TOKLET_CONTEXT_METAL : TOKLET_CONTEXT_CPU),
                  settings.loadUpper.map({ fresh.loadedLimit == UInt64($0) }) ?? true,
                  modelContext == nil || modelContext == retained else {
                throw invalid("Native context does not match the loaded model/settings")
            }
            let configuration = try NativeSessionConfiguration(context:fresh,capacity:settings.capacity,
                chunk:settings.chunk,wholeSessionBudget:settings.numericalBudget,acceptedRequestID:prior?.acceptedRequestID)
            guard prior == nil || prior == configuration else { throw invalid("Accepted session configuration changed") }
            guard acceptedIdentity == nil || acceptedIdentity?.matches(retained) == true else { throw invalid("Accepted identity does not match the retained model") }
            modelContext = retained; liveContext = fresh; sessionConfiguration = configuration
        } catch { sessionMemory = nil; modelContext = nil; acceptedIdentity = nil; acceptedLoadRequestID = nil; requestBudget = nil; throw error }
    }

    func decodeChat(_ ids: [UInt32]) throws -> String {
        guard let chat else { throw invalid("Choose a chat model first") }
        var bytes = [UInt8](repeating:0,count:NativeGenerationPolicy.maximumDecodedBytes)
        var count = 0; var error = TokletError()
        let code = ids.withUnsafeBufferPointer { ids in
            bytes.withUnsafeMutableBufferPointer { output in
                toklet_chat_decode(chat,ids.baseAddress,ids.count,output.baseAddress,output.count,&count,&error)
            }
        }
        try requireOK(code,&error)
        guard count <= bytes.count, let text = String(bytes:bytes.prefix(count),encoding:.utf8) else {
            throw invalid("Invalid text response")
        }
        return text
    }

    func generateChat(_ id: UUID, _ conversation: Data, _ maximum: UInt32, _ autoCancelAfter: Int,
                      _ flag: OpaquePointer, _ worker: NativeWorker) throws {
        chatResult = nil
        var generationStarted = false
        do {
            guard let chat, let modelContext, conversation.count <= NativeGenerationPolicy.maximumConversationBytes,
                  maximum > 0 else { throw invalid("Choose a chat model and a valid response limit") }
            let limits = NativeGenerationPolicy.limits(.chat,model:modelContext)
            try limits.validate(input:0,output:maximum)
            guard autoCancelAfter >= 0, UInt64(autoCancelAfter) <= limits.maximumOutput else { throw invalid("Invalid automatic cancellation count") }
            var ids = [UInt32](repeating:0,count:Int(limits.maximumPrompt))
            var count = 0; var error = TokletError()
            let code = conversation.withUnsafeBytes { input in
                ids.withUnsafeMutableBufferPointer { output in
                    toklet_chat_encode(chat,input.bindMemory(to:UInt8.self).baseAddress,input.count,
                        output.baseAddress,output.count,&count,&error)
                }
            }
            try requireOK(code,&error)
            guard count <= ids.count else { throw invalid("Encoded input exceeds current chat input limit") }
            if session == nil { try createSession() } // Explicit Send, never foreground auto-resume.
            var generationError: Error?
            do { try generate(3,Array(ids.prefix(count)),maximum,chatStops,autoCancelAfter,flag,worker,
                              chatOutput:true,didAdmit:{ generationStarted = true }) }
            catch { generationError = error }
            if !generationStarted, let generationError { throw generationError }
            let text = try decodeChat(worker.snapshot().ids)
            chatResult = NativeChatResult(id:id,text:text,reason:report.reason,
                error:generationError?.localizedDescription ?? "")
            if let generationError { throw generationError }
        } catch {
            if chatResult == nil { chatResult = NativeChatResult(id:id,text:"",reason:0,error:error.localizedDescription,rejected:!generationStarted) }
            throw error
        }
    }

    func releaseSession() throws {
        liveContext = nil; sessionConfiguration = nil; sessionMemory = nil; requestBudget = nil
        score = nil
        if let value = session {
            session = nil
            var error = TokletError()
            try requireOK(toklet_session_release(value, &error), &error)
        }
        report = TokletReport()
    }
    func unload() throws {
        loadMemory = nil
        acceptedIdentity = nil; acceptedLoadRequestID = nil
        var retiredError: NSError?
        if let retired = retiredSession {
            retiredSession = nil
            var error = TokletError()
            let code = toklet_session_release(retired,&error)
            if code != TOKLET_OK { retiredError = nativeError(code,&error) }
        }
        modelContext = nil; liveContext = nil; requestBudget = nil
        try releaseSession()
        if let value = chat {
            chat = nil; chatStops = []
            var error = TokletError()
            try requireOK(toklet_chat_release(value,&error),&error)
        }
        if let value = model {
            model = nil
            var error = TokletError()
            try requireOK(toklet_model_release(value, &error), &error)
        }
        compatibility = nil; runtimePlan = nil; compatibilityDirectory = ""
        saved = []; savedCount = 0; admissionUpperBound = 0; settings = nil; phase = "Unloaded; fresh prompt required"
        if let retiredError { throw retiredError }
    }
    func createSession() throws {
        guard let model, let settings else { throw invalid("Load a model first") }
        // Retained settings are resume policy, not a live accepted request.
        sessionConfiguration = nil; sessionMemory = nil
        var options = settings.options(); var error = TokletError()
        try requireOK(toklet_session_create(model, &options, &session, &error), &error)
        try requireOK(toklet_session_state(session, &report, &error), &error)
        admissionUpperBound = max(admissionUpperBound,report.admitted_bytes)
        try refreshContext()
        if let session, let context = liveContext, let configuration = sessionConfiguration {
            sessionMemory = NativeSessionMemorySnapshot.inspect(model:model,session:session,context:context,configuration:configuration)
        }
    }
    func refreshHistory() throws {
        guard let session else { return }
        var count = 0; var error = TokletError()
        let code = saved.withUnsafeMutableBufferPointer {
            toklet_session_history(session, $0.baseAddress, $0.count, &count, &error)
        }
        try requireOK(code, &error)
        savedCount = count
    }
    func generate(_ operation: UInt32, _ tokens: [UInt32], _ maximum: UInt32,
                  _ stops: [UInt32], _ autoCancel: Int, _ flag: OpaquePointer,
                  _ worker: NativeWorker, chatOutput: Bool = false, didAdmit: (() -> Void)? = nil) throws {
        guard let session else { throw invalid("Load or replay a session first") }
        try refreshContext()
        guard let modelContext, let liveContext else { throw invalid("Native context unavailable; reload the model") }
        let limits = NativeGenerationPolicy.limits(chatOutput ? .chat : .raw,model:modelContext)
        try limits.validate(input:tokens.count,output:maximum)
        guard stops.count <= Int(NativeGenerationPolicy.maximumRawPrompt),
              autoCancel >= 0, UInt64(autoCancel) <= limits.maximumOutput else { throw invalid("Request exceeds current mode limits") }
        var admission = TokletRequestAdmission(); var error = TokletError()
        let preflight = tokens.withUnsafeBufferPointer { input in
            stops.withUnsafeBufferPointer { stop in
                toklet_session_preflight(session,operation,input.baseAddress,input.count,maximum,
                    stop.baseAddress,stop.count,&admission,&error)
            }
        }
        if preflight == TOKLET_INVALID {
            let base = operation == 0 || operation == 3 ? 0 : liveContext.history
            let after = base.addingReportingOverflow(UInt64(tokens.count))
            let required = after.partialValue.addingReportingOverflow(UInt64(maximum))
            if !after.overflow, !required.overflow, required.partialValue > liveContext.capacity {
                throw invalid("This request needs \(required.partialValue.formatted()) tokens including its reply; the current shared window is \(liveContext.capacity.formatted()). Increase context, choose a shorter reply, or start a new chat. Even a new chat includes template tokens.")
            }
        }
        try requireOK(preflight,&error)
        let budget = try NativeRequestBudget(admission,expected:liveContext,operation:operation,input:tokens.count,output:maximum)
        try limits.validateHistory(tokens:budget.historyAfterInput,output:maximum)
        requestBudget = budget
        didAdmit?()
        let sink = Sink(worker, autoCancel)
        report = TokletReport()
        let code = withExtendedLifetime(sink) {
            tokens.withUnsafeBufferPointer { input in
                stops.withUnsafeBufferPointer { stop in
                toklet_session_generate(session, operation, input.baseAddress, input.count,
                    maximum, stop.baseAddress, stop.count, flag, nativeSink,
                    Unmanaged.passUnretained(sink).toOpaque(), &report, &error)
                }
            }
        }
        lastEmitted = report.emitted_tokens
        // Query exact state even on recoverable generation failure. Poisoned or
        // otherwise unreadable state cannot be offered for replay.
        do { try refreshHistory() }
        catch {
            savedCount = 0
            try? releaseSession()
            phase = "Session unavailable; reset or reload required"
            throw error
        }
        if report.abi_version != TOKLET_ABI_VERSION {
            var queryError = TokletError()
            try requireOK(toklet_session_state(session, &report, &queryError), &queryError)
        }
        phase = "Ready"
        try requireOK(code, &error)
    }
    func perform(_ command: NativeCommand, flag: OpaquePointer, worker: NativeWorker) throws {
        lastEmitted = 0
        score = nil
        requestBudget = nil
        switch command {
        case .preview, .replaceWithEmptySession:
            throw invalid("Planning previews must use the independent worker branch")
        case .chat(let id,let conversation,let maximum,let autoCancel):
            try generateChat(id,conversation,maximum,autoCancel,flag,worker)
        case .score(let target):
            guard let session else { throw invalid("Load or replay a session first") }
            var error = TokletError(); var value = TokletScore()
            try requireOK(toklet_session_score(session,target,&value,&error),&error)
            guard value.abi_version == TOKLET_ABI_VERSION,
                  value.struct_size == UInt32(MemoryLayout<TokletScore>.size),
                  value.target_token == target, value.vocab_size > 0,
                  target < value.vocab_size, value.greedy_token < value.vocab_size,
                  value.evaluated_tokens > 0, value.evaluated_tokens == report.evaluated_tokens,
                  report.history_tokens == report.evaluated_tokens, report.pending_tokens == 0,
                  value.target_logit.isFinite, value.greedy_logit.isFinite,
                  value.greedy_logit >= value.target_logit,
                  value.negative_log_likelihood.isFinite, value.negative_log_likelihood >= 0 else {
                throw invalid("Invalid committed-distribution score record")
            }
            score = NativeDistributionScore(target:target,greedy:value.greedy_token,
                vocabulary:value.vocab_size,evaluated:value.evaluated_tokens,
                targetLogit:value.target_logit,greedyLogit:value.greedy_logit,
                negativeLogLikelihood:value.negative_log_likelihood)
            phase = "Ready"

        case .load(let requested):
            compatibility = nil; runtimePlan = nil; compatibilityDirectory = ""
            guard model == nil else { throw invalid("Unload the current model first") }
            acceptedIdentity = nil; acceptedLoadRequestID = nil
            guard requested.packedDirectory.isEmpty || requested.metal,
                  !requested.mlx || (requested.metal && requested.packedDirectory.isEmpty) else {
                throw invalid("MLX requires Metal and no separate Q4 folder; packed Q4 requires Metal")
            }
            guard requested.loadIdentity == .ordinary || (requested.metal && requested.mlx && requested.chat && requested.packedDirectory.isEmpty) else {
                throw invalid("Audited managed identity requires native MLX Chat loading")
            }
            guard (1...32768).contains(requested.capacity), requested.chunk > 0 else { throw invalid("Invalid capacity/chunk") }
            let loadUpper = try requested.admittedLoadUpper()
            guard requested.directory.utf8.count <= 16384, requested.packedDirectory.utf8.count <= 16384 else {
                throw invalid("Artifact path exceeds 16 KiB")
            }
            let path = Array(requested.directory.utf8); let packed = Array(requested.packedDirectory.utf8)
            guard !path.isEmpty, path.count <= 16384, packed.count <= 16384,
                  !path.contains(0), !packed.contains(0) else { throw invalid("Invalid artifact path") }
            if requested.mlx {
                let budget = try NativeResourcePlan.numericalBudget(wholeBudget:requested.numericalBudget)
                let metadata = try NativeCompatibility.inspect(directory:requested.directory,context:loadUpper,budget:requested.numericalBudget)
                compatibility = metadata; compatibilityDirectory = requested.directory
                guard metadata.mayAttemptLoad else {
                    throw NSError(domain:"TokletCompatibility",code:Int(metadata.rejectionCode),userInfo:[NSLocalizedDescriptionKey:metadata.summary])
                }
                // Ordinary numerical metadata only, including for audited Smol.
                // Actual audited source/model/session admission remains final.
                let plan = try NativeResourcePlan.inspect(directory:requested.directory,capacity:requested.capacity,chunk:requested.chunk,budget:budget,context:loadUpper,attention:requested.planningAttention)
                runtimePlan = plan
                guard plan.fitsRequestedBudgets else {
                    throw NSError(domain:"TokletResourcePlan",code:Int(TOKLET_LOAD_ERROR),userInfo:[NSLocalizedDescriptionKey:plan.summary])
                }
                try worker.checkLoadAfterRuntimeInspection(plan)
            }
            // The real loader reopens and validates all payload/device/session
            // requirements; captured metadata is never authority to skip them.
            var options = try requested.loadOptions(); var error = TokletError()
            let code = path.withUnsafeBufferPointer { path in
                packed.withUnsafeBufferPointer { packed in
                    requested.loadIdentity == .managedSmol ? toklet_model_load_mlx_smol_audited(path.baseAddress,path.count,&options,&model,&error)
                    : requested.loadIdentity == .managedQwen2 ? toklet_model_load_mlx_qwen2_audited(path.baseAddress,path.count,&options,&model,&error)
                    : requested.loadIdentity == .managedQwen15 ? toklet_model_load_mlx_qwen15_audited(path.baseAddress,path.count,&options,&model,&error)
                    : requested.loadIdentity == .managedQwen3 ? toklet_model_load_mlx_qwen3_audited(path.baseAddress,path.count,&options,&model,&error)
                    : requested.loadIdentity == .managedLlama3 ? toklet_model_load_mlx_llama3_audited(path.baseAddress,path.count,&options,&model,&error)
                    : requested.loadIdentity == .managedMistral ? toklet_model_load_mlx_mistral_audited(path.baseAddress,path.count,&options,&model,&error)
                    : requested.mlx ? toklet_model_load_mlx(path.baseAddress,path.count,&options,&model,&error)
                    : requested.metal ? toklet_model_load_metal(path.baseAddress,path.count,packed.baseAddress,packed.count,&options,&model,&error)
                                    : toklet_model_load(path.baseAddress,path.count,&options,&model,&error)
                }
            }
            try requireOK(code, &error)
            settings = requested
            saved = Array(repeating: 0, count: Int(requested.capacity)); savedCount = 0
            do {
                try createSession()
                if requested.chat {
                    let chatCode: UInt32
                    switch requested.loadIdentity {
                    case .ordinary:
                        chatCode = toklet_chat_create(model,requested.textBudget,&chat,&error)
                    case .managedQwen3:
                        chatCode = toklet_chat_create_qwen3_audited(model,requested.textBudget,&chat,&error)
                    case .managedSmol, .managedQwen2, .managedQwen15, .managedMistral, .managedLlama3:
                        chatCode = toklet_chat_create_complete_audited(model,requested.textBudget,&chat,&error)
                    }
                    try requireOK(chatCode,&error)
                    var stops = [UInt32](repeating:0,count:8); var count = 0
                    let code = stops.withUnsafeMutableBufferPointer { buffer in
                        toklet_chat_stop_ids(chat,buffer.baseAddress,buffer.count,&count,&error)
                    }
                    try requireOK(code,&error)
                    guard (1...stops.count).contains(count), Set(stops.prefix(count)).count == count else { throw invalid("Invalid chat stop set") }
                    chatStops = Array(stops.prefix(count))
                }
                try requireOK(toklet_session_state(session,&report,&error),&error)
                if let profile = requested.loadIdentity.identityProfile {
                    guard let model, let chat, let modelContext else { throw invalid("Audited model setup is incomplete") }
                    // Publish only after every model/session/codec/stop/state setup
                    // step succeeds. Any query failure unwinds the whole new load.
                    let identity = try NativeAcceptedIdentity.inspect(model:model,chat:chat,context:modelContext)
                    guard identity.profile == profile else { throw invalid("Accepted identity differs from requested managed profile") }
                    acceptedIdentity = identity
                }
                acceptedLoadRequestID = requested.loadRequestID
            } catch { try? unload(); throw error }
            phase = "Ready"
        case .generate(let operation,let tokens,let maximum,let stops,let autoCancel):
            try generate(operation,tokens,maximum,stops,autoCancel,flag,worker)
        case .reset:
            guard let session else { throw invalid("Load or replay a session first") }
            var error = TokletError()
            try requireOK(toklet_session_reset(session,&error),&error)
            savedCount = 0
            try requireOK(toklet_session_state(session,&report,&error),&error)
            phase = "Ready"
        case .suspend:
            // Also invalidates a queued load that lifecycle cleanup superseded.
            compatibility = nil; runtimePlan = nil; compatibilityDirectory = ""
            try releaseSession()
            phase = model == nil ? "Unloaded" : "Session released; replay required"
        case .replay:
            guard session == nil else { throw invalid("Suspend the session before replay") }
            let prefix = Array(saved.prefix(savedCount))
            func restorePrefix() {
                saved.replaceSubrange(0..<prefix.count,with:prefix)
                savedCount = prefix.count
            }
            do {
                try createSession()
                if !prefix.isEmpty {
                    // Replay is transactional in the application: cancellation
                    // may leave native history empty or only partially evaluated.
                    try generate(0,prefix,0,[],0,flag,worker)
                    if report.reason == 3 {
                        try releaseSession(); restorePrefix(); report.reason = 3
                        phase = "Replay cancelled; prefix retained"
                        return
                    }
                    guard savedCount == prefix.count, Array(saved.prefix(savedCount)) == prefix,
                          report.evaluated_tokens == UInt32(prefix.count), report.pending_tokens == 0 else {
                        throw invalid("Incomplete replay; retained prefix required")
                    }
                }
                phase = "Ready"
            } catch {
                try? releaseSession(); restorePrefix()
                phase = "Replay failed; prefix retained"
                throw error
            }
        case .unload:
            try unload()
        }
    }
}

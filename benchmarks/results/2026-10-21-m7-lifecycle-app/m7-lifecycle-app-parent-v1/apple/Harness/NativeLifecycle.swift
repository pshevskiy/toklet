import Foundation

// The production UI and explicit lifecycle diagnostic use the same requests.
// These methods only request work; native cleanup remains on the worker thread.
enum NativeLifecycleAction: String, Codable, Sendable {
    case inactive, background, active, memoryWarning
}
func requestNativeLifecycle(_ action: NativeLifecycleAction, worker: NativeWorker) {
    switch action {
    case .inactive: worker.cancel()
    case .background: worker.interrupt(unload:false)
    case .active: break // Replay/resume requires an explicit coordinator action.
    case .memoryWarning: worker.interrupt(unload:true)
    }
}

enum NativeLifecycleOrigin: String, Codable, Sendable {
    case systemNotification, syntheticCheck, coordinator
}
struct NativeLifecycleState: Codable, Sendable {
    let busy: Bool
    let closed: Bool
    let workerFinished: Bool
    let hasModel: Bool
    let hasSession: Bool
    let status: UInt32
    let history: UInt32
    let evaluated: UInt32
    let pending: UInt32
    let reason: UInt32
    let savedHistory: Int
    let savedPrefix: [UInt32]?
    let ids: [UInt32]

    init(_ state: NativeSnapshot) {
        busy = state.busy;closed = state.closed;workerFinished = state.workerFinished
        hasModel = state.hasModel;hasSession = state.hasSession;status = state.status
        history = state.history;evaluated = state.evaluated;pending = state.pending;reason = state.reason
        savedHistory = state.savedHistory;savedPrefix = state.savedPrefix;ids = state.ids
    }
}
struct NativeLifecycleEvent: Codable, Sendable {
    let index: Int
    let name: String
    let origin: NativeLifecycleOrigin
    let observedUptimeNanoseconds: UInt64
    let recordedUptimeNanoseconds: UInt64
    let applicationState: Int?
    let command: String?
    let before: NativeLifecycleState
    let state: NativeLifecycleState
    let footprint: NativeFootprint
}
struct NativeLifecycleTraceReport: Codable, Sendable {
    let events: [NativeLifecycleEvent]
    let activeCommand: String?
    let error: String?
    let detachedAfterClose: Bool
}
private func lifecycleError(_ message: String) -> NSError {
    NSError(domain:"TokletLifecycle",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}

// Cold bounded trace only. No model/session pointer or strong worker owner is
// retained. Lock protects command identity, entries and terminal failure. Calls
// into the worker follow trace-lock -> worker-condition order; the worker never
// invokes this trace. Record sinks consume returned snapshots outside the lock.
final class NativeLifecycleTrace: @unchecked Sendable {
    private let lock = NSLock()
    private weak var worker: NativeWorker?
    private var entries: [NativeLifecycleEvent] = []
    private var command: String?
    private var failure: String?
    private var detachedAfterClose = false
    private let maximumEvents: Int
    private let maximumPrefix: Int
    private var lastTime: UInt64 = 0
    // Non-UI deterministic checks may wait for cleanup between snapshots.
    // Production leaves this nil; lifecycle callbacks never wait for cleanup.
    private let afterRequest: (@Sendable () -> Void)?

    init(worker: NativeWorker, maximumEvents: Int = 64, maximumPrefix: Int = 1152, afterRequest: (@Sendable () -> Void)? = nil) throws {
        guard (1...64).contains(maximumEvents), (1...1152).contains(maximumPrefix) else {
            throw lifecycleError("Invalid lifecycle trace bounds")
        }
        self.worker = worker;self.maximumEvents = maximumEvents;self.maximumPrefix = maximumPrefix;self.afterRequest = afterRequest
        entries.reserveCapacity(maximumEvents)
    }
    private func failed(_ message: String) -> NSError {
        if failure == nil { failure = message }
        worker?.cancel()
        return lifecycleError(message)
    }
    private func label(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 80 && name.utf8.allSatisfy {
            (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45,95].contains($0)
        }
    }
    func begin(_ name: String) throws {
        lock.lock();defer { lock.unlock() }
        guard failure == nil, label(name), command == nil else { throw failed("Invalid overlapping lifecycle command") }
        command = name
    }
    func end(_ name: String) throws {
        lock.lock();defer { lock.unlock() }
        guard failure == nil, command == name else { throw failed("Lifecycle command completion differs") }
        command = nil
    }
    @discardableResult
    func record(_ name: String, origin: NativeLifecycleOrigin, applicationState: Int?,
                observedUptimeNanoseconds: UInt64? = nil,
                action: NativeLifecycleAction? = nil) throws -> NativeLifecycleEvent {
        lock.lock();defer { lock.unlock() }
        let owner = worker
        let before = owner?.snapshot()
        // Resource safety does not depend on diagnostic admission. Exhausted
        // or failed traces must still request background/warning cleanup.
        if let action, let owner {
            requestNativeLifecycle(action,worker:owner);afterRequest?()
        }
        guard failure == nil else { throw lifecycleError(failure!) }
        let recorded = DispatchTime.now().uptimeNanoseconds
        let observed = observedUptimeNanoseconds ?? recorded
        guard label(name), entries.count < maximumEvents, observed > 0,
              observed <= recorded, recorded >= lastTime,
              applicationState == nil || (0...2).contains(applicationState!) else {
            throw failed("Lifecycle trace bounds/clock/state differs")
        }
        guard let owner, let before else { throw failed("Lifecycle worker is unavailable") }
        // Keep the original notification observation and before/requested states
        // distinct from a later completed-cleanup event. Concurrent observer
        // timestamps may precede earlier recorded entries; recorded time is
        // serialized here, and does not rewrite the original observation.
        let state = owner.snapshot()
        let footprint: NativeFootprint
        do { footprint = try NativeFootprint.read() }
        catch { throw failed("Lifecycle process footprint unavailable") }
        for value in [before,state] {
            guard value.ids.count <= 128, value.savedHistory >= 0, value.savedHistory <= maximumPrefix,
                  value.savedPrefix == nil || value.savedPrefix!.count <= maximumPrefix,
                  value.savedPrefix == nil || value.savedPrefix!.count == value.savedHistory else {
                throw failed("Lifecycle prefix exceeds admitted trace capacity")
            }
        }
        let event = NativeLifecycleEvent(index:entries.count,name:name,origin:origin,
            observedUptimeNanoseconds:observed,recordedUptimeNanoseconds:recorded,applicationState:applicationState,
            command:command,before:NativeLifecycleState(before),state:NativeLifecycleState(state),footprint:footprint)
        entries.append(event);lastTime = recorded
        return event
    }
    func snapshot() throws -> [NativeLifecycleEvent] {
        lock.lock();defer { lock.unlock() }
        if let failure { throw lifecycleError(failure) }
        return entries
    }
    // Failure evidence remains readable; successful admission still requires
    // error == nil, no active command and explicit completed-close detach.
    func report() -> NativeLifecycleTraceReport {
        lock.lock();defer { lock.unlock() }
        return NativeLifecycleTraceReport(events:entries,activeCommand:command,error:failure,
            detachedAfterClose:detachedAfterClose)
    }
    func detach() throws {
        lock.lock();defer { lock.unlock() }
        if let failure { throw lifecycleError(failure) }
        guard command == nil, let owner = worker, owner.snapshot().closed,
              owner.snapshot().workerFinished else { throw failed("Lifecycle detach requires completed close") }
        worker = nil;detachedAfterClose = true
    }
}

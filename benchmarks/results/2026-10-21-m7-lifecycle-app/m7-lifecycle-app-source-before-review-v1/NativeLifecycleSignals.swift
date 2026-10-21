import Foundation

struct NativeLifecycleRoundTrip: Codable, Sendable {
    let inactive: NativeLifecycleEvent
    let background: NativeLifecycleEvent
    let active: NativeLifecycleEvent
}
private func signalError(_ message: String) -> NSError {
    NSError(domain:"TokletLifecycleSignals",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}
// Coarse app integration boundary. The main-actor scene/notification handlers
// supply actual provenance/state. This box retains no native worker owner.
// Condition -> trace lock -> worker condition is the sole nested lock order.
final class NativeLifecycleSignals: @unchecked Sendable {
    private let condition = NSCondition()
    private weak var worker: NativeWorker?
    private var trace: NativeLifecycleTrace?
    private var armed = false
    private var inactive: NativeLifecycleEvent?
    private var background: NativeLifecycleEvent?
    private var active: NativeLifecycleEvent?
    private var failure: String?

    func attach(_ owner: NativeWorker, trace: NativeLifecycleTrace) throws {
        condition.lock();defer { condition.unlock() }
        guard worker == nil, self.trace == nil else { throw signalError("Lifecycle source already attached") }
        worker = owner;self.trace = trace
    }
    func detach() {
        condition.lock();worker = nil;trace = nil;armed = false;condition.broadcast();condition.unlock()
    }
    func observe(_ action: NativeLifecycleAction, origin: NativeLifecycleOrigin, applicationState: Int?) {
        condition.lock();defer { condition.broadcast();condition.unlock() }
        guard let trace else { return }
        let name: String
        switch action {
        case .inactive: name = "scene-inactive"
        case .background: name = "scene-background"
        case .active: name = "scene-active"
        case .memoryWarning: name = origin == .syntheticCheck ? "synthetic-memory-warning":"memory-warning"
        }
        do {
            // The trace dispatches cleanup even after its own terminal failure.
            let event = try trace.record(name,origin:origin,applicationState:applicationState,action:action)
            guard armed else { return }
            switch action {
            case .inactive:
                guard inactive == nil, background == nil, active == nil else { throw signalError("Repeated inactive phase") }
                inactive = event
            case .background:
                guard inactive != nil, background == nil, active == nil else { throw signalError("Background phase order differs") }
                background = event
            case .active:
                guard background != nil, active == nil else { throw signalError("Foreground phase order differs") }
                active = event
            case .memoryWarning: throw signalError("Unexpected memory warning during prefix round trip")
            }
        } catch {
            if failure == nil { failure = String(error.localizedDescription.prefix(512)) }
            worker?.cancel()
        }
    }
    func arm() throws {
        condition.lock();defer { condition.unlock() }
        guard failure == nil, worker != nil, trace != nil, !armed else { throw signalError("Lifecycle source cannot arm") }
        inactive = nil;background = nil;active = nil;armed = true
    }
    func check() throws {
        condition.lock();defer { condition.unlock() }
        if let failure { throw signalError(failure) }
    }
    func finish(seconds: TimeInterval, requireSystem: Bool) throws -> NativeLifecycleRoundTrip {
        guard seconds.isFinite, (0...180).contains(seconds) else { throw signalError("Invalid lifecycle wait") }
        condition.lock();defer { condition.unlock() }
        let deadline = Date(timeIntervalSinceNow:seconds)
        while armed, failure == nil, active == nil {
            if !condition.wait(until:deadline) { break }
        }
        if let failure { throw signalError(failure) }
        guard armed, let inactive, let background, let active else { throw signalError("Incomplete lifecycle round trip") }
        guard inactive.index < background.index, background.index < active.index,
              !requireSystem || [inactive,background,active].allSatisfy({ $0.origin == .scenePhase }) else {
            throw signalError("Lifecycle round trip provenance/order differs")
        }
        armed = false
        return NativeLifecycleRoundTrip(inactive:inactive,background:background,active:active)
    }
}

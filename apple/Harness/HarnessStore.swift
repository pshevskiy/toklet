import Foundation
import Combine

struct GenerationMetrics: Equatable, Sendable {
    let tokens: Int
    let firstTokenSeconds: Double?
    let decodeTokensPerSecond: Double?
    let totalSeconds: Double?
    init(_ state: NativeSnapshot) {
        tokens = state.tokenTimings.count
        let first = state.tokenTimings.first?.callbackUptimeNanoseconds
        let last = state.tokenTimings.last?.callbackUptimeNanoseconds
        if let start = state.requestStartedUptimeNanoseconds, let first, first >= start {
            firstTokenSeconds = Double(first - start) / 1e9
        } else { firstTokenSeconds = nil }
        if tokens > 1, let first, let last, last > first {
            decodeTokensPerSecond = Double(tokens - 1) * 1e9 / Double(last - first)
        } else { decodeTokensPerSecond = nil }
        if let start = state.requestStartedUptimeNanoseconds,
           let end = state.requestCompletedUptimeNanoseconds, end >= start {
            totalSeconds = Double(end - start) / 1e9
        } else { totalSeconds = nil }
    }
}

struct AppMemoryReading: Sendable {
    let usedBytes: UInt64
    let allowedBytes: UInt64
    static func current() throws -> Self {
        let used = try NativeFootprint.read().physicalBytes
        let device = try HubDeviceSnapshot.current()
        #if os(iOS)
        let sum = used.addingReportingOverflow(device.availableMemory)
        guard !sum.overflow else { throw HubFailure("Memory allowance unavailable") }
        let allowed = sum.partialValue
        #else
        let allowed = min(device.physicalMemory / 2,6 * 1024 * 1024 * 1024)
        #endif
        return Self(usedBytes:used,allowedBytes:allowed)
    }
}

@MainActor
final class HarnessStore: ObservableObject {
    var worker: NativeWorker?
    let diagnosticsOnly: Bool
    let lifecycleSignals: NativeLifecycleSignals?
    @Published var state = NativeSnapshot()
    @Published var inputError = ""
    @Published var expert = false
    @Published private(set) var memory: AppMemoryReading?
    private var memorySampleUptime: UInt64 = 0
    private var pendingMode: Bool?

    init(enabled: Bool, lifecycleEnabled: Bool = false) {
        diagnosticsOnly = !enabled
        lifecycleSignals = lifecycleEnabled ? NativeLifecycleSignals():nil
        do { worker = enabled ? try NativeWorker(requestSummaryMode:.configured) : nil }
        catch { worker = nil; inputError = error.localizedDescription }
    }
    func poll() {
        let now = DispatchTime.now().uptimeNanoseconds
        if now >= memorySampleUptime, now - memorySampleUptime >= 1_000_000_000 {
            memorySampleUptime = now
            memory = try? AppMemoryReading.current()
        }
        if let snapshot = worker?.snapshot(), snapshot.revision != state.revision {
            state = snapshot
        }
        if let mode = pendingMode, !state.busy, (!state.hasModel || state.closed) {
            pendingMode = nil
            setExpert(mode)
        }
    }
    func setExpert(_ value: Bool) {
        guard !diagnosticsOnly else { return }
        if state.closed || worker == nil {
            do { worker = try NativeWorker(requestSummaryMode:.configured); state = worker!.snapshot() }
            catch { inputError = error.localizedDescription; return }
        }
        if state.busy || state.hasModel {
            pendingMode = value
            worker?.interrupt(unload:true)
        } else { expert = value }
    }
    func submit(_ command: NativeCommand) {
        inputError = ""
        if worker?.submit(command) != true { inputError = "Worker is busy or closed" }
        poll()
    }
}

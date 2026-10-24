import Foundation
import Combine

@MainActor
final class HarnessStore: ObservableObject {
    var worker: NativeWorker?
    let diagnosticsOnly: Bool
    let lifecycleSignals: NativeLifecycleSignals?
    @Published var state = NativeSnapshot()
    @Published var inputError = ""
    @Published var expert = false
    private var pendingMode: Bool?

    init(enabled: Bool, lifecycleEnabled: Bool = false) {
        diagnosticsOnly = !enabled
        lifecycleSignals = lifecycleEnabled ? NativeLifecycleSignals():nil
        do { worker = enabled ? try NativeWorker() : nil }
        catch { worker = nil; inputError = error.localizedDescription }
    }
    func poll() {
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
            do { worker = try NativeWorker(); state = worker!.snapshot() }
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

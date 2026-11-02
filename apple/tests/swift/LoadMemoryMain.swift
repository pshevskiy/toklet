import Foundation
import CToklet

@main
struct LoadMemoryMain {
    static func check(_ ok: Bool, _ message: String) throws {
        if !ok { throw NativeContextFailure(message) }
    }
    static func footprint(_ current: UInt64, _ peak: UInt64) -> NativeFootprint {
        NativeFootprint(physicalBytes:current,lifetimePeakPhysicalBytes:peak,returnedNaturalCount:1)
    }
    static func main() throws {
        var reads = 0
        let disabled = NativeLoadMemoryProbe.begin(enabled:false) { reads += 1; return footprint(10,20) }
        _ = disabled?.finish(loadRequestID:nil,capacity:512,upper:1024,completed:true,model:1,session:2) {
            reads += 1; return footprint(15,30)
        }
        try check(disabled == nil && reads == 0,"Disabled observation performed a read")
        let id = UUID()
        let probe = NativeLoadMemoryProbe.begin(enabled:true) { footprint(10,20) }!
        func finish(_ current: UInt64, _ peak: UInt64, completed: Bool = true) -> NativeLoadMemorySnapshot {
            probe.finish(loadRequestID:id,capacity:512,upper:1024,completed:completed,model:1,session:2) { footprint(current,peak) }
        }
        let increased = finish(15,30)
        try check(increased.establishedProcessPeakBytes == 30,"New process peak was not attributed to the bracketed interval")
        let equal = finish(15,20)
        try check(equal.establishedProcessPeakBytes == nil && equal.peakExplanation.contains("prior process peak"),"Historical peak became a load peak")
        for sample in [finish(15,19),finish(31,30)] {
            try check(sample.establishedProcessPeakBytes == nil && sample.peakExplanation.contains("Inconsistent"),"Inconsistent counters established a peak")
        }
        let badStart = NativeLoadMemoryProbe.begin(enabled:true) { footprint(21,20) }!
        let bad = badStart.finish(loadRequestID:nil,capacity:512,upper:1024,completed:true,model:1,session:2) { footprint(25,30) }
        try check(bad.establishedProcessPeakBytes == nil,"Inconsistent start counter established a peak")
        let unavailable = NativeLoadMemoryProbe.begin(enabled:true) { throw NativeContextFailure("synthetic unavailable") }!
        let noStart = unavailable.finish(loadRequestID:nil,capacity:512,upper:1024,completed:true,model:1,session:2) { footprint(15,30) }
        let noEnd = probe.finish(loadRequestID:nil,capacity:512,upper:1024,completed:true,model:1,session:2) { throw NativeContextFailure("synthetic unavailable") }
        try check(noStart.completed && noEnd.completed && noStart.establishedProcessPeakBytes == nil && noEnd.establishedProcessPeakBytes == nil,
            "Unavailable observation changed successful outcome or established a peak")
        var raw = TokletContextInfo()
        raw.abi_version = TOKLET_ABI_VERSION; raw.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        raw.backend = TOKLET_CONTEXT_CPU; raw.model_instance = 1; raw.session_instance = 2
        raw.engine_global_position_ceiling = 32768; raw.loaded_position_limit = 1024; raw.session_capacity = 512
        raw.numerical_and_bridge_admitted_bytes = 1024*1024
        let model = NativeModelContext(try NativeContextFacts(raw))
        try check(increased.matches(model:model,loadRequestID:id) && !increased.matches(model:model,loadRequestID:UUID())
            && !increased.matches(model:nil,loadRequestID:id),"Load observation lost request/model binding")
        raw.model_instance = 3
        try check(!increased.matches(model:NativeModelContext(try NativeContextFacts(raw)),loadRequestID:id),"Different model accepted old load metrics")
        let failed = finish(15,30,completed:false)
        try check(!failed.completed && failed.modelInstance == nil && failed.initialSessionInstance == nil
            && !failed.matches(model:model,loadRequestID:id),"Failed attempt became a successful model observation")
        let harnessRequest = FriendlyLoadRequest(folder:URL(fileURLWithPath:"/never-opened"),mode:.managedQwen2,conversation:UUID(),measureLoadingMemory:true)
        try check(harnessRequest.measureLoadingMemory && NativeSettings(directory:"/never-opened").measureLoadingMemory == false,"Load observation opt-in/default changed")
        let worker = try NativeWorker()
        var failure: Error?
        do {
            var settings = NativeSettings(directory:FileManager.default.temporaryDirectory.appendingPathComponent("missing-model-" + UUID().uuidString).path)
            settings.measureLoadingMemory = true; settings.loadRequestID = UUID()
            try check(worker.submit(.load(settings)),"Failed-load observation submission rejected")
            guard let state = worker.waitUntilIdle(seconds:30), let record = state.loadMemory else { throw NativeContextFailure("Missing failed-attempt observation") }
            try check(state.status != 0 && !state.hasModel && !record.completed
                && record.loadRequestID == settings.loadRequestID && record.modelInstance == nil && record.initialSessionInstance == nil,
                "Failed load was attributed to a successful model")
            settings.measureLoadingMemory = false
            try check(worker.submit(.load(settings)),"Disabled failed-load submission rejected")
            guard let disabledState = worker.waitUntilIdle(seconds:30) else { throw NativeContextFailure("Disabled load timeout") }
            try check(disabledState.status != 0 && disabledState.loadMemory == nil,"Disabled load retained old attempt observation")
        } catch { failure = error }
        worker.close(); try check(worker.waitUntilClosed(seconds:30),"Memory-control worker close timeout")
        if let failure { throw failure }
        print("LOAD_MEMORY_CONTROLS passed")
    }
}

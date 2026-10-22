import Foundation
import CryptoKit

private func expectMLX(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain:"TokletMLXJobs",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func digestMLX(_ data: Data) -> String {
    SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined()
}
private func emitMLX<T: Encodable>(_ name: String, _ value: T) throws {
    let data = try JSONEncoder().encode(value)
    print(name+String(decoding:data,as:UTF8.self))
}
@main private struct MLXJobsMain {
    static func main() async throws { try await Task.detached { try run() }.value }
    private static func run() throws {
        guard CommandLine.arguments.count == 3 else { throw NSError(domain:"TokletMLXJobs",code:2) }
        let fixture = URL(fileURLWithPath:CommandLine.arguments[1],isDirectory:true)
        let root = URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
        try FileManager.default.createDirectory(at:root.appendingPathComponent("Model"),withIntermediateDirectories:true)
        var files: [String:String] = [:]
        for name in ["config.json","model.safetensors"] {
            let data = try Data(contentsOf:fixture.appendingPathComponent(name))
            try data.write(to:root.appendingPathComponent("Model/"+name),options:.withoutOverwriting)
            files["Model/"+name] = digestMLX(data)
        }
        let budget: UInt64 = 256*1024*1024, process: UInt64 = 2*1024*1024*1024
        var measurement = NativeMeasurementConfig(version:2,modelFolder:"Model",packedFolder:"",files:files,metal:true,
            capacity:32,chunk:3,numericalBudget:budget,processBudget:process,prompt:[1,7,13,3],maximum:4,
            expectedPrefix:[47,47,47,47],warmups:3,repetitions:10,minimumNativeSeconds:0)
        measurement.mlx = true
        var corpus = Data()
        for value: UInt32 in [1,7,13,3] { for shift in [0,8,16,24] { corpus.append(UInt8(truncatingIfNeeded:value>>shift)) } }
        try corpus.write(to:root.appendingPathComponent("corpus.u32le"),options:.withoutOverwriting)
        var quality = NativeQualityConfig(version:2,modelFolder:"Model",packedFolder:"",files:files,metal:true,
            capacity:4,chunk:1,numericalBudget:budget,processBudget:process,tokensFile:"corpus.u32le",
            tokensSHA256:digestMLX(corpus),targets:3,blockSize:3)
        quality.mlx = true
        let taskSet = NativeTaskSet(version:1,vocabulary:64,stops:[62,63],cases:[
            NativeTaskCase(id:"HumanEval/0",promptTokens:[1,7,13],maximum:3),
            NativeTaskCase(id:"second",promptTokens:[1,7,13,3],maximum:2)])
        let tasksData = try JSONEncoder().encode(taskSet)
        try tasksData.write(to:root.appendingPathComponent("tasks.json"),options:.withoutOverwriting)
        var tasks = NativeTasksConfig(version:2,modelFolder:"Model",packedFolder:"",files:files,metal:true,
            capacity:6,chunk:3,numericalBudget:budget,processBudget:process,tasksFile:"tasks.json",
            tasksSHA256:digestMLX(tasksData),caseCount:2)
        tasks.mlx = true
        var lifecycle = NativeLifecycleConfig(version:2,modelFolder:"Model",packedFolder:"",files:files,metal:true,
            capacity:32,chunk:3,numericalBudget:budget,processBudget:process,prompt:[1,7,13,3],maximum:25,vocabulary:64)
        lifecycle.mlx = true
        var rejected = 0
        let encodings = try [JSONEncoder().encode(measurement),JSONEncoder().encode(quality),JSONEncoder().encode(tasks),JSONEncoder().encode(lifecycle)]
        for (index,raw) in encodings.enumerated() {
            func admit(_ object: [String:Any]) throws {
                let bytes = try JSONSerialization.data(withJSONObject:object,options:.sortedKeys)
                let hash = digestMLX(bytes)
                try FileManager.default.createDirectory(at:root.appendingPathComponent("M7Jobs"),withIntermediateDirectories:true)
                let name = "M7Jobs/"+hash+".json"
                let path = root.appendingPathComponent(name)
                if !FileManager.default.fileExists(atPath:path.path) { try bytes.write(to:path,options:.withoutOverwriting) }
                switch index {
                case 0: _ = try admitMeasurementConfig(root:root,name:name,sha256:hash)
                case 1: _ = try admitQualityConfig(root:root,name:name,sha256:hash)
                case 2: _ = try admitTasksConfig(root:root,name:name,sha256:hash)
                default: _ = try admitLifecycleConfig(root:root,name:name,sha256:hash)
                }
            }
            let base = try JSONSerialization.jsonObject(with:raw) as! [String:Any]
            try admit(base)
            let changes: [(inout [String:Any]) -> Void] = [
                { $0.removeValue(forKey:"mlx") }, { $0["mlx"] = false }, { $0["mlx"] = NSNull() },
                { $0["mlx"] = "true" }, { $0["metal"] = false }, { $0["packedFolder"] = "Packed" },
                { $0["version"] = 1 }, { $0["version"] = 3 }, { $0["extra"] = 1 },
                { $0["version"] = true }]
            for change in changes {
                var object = base;change(&object)
                var failed = false;do { try admit(object) } catch { failed = true }
                try expectMLX(failed,"Invalid MLX schema/backend admitted for config \(index)")
                rejected += 1
            }
            var legacy = base;legacy["version"] = 1;legacy.removeValue(forKey:"mlx")
            try admit(legacy)
        }
        // Execute the authenticated decoded objects, not the authored originals.
        func binding(_ data: Data) throws -> (String,String) {
            let object = try JSONSerialization.jsonObject(with:data)
            let canonical = try JSONSerialization.data(withJSONObject:object,options:.sortedKeys)
            let hash = digestMLX(canonical)
            return ("M7Jobs/"+hash+".json",hash)
        }
        let m = try binding(encodings[0]);measurement = try admitMeasurementConfig(root:root,name:m.0,sha256:m.1)
        let q = try binding(encodings[1]);quality = try admitQualityConfig(root:root,name:q.0,sha256:q.1)
        let t = try binding(encodings[2]);tasks = try admitTasksConfig(root:root,name:t.0,sha256:t.1)
        let l = try binding(encodings[3]);lifecycle = try admitLifecycleConfig(root:root,name:l.0,sha256:l.1)
        var samples: [NativeMeasurementRequest] = []
        let measured = runNativeMeasurement(measurement,root:root) { samples.append($0) }
        try expectMLX(measured.passed && measured.closed && measured.released && measured.backend == "metal-mlx"
            && samples.count == 13,"MLX measurement failed: \(measured.error ?? "")")
        for sample in samples { try emitMLX("MLX_JOB_MEASUREMENT_SAMPLE=",sample) }
        try emitMLX("MLX_JOB_MEASUREMENT=",measured)
        var scores: [NativeQualitySample] = []
        let qualityResult = runNativeQuality(quality,root:root) { scores.append($0) }
        try expectMLX(qualityResult.passed && qualityResult.closed && qualityResult.released
            && qualityResult.backend == "metal-mlx" && qualityResult.samplesCompleted == 3
            && scores.map(\.target.targetIndex) == [0,1,2],"MLX quality failed: \(qualityResult.error ?? "")")
        for score in scores { try emitMLX("MLX_JOB_QUALITY_SAMPLE=",score) }
        try emitMLX("MLX_JOB_QUALITY=",qualityResult)
        var responses: [NativeTaskCaseSample] = []
        let taskResult = runNativeTasks(tasks,root:root,onChunk:{ _ in },onCase:{ responses.append($0) })
        try expectMLX(taskResult.passed && taskResult.closed && taskResult.released && taskResult.backend == "metal-mlx"
            && responses.map(\.response.ids) == [[47,47,47],[47,47]],"MLX tasks failed: \(taskResult.error ?? "")")
        for response in responses { try emitMLX("MLX_JOB_TASK_SAMPLE=",response) }
        try emitMLX("MLX_JOB_TASKS=",taskResult)
        let state = runNativeState(lifecycle,root:root,signals:NativeLifecycleSignals(),requirePrefillWitness:false)
        try expectMLX(state.passed && state.closed && state.released && state.backend == "metal-mlx",
                      "MLX state failed: \(state.error ?? "")")
        try emitMLX("MLX_JOB_STATE=",state)
        let interrupted = runNativeLifecycle(lifecycle,root:root,signals:NativeLifecycleSignals(),requireSystem:false) { _ in
            throw NSError(domain:"TokletMLXJobs",code:3,userInfo:[NSLocalizedDescriptionKey:"Forced READY sink failure"])
        }
        try expectMLX(!interrupted.passed && interrupted.closed && interrupted.released && interrupted.backend == "metal-mlx"
            && interrupted.filesBefore == files && interrupted.filesAfter == files && interrupted.loadedAdmittedBytes > 0
            && interrupted.cleanupErrors.isEmpty && interrupted.endpointErrors.isEmpty
            && interrupted.trace?.error == nil && interrupted.trace?.activeCommand == nil
            && interrupted.trace?.detachedAfterClose == true
            && ["No demonstrably active generation window","Forced READY sink failure"].contains(interrupted.error ?? ""),
            "MLX lifecycle failed to close on bounded tiny window")
        try emitMLX("MLX_JOB_LIFECYCLE_FAILURE=",interrupted)
        print("MLX_JOBS_SUMMARY={\"passed\":true,\"schema_rejections\":\(rejected),\"measurement_samples\":13,\"quality_targets\":3,\"task_cases\":2,\"state_passed\":true,\"lifecycle_failed_closed\":true,\"physical_device_qualified\":false,\"full_quality_qualified\":false}")
    }
}

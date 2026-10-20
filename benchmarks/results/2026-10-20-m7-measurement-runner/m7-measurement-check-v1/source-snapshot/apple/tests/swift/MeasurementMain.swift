import Foundation
import Darwin
import CryptoKit

private func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain:"MeasurementChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func digest(_ data: Data) -> String {
    SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined()
}
private func rejected(_ body: () throws -> Void) throws {
    var failed = false
    do { try body() } catch { failed = true }
    try check(failed,"Rejected admission succeeded")
}

@main struct MeasurementMain {
    static func main() {
        DispatchQueue.global(qos:.userInitiated).async {
            do {
                try run()
                exit(0)
            } catch {
                try? FileHandle.standardError.write(contentsOf:Data("Measurement checks: \(error)\n".utf8))
                exit(1)
            }
        }
        dispatchMain()
    }
    private static func run() throws {
        guard CommandLine.arguments.count == 3 else { throw NSError(domain:"MeasurementChecks",code:2) }
        let fixture = URL(fileURLWithPath:CommandLine.arguments[1],isDirectory:true)
        let root = URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
        try check(!FileManager.default.fileExists(atPath:root.path),"Exclusive check folder required")
        try FileManager.default.createDirectory(at:root.appendingPathComponent("Model"),withIntermediateDirectories:true)
        var files: [String:String] = [:]
        for name in ["config.json","model.safetensors"] {
            let data = try Data(contentsOf:fixture.appendingPathComponent(name))
            try data.write(to:root.appendingPathComponent("Model/"+name),options:.withoutOverwriting)
            files["Model/"+name] = digest(data)
        }
        func config(_ metal: Bool = false, expected: [UInt32] = [47,47], numerical: UInt64 = 128*1024*1024,
                    process: UInt64 = 512*1024*1024) -> NativeMeasurementConfig {
            NativeMeasurementConfig(version:1,modelFolder:"Model",packedFolder:"",files:files,metal:metal,
                capacity:32,chunk:3,numericalBudget:numerical,processBudget:process,prompt:[1,7,13,3],
                maximum:2,expectedPrefix:expected,warmups:3,repetitions:10,minimumNativeSeconds:0)
        }
        var cases: [String] = []
        let encoded = try JSONEncoder().encode(config())
        try encoded.write(to:root.appendingPathComponent("job.json"),options:.withoutOverwriting)
        let admitted = try admitMeasurementConfig(root:root,name:"job.json",sha256:digest(encoded))
        try check(admitted.files == files,"Config admission changed inventory")
        try rejected { _ = try admitMeasurementConfig(root:root,name:"job.json",sha256:String(repeating:"0",count:64)) }
        try rejected { _ = try admitMeasurementConfig(root:root,name:"../job.json",sha256:digest(encoded)) }
        var extra = try JSONSerialization.jsonObject(with:encoded) as! [String:Any]; extra["misspelledPolicy"] = 1
        let extraData = try JSONSerialization.data(withJSONObject:extra)
        try extraData.write(to:root.appendingPathComponent("extra-job.json"),options:.withoutOverwriting)
        try rejected { _ = try admitMeasurementConfig(root:root,name:"extra-job.json",sha256:digest(extraData)) }
        let largeData = Data(repeating:32,count:512*1024+1)
        try largeData.write(to:root.appendingPathComponent("large-job.json"),options:.withoutOverwriting)
        try rejected { _ = try admitMeasurementConfig(root:root,name:"large-job.json",sha256:digest(largeData)) }
        cases.append("config SHA/path/unknown-field/size admission")
        let initial = try NativeFootprint.read()
        try check(initial.physicalBytes > 0 && initial.lifetimePeakPhysicalBytes >= initial.physicalBytes,"Self physical footprint unavailable")
        cases.append("SDK-layout bounded self TASK_VM_INFO")
        for metal in [false,true] {
            var samples: [NativeMeasurementRequest] = []
            let result = runNativeMeasurement(config(metal),root:root) { samples.append($0) }
            try check(result.passed && result.closed && result.released && result.error == nil && result.cleanupErrors.isEmpty,
                      "Actual backend measurement failed: \(String(describing:result.error)) \(result.cleanupErrors)")
            try check(result.warmupsCompleted == 3 && result.measuredCompleted == 10 && result.sumNativeLastEventSeconds > 0,
                      "Warmup/measured sequence differs")
            try check(samples.count == 13 && result.filesBefore == files && result.filesAfter == files,"Incomplete sequence/proofs")
            for (index,sample) in samples.enumerated() {
                try check(sample.index == index && sample.role == (index < 3 ? "warmup" : "measured"),"Role/index differs")
                try check(sample.tokens == [47,47] && sample.events.count == 2 && sample.status == 0,"Independent tiny ID/timing oracle")
                try check(sample.footprint.physicalBytes > 0 && sample.footprint.lifetimePeakPhysicalBytes >= initial.lifetimePeakPhysicalBytes,
                          "Whole-process peak did not retain prior history")
            }
            let resultData = try JSONEncoder().encode(result)
            print("MEASUREMENT_CHECK_RESULT_JSON=" + String(decoding:resultData,as:UTF8.self))
            for sample in samples {
                let data = try JSONEncoder().encode(sample)
                print("MEASUREMENT_CHECK_SAMPLE_JSON=" + String(decoding:data,as:UTF8.self))
            }
            cases.append("\(metal ? "metal" : "cpu") actual3warmups10requests/exactIDs/state/timing/memory/close/release")
        }
        let modelConfig = root.appendingPathComponent("Model/config.json")
        let original = try Data(contentsOf:modelConfig)
        try (original + Data([10])).write(to:modelConfig)
        var result = runNativeMeasurement(config(),root:root) { _ in throw NSError(domain:"UnexpectedSample",code:1) }
        try check(!result.passed && result.warmupsCompleted == 0 && result.measuredCompleted == 0,"Changed artifact admitted")
        try original.write(to:modelConfig)
        cases.append("changed artifact rejected before inference")
        let extraFile = root.appendingPathComponent("Model/unadmitted.txt")
        try Data([1]).write(to:extraFile,options:.withoutOverwriting)
        result = runNativeMeasurement(config(),root:root) { _ in }
        try check(!result.passed && result.warmupsCompleted == 0,"Extra artifact admitted")
        try FileManager.default.removeItem(at:extraFile)
        cases.append("exact inventory rejects extra artifact")
        let weights = root.appendingPathComponent("Model/model.safetensors")
        let outside = root.appendingPathComponent("outside.safetensors")
        try FileManager.default.moveItem(at:weights,to:outside)
        try FileManager.default.createSymbolicLink(atPath:weights.path,withDestinationPath:"../outside.safetensors")
        result = runNativeMeasurement(config(),root:root) { _ in }
        try check(!result.passed && result.warmupsCompleted == 0,"Artifact symlink admitted")
        try FileManager.default.removeItem(at:weights)
        try FileManager.default.moveItem(at:outside,to:weights)
        cases.append("artifact symlink rejected")
        result = runNativeMeasurement(config(numerical:4097,process:8192),root:root) { _ in }
        try check(!result.passed && result.warmupsCompleted == 0,"Whole-process budget admitted")
        cases.append("whole-process budget rejected before load")
        var count = 0
        result = runNativeMeasurement(config(),root:root) { _ in count += 1; throw NSError(domain:"DeliberateSinkFailure",code:1) }
        try check(!result.passed && result.closed && result.released && count == 1,"Sink failure lost cleanup")
        cases.append("output failure preserves native close/release")
        count = 0
        result = runNativeMeasurement(config(expected:[0,0]),root:root) { _ in count += 1 }
        try check(!result.passed && result.closed && result.released && count == 1,"Wrong oracle accepted")
        cases.append("wrong independent oracle rejected with cleanup")
        count = 0
        result = runNativeMeasurement(config(),root:root) { _ in
            count += 1
            if count == 1 { try (original + Data([10])).write(to:modelConfig) }
        }
        try check(!result.passed && result.closed && result.released && count == 13,"Endpoint mutation was lost")
        try original.write(to:modelConfig)
        cases.append("post-load endpoint mutation rejected after actual requests")
        let summary = try JSONSerialization.data(withJSONObject:["passed":true,"cases":cases,"physicalDeviceQualified":false],options:.sortedKeys)
        print("MEASUREMENT_CHECKS_JSON=" + String(decoding:summary,as:UTF8.self))
    }
}

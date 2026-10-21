import Foundation
import CryptoKit
import Darwin

private func checkQuality(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw NSError(domain:"TokletQualityChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func rejectsQuality(_ action: () throws -> Void) throws {
    var rejected = false;do { try action() } catch { rejected = true }
    try checkQuality(rejected,"Invalid quality input admitted")
}
private func dataSHA(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
private func emitQuality<T: Encodable>(_ marker: String, _ value: T) throws {
    let data = try JSONEncoder().encode(value)
    try FileHandle.standardOutput.write(contentsOf:Data((marker+String(decoding:data,as:UTF8.self)+"\n").utf8))
}
private struct QualityFailureRecord: Encodable {
    let label: String
    let result: NativeQualityResult
}
private struct QualityCheckSummary: Encodable {
    let passed: Bool
    let cases: [String]
    let physicalDeviceQualified = false
}
@main private struct QualityMain {
    static func main() {
        DispatchQueue.global(qos:.userInitiated).async {
            do {
                guard CommandLine.arguments.count == 3 else { throw NSError(domain:"QualityArguments",code:1) }
                let fixture = URL(fileURLWithPath:CommandLine.arguments[1],isDirectory:true)
                let cases = URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
                try FileManager.default.createDirectory(at:cases,withIntermediateDirectories:false)
                let files = ["Model/config.json":"84b6e6078f35006f49ae2e2a8663093cdc8d2813e2b7f28ebfa6142113e9c1d9",
                             "Model/model.safetensors":"b01a114bf2d44f0c6cb438035e85535b658e803c76691729facd78b15eefbce2"]
                var names: [String] = []
                func make(_ name: String, metal: Bool = false, tokens: [UInt32] = [1,7,13,3]) throws -> (URL,NativeQualityConfig) {
                    let root = cases.appendingPathComponent(name,isDirectory:true)
                    try FileManager.default.createDirectory(at:root.appendingPathComponent("Model"),withIntermediateDirectories:true)
                    for key in files.keys {
                        let target = root.appendingPathComponent(key)
                        try FileManager.default.linkItem(at:fixture.appendingPathComponent(target.lastPathComponent),to:target)
                    }
                    var data = Data()
                    for value in tokens { for shift in [0,8,16,24] { data.append(UInt8(truncatingIfNeeded:value>>shift)) } }
                    try data.write(to:root.appendingPathComponent("corpus.u32le"),options:.withoutOverwriting)
                    let config = NativeQualityConfig(version:1,modelFolder:"Model",packedFolder:"",files:files,metal:metal,
                        capacity:4,chunk:1,numericalBudget:128*1024*1024,processBudget:192*1024*1024,
                        tokensFile:"corpus.u32le",tokensSHA256:dataSHA(data),targets:3,blockSize:3)
                    return (root,config)
                }
                func changed(_ source: NativeQualityConfig, _ change: (inout [String:Any]) -> Void) throws -> NativeQualityConfig {
                    var object = try JSONSerialization.jsonObject(with:JSONEncoder().encode(source)) as! [String:Any]
                    change(&object);return try JSONDecoder().decode(NativeQualityConfig.self,from:JSONSerialization.data(withJSONObject:object))
                }
                for metal in [false,true] {
                    let label = metal ? "metal":"cpu";let (root,config) = try make(label,metal:metal)
                    let raw = try JSONEncoder().encode(config);try raw.write(to:root.appendingPathComponent("job.json"),options:.withoutOverwriting)
                    let admitted = try admitQualityConfig(root:root,name:"job.json",sha256:dataSHA(raw))
                    let admittedTokens = try admitQualityTokens(admitted,root:root)
                    try checkQuality(admittedTokens == [1,7,13,3],"Little-endian corpus")
                    var samples: [NativeQualitySample] = []
                    let result = runNativeQuality(admitted,root:root) { sample in
                        samples.append(sample);try emitQuality("QUALITY_CHECK_SAMPLE_JSON=",sample)
                    }
                    try checkQuality(result.passed && result.closed && result.released && result.samplesCompleted == 3,"Actual quality result: \(String(describing:result.error))")
                    try checkQuality(result.summary?.targets == 3 && result.summary?.blockSize == 3 && samples.map(\.target.targetIndex) == [0,1,2],"Actual target coverage")
                    try checkQuality(result.filesBefore == files && result.filesAfter == files && result.tokensBeforeSHA256 == config.tokensSHA256 && result.tokensAfterSHA256 == config.tokensSHA256,"Endpoint artifact/corpus proofs")
                    try checkQuality(result.checkpoints.map(\.phase) == ["before-artifact-proof","after-artifact-proof","loaded","block-0","unloaded","closed","after-final-artifact-proof"],"Quality memory phases")
                    try emitQuality("QUALITY_CHECK_RESULT_JSON=",result);names.append(label+"-complete")
                }
                let (root,base) = try make("admission")
                let invalidPolicies: [(String,(inout [String:Any]) -> Void)] = [
                    ("version",{ (d: inout [String:Any]) in d["version"] = 2 }),
                    ("overflow-targets",{ d in d["targets"] = Int.max }),
                    ("overflow-block",{ d in d["blockSize"] = Int.max }),
                    ("capacity",{ d in d["capacity"] = 5 }),
                    ("chunk",{ d in d["chunk"] = 2 }),
                    ("budget",{ d in d["processBudget"] = 1 }),
                    ("corpus-path",{ d in d["tokensFile"] = "../corpus.u32le" }),
                    ("corpus-in-model",{ d in d["tokensFile"] = "Model/corpus.u32le" }),
                    ("corpus-hash",{ d in d["tokensSHA256"] = "bad" }),
                    ("inventory",{ d in d["files"] = [:] as [String:String] })]
                for (label,change) in invalidPolicies {
                    let config = try changed(base,change)
                    try rejectsQuality { try config.validate() }
                    if label == "overflow-targets" || label == "overflow-block" {
                        let result = runNativeQuality(config,root:root) { _ in throw NSError(domain:"UnexpectedSample",code:1) }
                        try checkQuality(!result.passed && result.error != nil && result.samplesCompleted == 0,"Invalid config overflowed final result")
                        try emitQuality("QUALITY_CHECK_FAILURE_JSON=",QualityFailureRecord(label:label,result:result))
                    }
                    names.append("reject-"+label)
                }
                do {
                    let raw = try JSONEncoder().encode(base);var object = try JSONSerialization.jsonObject(with:raw) as! [String:Any]
                    object["extra"] = true;let extra = try JSONSerialization.data(withJSONObject:object)
                    try extra.write(to:root.appendingPathComponent("extra.json"),options:.withoutOverwriting)
                    try rejectsQuality { _ = try admitQualityConfig(root:root,name:"extra.json",sha256:dataSHA(extra)) }
                    try raw.write(to:root.appendingPathComponent("wrong-hash.json"),options:.withoutOverwriting)
                    try rejectsQuality { _ = try admitQualityConfig(root:root,name:"wrong-hash.json",sha256:String(repeating:"0",count:64)) }
                    names += ["reject-extra-fields","reject-config-sha"]
                }
                do {
                    let (folder,config) = try make("corpus-oversize")
                    let handle = try FileHandle(forWritingTo:folder.appendingPathComponent("corpus.u32le"));_ = try handle.seekToEnd();try handle.write(contentsOf:Data([0]));try handle.close()
                    try rejectsQuality { _ = try admitQualityTokens(config,root:folder) };names.append("reject-corpus-size")
                }
                do {
                    let (folder,config) = try make("corpus-symlink")
                    try FileManager.default.moveItem(at:folder.appendingPathComponent("corpus.u32le"),to:folder.appendingPathComponent("actual.u32le"))
                    try FileManager.default.createSymbolicLink(atPath:folder.appendingPathComponent("corpus.u32le").path,withDestinationPath:"actual.u32le")
                    try rejectsQuality { _ = try admitQualityTokens(config,root:folder) };names.append("reject-corpus-symlink")
                }
                for kind in ["output","model-endpoint","corpus-endpoint"] {
                    let (folder,config) = try make(kind);var count = 0
                    let result = runNativeQuality(config,root:folder) { _ in
                        if kind == "output" && count == 1 { throw NSError(domain:"InjectedOutput",code:1) }
                        if count == 0 && kind == "model-endpoint" { try Data([0]).write(to:folder.appendingPathComponent("Model/unexpected"),options:.withoutOverwriting) }
                        if count == 0 && kind == "corpus-endpoint" { try Data([0]).write(to:folder.appendingPathComponent("corpus.u32le")) }
                        count += 1
                    }
                    try checkQuality(!result.passed && result.closed && result.released,"Failed quality cleanup: \(kind)")
                    if kind == "output" { try checkQuality(result.error != nil && result.summary == nil && result.samplesCompleted == 1 && count == 1,"Output failure completed a summary") }
                    else { try checkQuality(!result.endpointErrors.isEmpty && result.samplesCompleted == 3,"Endpoint mutation was not rejected") }
                    try emitQuality("QUALITY_CHECK_FAILURE_JSON=",QualityFailureRecord(label:kind,result:result))
                    names.append("reject-"+kind)
                }
                try emitQuality("QUALITY_CHECKS_JSON=",QualityCheckSummary(passed:true,cases:names));exit(0)
            } catch { fputs("Quality checks: \(error)\n",stderr);exit(1) }
        }
        dispatchMain()
    }
}

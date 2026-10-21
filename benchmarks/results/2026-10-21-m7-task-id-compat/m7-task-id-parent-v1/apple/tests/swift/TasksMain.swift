import Foundation
import CryptoKit
import Darwin

private func checkTasks(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw NSError(domain:"TokletTasksChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func rejectsTasks(_ action: () throws -> Void) throws {
    var rejected = false;do { try action() } catch { rejected = true }
    try checkTasks(rejected,"Invalid task artifact admitted")
}
private func tasksSHA(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
private func emitTasks<T: Encodable>(_ marker: String, _ value: T) throws {
    let data = try JSONEncoder().encode(value)
    try FileHandle.standardOutput.write(contentsOf:Data((marker+String(decoding:data,as:UTF8.self)+"\n").utf8))
}
private struct TasksFailureRecord: Encodable { let label: String;let result: NativeTasksResult }
private struct TasksCheckSummary: Encodable { let passed: Bool;let cases: [String];let physicalDeviceQualified = false }
@main private struct TasksMain {
    static func main() {
        DispatchQueue.global(qos:.userInitiated).async {
            do {
                guard CommandLine.arguments.count == 3 else { throw NSError(domain:"TasksArguments",code:1) }
                let fixture = URL(fileURLWithPath:CommandLine.arguments[1],isDirectory:true)
                let cases = URL(fileURLWithPath:CommandLine.arguments[2],isDirectory:true)
                try FileManager.default.createDirectory(at:cases,withIntermediateDirectories:false)
                let files = ["Model/config.json":"84b6e6078f35006f49ae2e2a8663093cdc8d2813e2b7f28ebfa6142113e9c1d9",
                             "Model/model.safetensors":"b01a114bf2d44f0c6cb438035e85535b658e803c76691729facd78b15eefbce2"]
                let taskSet = NativeTaskSet(version:1,vocabulary:64,stops:[62,63],cases:[
                    NativeTaskCase(id:"first",promptTokens:[1,7,13],maximum:3),
                    NativeTaskCase(id:"second",promptTokens:[1,7,13,3],maximum:2)])
                let rawTasks = try JSONEncoder().encode(taskSet)
                var names: [String] = []
                func make(_ name: String, metal: Bool = false) throws -> (URL,NativeTasksConfig) {
                    let root = cases.appendingPathComponent(name,isDirectory:true)
                    try FileManager.default.createDirectory(at:root.appendingPathComponent("Model"),withIntermediateDirectories:true)
                    for key in files.keys {
                        let target = root.appendingPathComponent(key)
                        try FileManager.default.linkItem(at:fixture.appendingPathComponent(target.lastPathComponent),to:target)
                    }
                    try rawTasks.write(to:root.appendingPathComponent("tasks.json"),options:.withoutOverwriting)
                    return (root,NativeTasksConfig(version:1,modelFolder:"Model",packedFolder:"",files:files,metal:metal,
                        capacity:6,chunk:3,numericalBudget:128*1024*1024,processBudget:192*1024*1024,
                        tasksFile:"tasks.json",tasksSHA256:tasksSHA(rawTasks),caseCount:2))
                }
                func changed(_ source: NativeTasksConfig, _ change: (inout [String:Any]) -> Void) throws -> NativeTasksConfig {
                    var object = try JSONSerialization.jsonObject(with:JSONEncoder().encode(source)) as! [String:Any]
                    change(&object);return try JSONDecoder().decode(NativeTasksConfig.self,from:JSONSerialization.data(withJSONObject:object))
                }
                for metal in [false,true] {
                    let label = metal ? "metal":"cpu";let (root,config) = try make(label,metal:metal)
                    let raw = try JSONEncoder().encode(config);try raw.write(to:root.appendingPathComponent("job.json"),options:.withoutOverwriting)
                    let admitted = try admitTasksConfig(root:root,name:"job.json",sha256:tasksSHA(raw))
                    let admittedSet = try admitTaskSet(admitted,root:root)
                    try checkTasks(admittedSet.cases.map(\.id) == ["first","second"],"Ordered task admission")
                    var chunks: [NativeTaskChunkSample] = [];var responses: [NativeTaskCaseSample] = []
                    let result = runNativeTasks(admitted,root:root,onChunk:{ sample in
                        chunks.append(sample);try emitTasks("TASKS_CHECK_CHUNK_JSON=",sample)
                    },onCase:{ sample in
                        responses.append(sample);try emitTasks("TASKS_CHECK_CASE_JSON=",sample)
                    })
                    try checkTasks(result.passed && result.closed && result.released && result.casesCompleted == 2 && result.chunksCompleted == 2,
                        "Actual task result: \(String(describing:result.error))")
                    try checkTasks(chunks.map(\.caseIndex) == [0,1] && responses.map(\.caseID) == ["first","second"],"Case/chunk coverage")
                    try checkTasks(responses[0].response.ids == [47,47,47] && responses[1].response.ids == [47,47],"Independent tiny greedy IDs")
                    try checkTasks(result.filesBefore == files && result.filesAfter == files && result.tasksBeforeSHA256 == config.tasksSHA256
                        && result.tasksAfterSHA256 == config.tasksSHA256 && result.loadedAdmittedBytes > 0,"Endpoint/admission proofs")
                    try checkTasks(result.checkpoints.map(\.phase) == ["before-artifact-proof","after-artifact-proof","loaded","case-0","case-1","unloaded","closed","after-final-artifact-proof"],"Task memory phases")
                    try emitTasks("TASKS_CHECK_RESULT_JSON=",result);names.append(label+"-complete")
                }
                let (root,base) = try make("admission")
                let policies: [(String,(inout [String:Any]) -> Void)] = [
                    ("version",{ d in d["version"] = 2 }),("overflow-count",{ d in d["caseCount"] = Int.max }),
                    ("capacity",{ d in d["capacity"] = 563 }),("chunk",{ d in d["chunk"] = 129 }),
                    ("budget",{ d in d["processBudget"] = 1 }),("tasks-path",{ d in d["tasksFile"] = "../tasks.json" }),
                    ("tasks-in-model",{ d in d["tasksFile"] = "Model/tasks.json" }),("tasks-hash",{ d in d["tasksSHA256"] = "bad" }),
                    ("inventory",{ d in d["files"] = [:] as [String:String] })]
                for (label,change) in policies {
                    let config = try changed(base,change);try rejectsTasks { try config.validate() }
                    if label == "overflow-count" {
                        let result = runNativeTasks(config,root:root,onChunk:{ _ in throw NSError(domain:"UnexpectedChunk",code:1) },onCase:{ _ in throw NSError(domain:"UnexpectedCase",code:1) })
                        try checkTasks(!result.passed && result.error != nil && result.casesCompleted == 0 && result.chunksCompleted == 0,"Invalid count issued native work")
                        try emitTasks("TASKS_CHECK_FAILURE_JSON=",TasksFailureRecord(label:label,result:result))
                    }
                    names.append("reject-"+label)
                }
                do {
                    let raw = try JSONEncoder().encode(base);var object = try JSONSerialization.jsonObject(with:raw) as! [String:Any]
                    object["extra"] = true;let extra = try JSONSerialization.data(withJSONObject:object)
                    try extra.write(to:root.appendingPathComponent("extra.json"),options:.withoutOverwriting)
                    try rejectsTasks { _ = try admitTasksConfig(root:root,name:"extra.json",sha256:tasksSHA(extra)) }
                    try raw.write(to:root.appendingPathComponent("wrong-sha.json"),options:.withoutOverwriting)
                    try rejectsTasks { _ = try admitTasksConfig(root:root,name:"wrong-sha.json",sha256:String(repeating:"0",count:64)) }
                    names += ["reject-extra-config","reject-config-sha"]
                }
                let invalidSets: [(String,(inout [String:Any]) -> Void)] = [
                    ("set-version",{ d in d["version"] = 2 }),("vocabulary",{ d in d["vocabulary"] = 0 }),
                    ("duplicate-stops",{ d in d["stops"] = [62,62] }),("stop-range",{ d in d["stops"] = [64] }),
                    ("case-count",{ d in d["cases"] = [] as [[String:Any]] }),
                    ("extra-set",{ d in d["extra"] = true }),
                    ("duplicate-case",{ d in var rows = d["cases"] as! [[String:Any]];rows[1]["id"] = "first";d["cases"] = rows }),
                    ("case-id",{ d in var rows = d["cases"] as! [[String:Any]];rows[0]["id"] = "bad/id";d["cases"] = rows }),
                    ("overflow-output",{ d in var rows = d["cases"] as! [[String:Any]];rows[0]["maximum"] = Int.max;d["cases"] = rows }),
                    ("prompt-range",{ d in var rows = d["cases"] as! [[String:Any]];rows[0]["promptTokens"] = [64];d["cases"] = rows }),
                    ("prompt-capacity",{ d in var rows = d["cases"] as! [[String:Any]];rows[0]["maximum"] = 4;d["cases"] = rows }),
                    ("extra-case",{ d in var rows = d["cases"] as! [[String:Any]];rows[0]["extra"] = true;d["cases"] = rows })]
                for (label,change) in invalidSets {
                    var object = try JSONSerialization.jsonObject(with:rawTasks) as! [String:Any];change(&object)
                    let raw = try JSONSerialization.data(withJSONObject:object);try raw.write(to:root.appendingPathComponent("tasks.json"))
                    let config = try changed(base,{ $0["tasksSHA256"] = tasksSHA(raw) })
                    try rejectsTasks { _ = try admitTaskSet(config,root:root) };names.append("reject-"+label)
                }
                do {
                    let (folder,config) = try make("task-symlink")
                    try FileManager.default.moveItem(at:folder.appendingPathComponent("tasks.json"),to:folder.appendingPathComponent("actual.json"))
                    try FileManager.default.createSymbolicLink(atPath:folder.appendingPathComponent("tasks.json").path,withDestinationPath:"actual.json")
                    try rejectsTasks { _ = try admitTaskSet(config,root:folder) };names.append("reject-tasks-symlink")
                }
                do {
                    let (folder,config) = try make("task-oversize");let raw = Data(repeating:32,count:512*1024+1)
                    try raw.write(to:folder.appendingPathComponent("tasks.json"));let changedConfig = try changed(config,{ $0["tasksSHA256"] = tasksSHA(raw) })
                    try rejectsTasks { _ = try admitTaskSet(changedConfig,root:folder) };names.append("reject-tasks-size")
                }
                for kind in ["output-chunk","output-case","model-endpoint","tasks-endpoint"] {
                    let (folder,config) = try make(kind);var chunks = 0;var completed = 0
                    let result = runNativeTasks(config,root:folder,onChunk:{ _ in
                        if kind == "output-chunk" && chunks == 1 { throw NSError(domain:"InjectedChunk",code:1) }
                        if chunks == 0 && kind == "model-endpoint" { try Data([0]).write(to:folder.appendingPathComponent("Model/unexpected"),options:.withoutOverwriting) }
                        if chunks == 0 && kind == "tasks-endpoint" { try Data([0]).write(to:folder.appendingPathComponent("tasks.json")) }
                        chunks += 1
                    },onCase:{ _ in
                        if kind == "output-case" && completed == 1 { throw NSError(domain:"InjectedCase",code:1) };completed += 1
                    })
                    try checkTasks(!result.passed && result.closed && result.released,"Failed task cleanup: \(kind)")
                    if kind.hasPrefix("output") {
                        try checkTasks(result.error != nil && result.casesCompleted == 1 && completed == 1
                            && result.chunksCompleted == (kind == "output-chunk" ? 1:2),"Output failure completed a case")
                    } else { try checkTasks(!result.endpointErrors.isEmpty && result.casesCompleted == 2,"Endpoint mutation accepted") }
                    try emitTasks("TASKS_CHECK_FAILURE_JSON=",TasksFailureRecord(label:kind,result:result));names.append("reject-"+kind)
                }
                try emitTasks("TASKS_CHECKS_JSON=",TasksCheckSummary(passed:true,cases:names));exit(0)
            } catch { fputs("Tasks checks: \(error)\n",stderr);exit(1) }
        }
        dispatchMain()
    }
}

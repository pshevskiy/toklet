import Foundation
import CToklet

@main
struct PlanningTypesSmokeMain {
    static func main() throws {
        precondition(CommandLine.arguments.count == 3)
        let fixture = CommandLine.arguments[1], metal = CommandLine.arguments[2] == "metal"
        precondition(MemoryLayout<TokletNativeLimitsOptions>.size == 16)
        precondition(MemoryLayout<TokletRuntimePlanningOptions>.size == 56)
        let limits = try NativeContextLimits.inspect(directory:fixture)
        precondition(limits.status == .compatibleUntested && limits.factoryPositionUpper == 32)
        precondition(limits.backendAvailableInBuild == metal)
        precondition(limits.planningUpper == (metal ? 32 : nil))
        precondition(!limits.qualificationEvaluated && !limits.budgetFitEvaluated)
        let audited = try NativeContextLimits.inspect(directory:fixture,profile:.smolAudited)
        precondition(audited.status == .unsupported && audited.factoryPositionUpper == nil)
        precondition(audited.completeSourceAuthenticationRequired && !audited.auditedSourceMetadataValidated)
        let budget: UInt64 = 128 * 1024 * 1024
        let request = NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:budget,sessionBudget:budget)
        if metal {
            let plan = try NativeCapacityPlan.inspect(directory:fixture,request:request)
            let old = try NativeResourcePlan.inspect(directory:fixture,capacity:32,chunk:3,budget:budget)
            precondition(plan.capacity.selected?.capacity == 32 && plan.capacity.firstExcluded == nil)
            precondition(plan.selectedFitsNumericalBudgets)
            precondition(plan.resources.sessionMemory.admittedBytes == old.sessionMemory.admittedBytes)
            precondition(plan.resources.attention == old.attention && old.attention == .reference)
            let smaller = try NativeResourcePlan.inspect(directory:fixture,capacity:17,chunk:3,budget:budget)
            let bounded = NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,
                loadBudget:budget,sessionBudget:smaller.sessionMemory.admittedBytes)
            let partial = try NativeCapacityPlan.inspect(directory:fixture,request:bounded)
            precondition(partial.capacity.selected?.capacity == 17 && partial.capacity.firstExcluded?.capacity == 18)
            precondition(partial.selectedFitsNumericalBudgets && !partial.resources.fitsSessionBudget)
            let noBudget = NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:budget,sessionBudget:0)
            let none = try NativeCapacityPlan.inspect(directory:fixture,request:noBudget)
            precondition(none.capacity.selected == nil && none.capacity.firstExcluded?.capacity == 3)
            let lowLoad = NativeCapacityPlan.Request(context:32,upper:31,chunk:3,offset:1,loadBudget:1,sessionBudget:budget)
            let loadFailure = try NativeCapacityPlan.inspect(directory:fixture,request:lowLoad)
            precondition(loadFailure.capacity.selected?.capacity == 31 && !loadFailure.selectedFitsNumericalBudgets)
            precondition(!loadFailure.resources.fitsSourceLoadBudget && !loadFailure.resources.fitsResidentLoadBudget)
        } else {
            var rejected = false
            do { _ = try NativeCapacityPlan.inspect(directory:fixture,request:request) }
            catch { rejected = (error as NSError).code == Int(TOKLET_UNSUPPORTED) }
            precondition(rejected)
        }
        // Metadata snapshots are retained values, not live payload identities.
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("toklet-limits-swift-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:false)
        defer { try? FileManager.default.removeItem(at:temp) }
        for name in ["config.json","model.safetensors"] {
            try FileManager.default.copyItem(at:URL(fileURLWithPath:fixture).appendingPathComponent(name),to:temp.appendingPathComponent(name))
        }
        let before = try NativeContextLimits.inspect(directory:temp.path)
        let config = temp.appendingPathComponent("config.json")
        var document = try JSONSerialization.jsonObject(with:Data(contentsOf:config)) as! [String:Any]
        document["max_position_embeddings"] = 8
        try JSONSerialization.data(withJSONObject:document).write(to:config)
        let after = try NativeContextLimits.inspect(directory:temp.path)
        precondition(before.factoryPositionUpper == 32 && after.factoryPositionUpper == 8)
        precondition(limits.factoryPositionUpper == 32)
        print("{\"passed\":true,\"compiledMetal\":\(metal),\"actual_c_producer\":true,\"model_forward\":false,\"qualification_evaluated\":false}")
    }
}

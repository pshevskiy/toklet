import Foundation
import CToklet

private func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try value() == false { throw NSError(domain:"PlanningTypesChecks",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
}
private func rejected(_ body: () throws -> Void) throws {
    var failed = false
    do { try body() } catch { failed = true }
    try require(failed,"Malformed or mismatched planning data accepted")
}
private func data(_ value: [String:Any]) throws -> Data { try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]) }

@main
struct PlanningTypesMain {
    static func main() throws {
        var checks = 0
        func reject(_ body: () throws -> Void) throws { try rejected(body); checks += 1 }
        let limit: [String:Any] = [
            "schema_version":1,"report_kind":"native_mlx_context_limits","load_profile":"ordinary",
            "status":"compatible_untested","reason":"metadata_accepted","message":"Independent wire witness",
            "architecture":"qwen3","declared_total_positions":40960,"engine_global_position_ceiling":32768,
            "graph_position_cap":512,"declared_sliding_window":NSNull(),"factory_position_upper":512,
            "backend_available_in_build":true,"tensor_metadata_validated":true,"audited_source_metadata_validated":false,
            "complete_source_authentication_required":false,"payload_validation_required":true,
            "runtime_device_checks_required":true,"session_admission_required":true,"budget_fit_evaluated":false,
            "qualification_evaluated":false,"text_profile_checked":false,"whole_app_device_fit_evaluated":false,
            "metadata_reader_limit_bytes":UInt64(6)*1024*1024*1024]
        func limits(_ value: [String:Any], _ profile: NativeContextLimits.LoadProfile = .ordinary) throws -> NativeContextLimits {
            try NativeContextLimits.decode(data(value),profile:profile)
        }
        try require(try limits(limit).planningUpper == 512,"Accepted upper")
        var unavailable = limit; unavailable["backend_available_in_build"] = false; unavailable["reason"] = "backend_unavailable"
        try require(try limits(unavailable).planningUpper == nil,"Unavailable backend was promoted")
        var qwen = limit; qwen["architecture"] = "qwen2"; qwen["graph_position_cap"] = NSNull(); qwen["factory_position_upper"] = 32768
        try require(try limits(qwen).planningUpper == 32768,"Invented Qwen2 cap")
        var phi = limit; phi["architecture"] = "phi3"; phi["declared_sliding_window"] = 7; phi["factory_position_upper"] = 7
        try require(try limits(phi).planningUpper == 7,"Lost enforced window")
        var small = limit; small["declared_total_positions"] = 3; small["factory_position_upper"] = 3
        try require(try limits(small).planningUpper == 3,"Lost smaller declaration")
        var audited = limit; audited["load_profile"] = "smol_audited"; audited["complete_source_authentication_required"] = true; audited["audited_source_metadata_validated"] = true
        try require(try limits(audited,.smolAudited).planningUpper == 512,"Audited metadata decoding")
        try reject { _ = try limits(audited) }
        var unknown = limit
        for key in ["architecture","declared_total_positions","graph_position_cap","factory_position_upper"] { unknown[key] = NSNull() }
        unknown["status"] = "unknown"; unknown["reason"] = "unknown_architecture"; unknown["tensor_metadata_validated"] = false
        try require(try limits(unknown).planningUpper == nil,"Unknown upper inferred")
        var unsupported = unknown; unsupported["status"] = "unsupported"; unsupported["reason"] = "artifact_rejected"
        try require(try limits(unsupported).planningUpper == nil,"Rejected upper inferred")
        var badGeometry = limit; badGeometry["status"] = "unsupported"; badGeometry["reason"] = "unsupported_geometry"; badGeometry["graph_position_cap"] = NSNull(); badGeometry["factory_position_upper"] = NSNull()
        try require(try limits(badGeometry).planningUpper == nil,"Geometry rejection promoted")
        for (key,value) in [
            ("schema_version",2),("report_kind","capacity"),("load_profile","future"),("status","future"),
            ("reason","backend_unavailable"),("declared_total_positions",0),("graph_position_cap",0),
            ("factory_position_upper",513),("factory_position_upper",NSNull()),("engine_global_position_ceiling",40960),
            ("metadata_reader_limit_bytes",1),("declared_sliding_window",0),("architecture",""),
            ("tensor_metadata_validated",false),("audited_source_metadata_validated",true),
            ("complete_source_authentication_required",true),("payload_validation_required",false),
            ("runtime_device_checks_required",false),("session_admission_required",false),
            ("budget_fit_evaluated",true),("qualification_evaluated",true),("text_profile_checked",true),
            ("whole_app_device_fit_evaluated",true)
        ] as [(String,Any)] {
            var changed = limit; changed[key] = value
            try reject { _ = try limits(changed) }
        }
        var invented = unknown; invented["factory_position_upper"] = 32768
        try reject { _ = try limits(invented) }
        var falseAudit = audited; falseAudit["audited_source_metadata_validated"] = false
        try reject { _ = try limits(falseAudit,.smolAudited) }

        let request = NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:3000,sessionBudget:3000)
        let memory: [String:Any] = ["weights_bytes":900,"kv_bytes":100,"scratch_bytes":100,"history_bytes":128,
            "logits_bytes":512,"metadata_bytes":24,"metadata_allowance_bytes":16,"admitted_bytes":1780]
        let resources: [String:Any] = [
            "schema_version":1,"device_name":"Independent wire witness","maximum_buffer_bytes":4096,
            "requested_load_context":32,"requested_capacity":32,"requested_chunk_size":3,"requested_position_offset":0,
            "requested_load_budget_bytes":3000,"requested_session_budget_bytes":3000,"projection":"MlxAffine",
            "normalization":"Compensated","readback":"FullLogits","attention":"Reference",
            "source_payload_bytes":1000,"source_admitted_load_bytes":2000,"resident_weights_bytes":900,"resident_admitted_load_bytes":1800,
            "session_memory":memory,"fits_source_load_budget":true,"fits_resident_load_budget":true,
            "fits_session_budget":true,"fits_requested_budgets":true,"numerical_buffers_allocated":false,
            "payload_validation_required":true,"qualification_evaluated":false]
        func point(_ n: UInt32,_ bytes: UInt64,_ largest: UInt64,_ budget: Bool,_ buffer: Bool) -> [String:Any] {
            ["capacity":n,"footprint":["admitted_bytes":bytes,"largest_buffer_bytes":largest],"fits_budget":budget,"fits_buffer":buffer]
        }
        let range: [String:Any] = ["minimum":3,"maximum":32,"limits":["budget_bytes":3000,"maximum_buffer_bytes":4096],
            "selected":point(32,1780,900,true,true),"first_excluded":NSNull()]
        let wire: [String:Any] = ["resources":resources,"capacity":range]
        func capacity(_ value: [String:Any],_ asked: NativeCapacityPlan.Request) throws -> NativeCapacityPlan {
            try NativeCapacityPlan.decode(data(value),request:asked)
        }
        func capacity(_ value: [String:Any]) throws -> NativeCapacityPlan { try capacity(value,request) }
        let old = try NativeResourcePlan.decode(data(resources),capacity:32,chunk:3,budget:3000)
        let full = try capacity(wire)
        try require(full.capacity.selected?.capacity == 32 && full.selectedFitsNumericalBudgets,"Upper fit")
        try require(old.sessionMemory.admittedBytes == full.resources.sessionMemory.admittedBytes,"Shared old validation differs")
        var lowLoad = resources; lowLoad["requested_load_budget_bytes"] = 1; lowLoad["fits_source_load_budget"] = false; lowLoad["fits_resident_load_budget"] = false; lowLoad["fits_requested_budgets"] = false
        let lowRequest = NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:1,sessionBudget:3000)
        let low = try capacity(["resources":lowLoad,"capacity":range],lowRequest)
        try require(low.capacity.selected?.capacity == 32 && !low.selectedFitsNumericalBudgets,"Session fit overrode load failure")
        var partialResources = resources; partialResources["requested_session_budget_bytes"] = 1500; partialResources["fits_session_budget"] = false; partialResources["fits_requested_budgets"] = false
        var partialRange = range; partialRange["limits"] = ["budget_bytes":1500,"maximum_buffer_bytes":4096]
        partialRange["selected"] = point(17,1500,900,true,true); partialRange["first_excluded"] = point(18,1501,900,false,true)
        let partialRequest = NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:3000,sessionBudget:1500)
        let partial = try capacity(["resources":partialResources,"capacity":partialRange],partialRequest)
        try require(partial.capacity.selected?.capacity == 17 && partial.selectedFitsNumericalBudgets,"Selected lower fit lost")
        var noRange = partialRange; noRange["selected"] = NSNull(); noRange["first_excluded"] = point(3,1501,900,false,true)
        try require(try capacity(["resources":partialResources,"capacity":noRange],partialRequest).capacity.selected == nil,"No fit lost")
        var bufferResources = resources; bufferResources["maximum_buffer_bytes"] = 1000
        var bufferRange = range; bufferRange["limits"] = ["budget_bytes":3000,"maximum_buffer_bytes":1000]
        bufferRange["selected"] = point(17,1500,1000,true,true); bufferRange["first_excluded"] = point(18,1501,1001,true,false)
        try require(try capacity(["resources":bufferResources,"capacity":bufferRange]).capacity.selected?.capacity == 17,"Buffer-bound witness lost")
        for (key,value) in [("minimum",4),("maximum",31),("selected",NSNull()),("first_excluded",point(32,1780,900,true,true))] as [(String,Any)] {
            var changed = range; changed[key] = value
            try reject { _ = try capacity(["resources":resources,"capacity":changed]) }
        }
        for bad in [point(31,1780,900,true,true),point(32,1779,900,true,true),point(32,1780,900,false,true),
                    point(32,1780,900,true,false),point(32,1780,0,true,true),point(32,1780,1781,true,true),point(33,1780,900,true,true)] {
            var changed = range; changed["selected"] = bad
            try reject { _ = try capacity(["resources":resources,"capacity":changed]) }
        }
        for bad in [point(19,1501,900,false,true),point(18,1500,900,true,true),point(18,1501,899,false,true)] {
            var changed = partialRange; changed["first_excluded"] = bad
            try reject { _ = try capacity(["resources":partialResources,"capacity":changed],partialRequest) }
        }
        for (key,value) in [("budget_bytes",3001),("maximum_buffer_bytes",4095)] {
            var changed = range, limits = range["limits"] as! [String:Any]; limits[key] = value; changed["limits"] = limits
            try reject { _ = try capacity(["resources":resources,"capacity":changed]) }
        }
        for (key,value) in [("requested_load_context",31),("requested_capacity",31),("requested_position_offset",1),
                            ("requested_chunk_size",4),("requested_session_budget_bytes",3001),("requested_load_budget_bytes",3001),
                            ("normalization","Original"),("readback","GreedyToken"),("projection","Half"),
                            ("numerical_buffers_allocated",true),("payload_validation_required",false),("qualification_evaluated",true),
                            ("fits_source_load_budget",false),("fits_resident_load_budget",false),("fits_session_budget",false),("fits_requested_budgets",false)] as [(String,Any)] {
            var changed = resources; changed[key] = value
            try reject { _ = try capacity(["resources":changed,"capacity":range]) }
        }
        var overflowMemory = memory; overflowMemory["scratch_bytes"] = UInt64.max
        var overflowResources = resources; overflowResources["session_memory"] = overflowMemory
        try reject { _ = try capacity(["resources":overflowResources,"capacity":range]) }
        var explicitOnline = request; explicitOnline.attention = .online
        try reject { _ = try capacity(wire,explicitOnline) }
        let staleRequests = [
            NativeCapacityPlan.Request(context:32,upper:31,chunk:3,offset:1,loadBudget:3000,sessionBudget:3000),
            NativeCapacityPlan.Request(context:32,upper:32,chunk:4,offset:0,loadBudget:3000,sessionBudget:3000),
            NativeCapacityPlan.Request(context:33,upper:32,chunk:3,offset:0,loadBudget:3000,sessionBudget:3000)]
        for stale in staleRequests { try reject { _ = try capacity(wire,stale) } }
        for invalid in [
            NativeCapacityPlan.Request(context:0,upper:1,chunk:1,offset:0,loadBudget:1,sessionBudget:0),
            NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:UInt32.max,loadBudget:1,sessionBudget:0),
            NativeCapacityPlan.Request(context:32,upper:32,chunk:33,offset:0,loadBudget:1,sessionBudget:0),
            NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:0,sessionBudget:0),
            NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:1,sessionBudget:UInt64.max)] {
            try reject { try invalid.validate() }
        }
        var singleResources = resources
        singleResources["requested_load_context"] = 1; singleResources["requested_capacity"] = 1; singleResources["requested_chunk_size"] = 1
        var singleRange = range; singleRange["minimum"] = 1; singleRange["maximum"] = 1; singleRange["selected"] = point(1,1780,900,true,true)
        let singleRequest = NativeCapacityPlan.Request(context:1,upper:1,chunk:1,offset:0,loadBudget:3000,sessionBudget:3000)
        try require(try capacity(["resources":singleResources,"capacity":singleRange],singleRequest).capacity.selected?.capacity == 1,"Single-point interval")
        var zeroResources = partialResources; zeroResources["requested_session_budget_bytes"] = 0
        var zeroRange = noRange; zeroRange["limits"] = ["budget_bytes":0,"maximum_buffer_bytes":4096]
        let zeroRequest = NativeCapacityPlan.Request(context:32,upper:32,chunk:3,offset:0,loadBudget:3000,sessionBudget:0)
        try require(try capacity(["resources":zeroResources,"capacity":zeroRange],zeroRequest).capacity.selected == nil,"Zero session budget must report no fit")
        var explicitRequest = request; explicitRequest.normalization = .original; explicitRequest.readback = .greedyToken; explicitRequest.attention = .online
        var explicitResources = resources; explicitResources["normalization"] = "Original"; explicitResources["readback"] = "GreedyToken"; explicitResources["attention"] = "Online"
        try require(try capacity(["resources":explicitResources,"capacity":range],explicitRequest).capacity.selected?.capacity == 32,"Explicit modes must match rather than be silently replaced")
        var scaledResources = resources, scaledMemory = memory, scaledRange = range
        scaledResources["positional_resident_bytes"] = 120; scaledMemory["positional_bytes"] = 120; scaledMemory["admitted_bytes"] = 1900
        scaledResources["session_memory"] = scaledMemory; scaledRange["selected"] = point(32,1900,900,true,true)
        let scaled = try capacity(["resources":scaledResources,"capacity":scaledRange])
        try require(scaled.resources.sessionMemory.sum == 1900,"Positional owner counted once")
        scaledMemory["positional_bytes"] = 121; scaledMemory["admitted_bytes"] = 1901; scaledResources["session_memory"] = scaledMemory
        try reject { _ = try capacity(["resources":scaledResources,"capacity":scaledRange]) }
        for malformed in [Data(),Data("{".utf8),Data(repeating:0,count:65537)] {
            try reject { _ = try NativeCapacityPlan.decode(malformed,request:request) }
            try reject { _ = try NativeContextLimits.decode(malformed,profile:.ordinary) }
        }
        print("{\"passed\":true,\"rejected_controls\":\(checks),\"model_executed\":false,\"gpu_executed\":false}")
    }
}

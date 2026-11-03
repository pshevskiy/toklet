import Foundation
import CToklet

@main
@MainActor
struct LlamaCatalogMain {
    static func main() throws {
        var checks = 0
        func check(_ value: Bool, _ message: String) throws {
            checks += 1
            if !value { throw NativeContextFailure(message) }
        }
        func rejects(_ body: () throws -> Void) throws {
            var failed = false; do { try body() } catch { failed = true }
            try check(failed,"Expected rejection")
        }
        #if os(macOS)
        let enabled = true
        #else
        let enabled = false
        #endif
        try check(HubProfile.catalog.contains(.llama3) == enabled,"Platform catalog admission")
        try check(HubProfile.catalog.count == (enabled ? 5 : 4),"Platform catalog size")
        try check((HubProfile.named("llama3") != nil) == enabled,"Named lookup bypassed gate")
        try check(!HubProfile.matching("Llama-3.2-1B").isEmpty == enabled,"Search bypassed gate")
        #if os(macOS)
        try check(try FriendlyLoadRequest.managedMode(.llama3) == .managedLlama3,"Audited route missing")
        #else
        try rejects { _ = try FriendlyLoadRequest.managedMode(.llama3) }
        #endif
        try check(FriendlyLoadRequest.matches(.managedLlama3,profile:.llama3),"Exact route mismatch")
        try check(!FriendlyLoadRequest.matches(.ordinary,profile:.llama3),"Ordinary folder path admitted Llama")
        try check(!FriendlyLoadRequest.matches(.managedLlama3,profile:.smol),"Llama family confused with artifact")
        try check(HubProfile.llama3.supportNotice != nil,"Bounded-window notice missing")
        try HubProfile.llama3.validate()
        let profile = NativeAcceptedIdentity.Profile.llama3
        try check(profile.admits(loaded:512,ranges:374,reads:374),"512 identity rejected")
        try check(!profile.admits(loaded:513,ranges:374,reads:374),"Factory upper expanded")
        var context = TokletContextInfo()
        context.abi_version = TOKLET_ABI_VERSION; context.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        context.model_instance = 7; context.session_instance = 8; context.backend = TOKLET_CONTEXT_METAL
        context.engine_global_position_ceiling = 32768; context.loaded_position_limit = 512
        context.session_capacity = 512; context.numerical_and_bridge_admitted_bytes = 1000
        let model = NativeModelContext(try NativeContextFacts(context))
        let n = profile.namespace
        let roles = [["source_metadata_consumed","codec_config_verified"],["source_bytes_delivered_to_assembled_model"],
                     ["source_metadata_consumed"],["tokenizer_parser_consumed"],["fixed_text_policy_verified"],["ancillary_verified"]]
        let assets: [[String:Any]] = HubProfile.llama3.files.enumerated().map { i,file in
            ["name":file.name,"bytes":file.bytes,"sha256":file.sha256,"roles":roles[i]]
        }
        let wire: [String:Any] = ["schema_version":1,"report_kind":"accepted_model_chat_identity",
            "repository":profile.repository,"revision":profile.revision,
            "preference_namespace":["version":1,"manifest_algorithm":n.manifestAlgorithm,"manifest_sha256":n.manifestSHA256,
                "backend":n.backend,"load":n.load,"text":n.text,"planning":n.planning],
            "assets":assets,"model_instance":7,"loaded_position_limit":512,
            "source_container_physical_ranges":374,"source_container_verified_reads":374,
            "model_assembled":true,"codec_constructed":true,"exact_model_pair":true,"source_receipts_equal":true,
            "historical_evidence":true,"current_disk_identity_evaluated":false,"resident_buffer_identity_evaluated":false,
            "qualification_evaluated":false,"whole_app_device_fit_evaluated":false,"preferences_require_fresh_admission":true]
        func decode(_ value: [String:Any]) throws -> NativeAcceptedIdentity {
            try NativeAcceptedIdentity.decode(JSONSerialization.data(withJSONObject:value),context:model)
        }
        try check(try decode(wire).profile == .llama3,"Pinned identity rejected")
        for i in assets.indices {
            var changed = assets; changed[i]["sha256"] = String(repeating:"0",count:64)
            var value = wire; value["assets"] = changed
            try rejects { _ = try decode(value) }
        }
        var wrong = wire; wrong["loaded_position_limit"] = 513
        try rejects { _ = try decode(wrong) }
        wrong = wire; wrong["revision"] = HubProfile.smol.revision
        try rejects { _ = try decode(wrong) }
        let keys = try NativeAcceptedIdentity.Profile.allCases.map { try FriendlyModelPreference.filename($0.namespace) }
        try check(Set(keys).count == 5,"Preference namespaces collide")
        let preference = FriendlyModelPreference(namespace:n,capacity:256,chunk:16,reply:32,wholeBudget:HubProfile.llama3.numericalBudget)
        let bytes = try preference.encoded()
        try check(try FriendlyModelPreference.decode(bytes,namespace:n) == preference,"Preference roundtrip")
        try rejects { _ = try FriendlyModelPreference.decode(bytes,namespace:NativeAcceptedIdentity.Profile.smol.namespace) }
        print("LLAMA_CATALOG checks=\(checks) enabled=\(enabled)")
    }
}

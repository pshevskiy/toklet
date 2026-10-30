import Foundation
import CToklet

// Fixed independent wire witness. Never a native accepted-model fixture.
@main
struct AcceptedIdentityMain {
    static func context(model: UInt64 = 7, backend: UInt32 = TOKLET_CONTEXT_METAL,
                        loaded: UInt64 = 512, capacity: UInt64 = 256) throws -> NativeModelContext {
        var value = TokletContextInfo()
        value.abi_version = TOKLET_ABI_VERSION; value.struct_size = UInt32(MemoryLayout<TokletContextInfo>.size)
        value.model_instance = model; value.session_instance = 8; value.backend = backend
        value.engine_global_position_ceiling = 32768; value.loaded_position_limit = loaded
        value.session_capacity = capacity; value.numerical_and_bridge_admitted_bytes = 1000
        return NativeModelContext(try NativeContextFacts(value))
    }
    static func main() throws {
        var checks = 0
        func require(_ value: Bool, _ message: String) throws {
            checks += 1
            if !value { throw NativeContextFailure(message) }
        }
        func rejected(_ body: () throws -> Void) throws {
            var failed = false; do { try body() } catch { failed = true }
            try require(failed,"Malformed or mismatched identity accepted")
        }
        func data(_ object: [String:Any]) throws -> Data { try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys]) }
        let assets: [[String:Any]] = [
            ["name":"config.json","bytes":937,"sha256":"9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240","roles":["source_metadata_consumed","codec_config_verified"]],
            ["name":"model.safetensors","bytes":633442994,"sha256":"3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397","roles":["source_bytes_delivered_to_assembled_model"]],
            ["name":"model.safetensors.index.json","bytes":49731,"sha256":"45addf7779cdbd5af18d92f9f7b79c188f6ebbf32cff7b4ae7dbe68a91340ebc","roles":["source_metadata_consumed"]],
            ["name":"tokenizer.json","bytes":11422654,"sha256":"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4","roles":["tokenizer_parser_consumed"]],
            ["name":"tokenizer_config.json","bytes":10219,"sha256":"8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6","roles":["fixed_text_policy_verified"]],
            ["name":"special_tokens_map.json","bytes":613,"sha256":"76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd","roles":["ancillary_verified"]]
        ]
        let namespace: [String:Any] = ["version":1,"manifest_algorithm":"toklet-full-manifest-v1",
            "manifest_sha256":"5e9aafa305e0e48182a856cafe3ae2b597599033e61b16664f42b09a4af5ff51",
            "backend":"metal_native_dense_v1","load":"qwen3_audited_v1","text":"qwen3-plain-nonthinking-v1","planning":"native_mlx_planning_v1"]
        let wire: [String:Any] = ["schema_version":1,"report_kind":"accepted_model_chat_identity",
            "repository":"mlx-community/Qwen3-0.6B-8bit","revision":"11de96878523501bcaa86104e3c186de07ff9068",
            "preference_namespace":namespace,"assets":assets,"model_instance":7,"loaded_position_limit":512,
            "source_container_physical_ranges":706,"source_container_verified_reads":706,
            "model_assembled":true,"codec_constructed":true,"exact_model_pair":true,"source_receipts_equal":true,
            "historical_evidence":true,"current_disk_identity_evaluated":false,"resident_buffer_identity_evaluated":false,
            "qualification_evaluated":false,"whole_app_device_fit_evaluated":false,"preferences_require_fresh_admission":true]
        let model = try context()
        func decode(_ object: [String:Any]) throws -> NativeAcceptedIdentity { try NativeAcceptedIdentity.decode(data(object),context:model) }
        let accepted = try decode(wire)
        try require(accepted.modelInstance == 7 && accepted.loadedLimit == 512 && model.configuredCapacity == 256 && accepted.assets.count == 6,"Valid model-scoped identity rejected")
        var reverse = wire; reverse["assets"] = Array(assets.reversed())
        try require(try decode(reverse) == accepted,"Canonical manifest depends on asset order")
        try require(accepted.matches(try context(capacity:32)),"Session capacity was confused with model identity")
        for (key,value) in [("schema_version",2),("report_kind","future"),("repository","other/repository"),
                            ("revision",String(repeating:"a",count:40)),("model_instance",0),("model_instance",9),
                            ("model_instance",true),("loaded_position_limit",513),("loaded_position_limit",256),
                            ("source_container_physical_ranges",2),("source_container_physical_ranges",4097),
                            ("source_container_verified_reads",705),("future",true)] as [(String,Any)] {
            var wrong = wire; wrong[key] = value; try rejected { _ = try decode(wrong) }
        }
        for key in ["model_assembled","codec_constructed","exact_model_pair","source_receipts_equal","historical_evidence","preferences_require_fresh_admission"] {
            var wrong = wire; wrong[key] = false; try rejected { _ = try decode(wrong) }
        }
        for key in ["current_disk_identity_evaluated","resident_buffer_identity_evaluated","qualification_evaluated","whole_app_device_fit_evaluated"] {
            var wrong = wire; wrong[key] = true; try rejected { _ = try decode(wrong) }
        }
        var numericFlag = wire; numericFlag["model_assembled"] = 1; try rejected { _ = try decode(numericFlag) }
        var missing = wire; missing.removeValue(forKey:"codec_constructed"); try rejected { _ = try decode(missing) }
        for (key,value) in [("version",2),("manifest_algorithm","future"),("manifest_sha256",String(repeating:"a",count:64)),
                            ("backend","cpu"),("load","ordinary"),("text","other"),("planning","v2")] as [(String,Any)] {
            var wrong = wire, policy = namespace; policy[key] = value; wrong["preference_namespace"] = policy
            try rejected { _ = try decode(wrong) }
        }
        for (key,value) in [("name","unknown"),("bytes",0),("bytes",938),("sha256",String(repeating:"A",count:64)),
                            ("roles",["ancillary_verified"]),("future",true)] as [(String,Any)] {
            var wrong = wire, rows = assets; rows[0][key] = value; wrong["assets"] = rows
            try rejected { _ = try decode(wrong) }
        }
        var duplicate = wire, rows = assets; rows[1] = rows[0]; duplicate["assets"] = rows
        try rejected { _ = try decode(duplicate) }
        var absent = wire; absent["assets"] = Array(assets.dropLast()); try rejected { _ = try decode(absent) }
        for other in [try context(model:9),try context(backend:TOKLET_CONTEXT_CPU),try context(loaded:256)] {
            try rejected { _ = try NativeAcceptedIdentity.decode(data(wire),context:other) }
        }
        let text = String(decoding:try data(wire),as:UTF8.self)
        for altered in [text.replacingOccurrences(of:"\"model_instance\":7",with:"\"model_instance\":7,\"model_instance\":7"),
                        text.replacingOccurrences(of:"\"version\":1",with:"\"version\":1,\"version\":1"),
                        text.replacingOccurrences(of:"\"model_instance\":7",with:"\"model_instance\":7.0"),
                        text.replacingOccurrences(of:"\"model_instance\":7",with:"\"model_instance\":18446744073709551616")] {
            try rejected { _ = try NativeAcceptedIdentity.decode(Data(altered.utf8),context:model) }
        }
        try rejected { _ = try NativeAcceptedIdentity.decode(Data(repeating:32,count:65537),context:model) }
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:["checks":checks,"passed":true,"actual_model_getter":false],options:[.sortedKeys]) + Data([10]))
    }
}

import Foundation
import CryptoKit
import CToklet

// Requested route only. A catalog selection is not consumed-byte evidence.
// Ordinary remains the default for Expert, external folders and other profiles.
enum NativeLoadIdentity: String, Sendable, Equatable {
    case ordinary
    case managedQwen3 = "managed_qwen3_audited_v1"
}

// Historical immutable model/codec evidence; not session/history or disk authority.
struct NativeAcceptedIdentity: Equatable, Sendable {
    struct Namespace: Equatable, Sendable {
        let version: UInt64
        let manifestAlgorithm, manifestSHA256, backend, load, text, planning: String
    }
    struct Asset: Equatable, Sendable {
        let name: String
        let bytes: UInt64
        let sha256: String
        let roles: [String]
    }
    static let repository = "mlx-community/Qwen3-0.6B-8bit"
    static let revision = "11de96878523501bcaa86104e3c186de07ff9068"
    static let manifestSHA256 = "5e9aafa305e0e48182a856cafe3ae2b597599033e61b16664f42b09a4af5ff51"
    static let requiredApplicationAllowance = UInt64(TOKLET_COMPATIBILITY_JSON_ALLOWANCE_BYTES)
    let namespace: Namespace
    let assets: [Asset]
    let modelInstance, loadedLimit, sourceRanges, sourceReads: UInt64

    func matches(_ context: NativeModelContext) -> Bool {
        context.modelInstance == modelInstance && context.backend == TOKLET_CONTEXT_METAL
            && context.loadedLimit == loadedLimit
        // configuredCapacity/session instance may change while model+codec remain.
    }
    private static func failure(_ message: String, code: UInt32 = TOKLET_INVALID) -> NSError {
        NSError(domain:"TokletAcceptedIdentity",code:Int(code),userInfo:[NSLocalizedDescriptionKey:message])
    }
    private static func canonical(_ assets: [Asset]) throws -> String {
        var data = Data("toklet-full-manifest-v1\0".utf8)
        func integer(_ value: UInt64) { var value = value.littleEndian; withUnsafeBytes(of:&value) { data.append(contentsOf:$0) } }
        func text(_ value: String) { integer(UInt64(value.utf8.count)); data.append(contentsOf:value.utf8) }
        text(repository); text(revision); integer(UInt64(assets.count))
        for asset in assets.sorted(by:{ $0.name.utf8.lexicographicallyPrecedes($1.name.utf8) }) {
            text(asset.name); integer(asset.bytes)
            let bytes = Array(asset.sha256.utf8)
            func digit(_ value: UInt8) -> UInt8 { value <= 57 ? value - 48 : value - 87 }
            for index in stride(from:0,to:64,by:2) { data.append(digit(bytes[index]) * 16 + digit(bytes[index+1])) }
        }
        return SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined()
    }
    static func decode(_ data: Data, context: NativeModelContext) throws -> Self {
        let object = try GenerationJSON.object(data,maximumBytes:Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES),maximumStringBytes:128)
        try GenerationJSON.keys(object,["schema_version","report_kind","repository","revision","preference_namespace","assets",
            "model_instance","loaded_position_limit","source_container_physical_ranges","source_container_verified_reads",
            "model_assembled","codec_constructed","exact_model_pair","source_receipts_equal","historical_evidence",
            "current_disk_identity_evaluated","resident_buffer_identity_evaluated","qualification_evaluated",
            "whole_app_device_fit_evaluated","preferences_require_fresh_admission"])
        guard try GenerationJSON.integer(object["schema_version"]) == 1,
              try GenerationJSON.string(object["report_kind"],maximumBytes:64) == "accepted_model_chat_identity",
              try GenerationJSON.string(object["repository"],maximumBytes:128) == repository,
              try GenerationJSON.string(object["revision"],maximumBytes:40) == revision else { throw failure("Unsupported accepted identity schema or artifact") }
        for key in ["model_assembled","codec_constructed","exact_model_pair","source_receipts_equal","historical_evidence","preferences_require_fresh_admission"] {
            guard try GenerationJSON.boolean(object[key]) else { throw failure("Incomplete accepted identity evidence") }
        }
        for key in ["current_disk_identity_evaluated","resident_buffer_identity_evaluated","qualification_evaluated","whole_app_device_fit_evaluated"] {
            guard try GenerationJSON.boolean(object[key]) == false else { throw failure("Unsupported identity qualification claim") }
        }
        let policy = try GenerationJSON.dictionary(object["preference_namespace"])
        try GenerationJSON.keys(policy,["version","manifest_algorithm","manifest_sha256","backend","load","text","planning"])
        let namespace = try Namespace(version:GenerationJSON.integer(policy["version"]),
            manifestAlgorithm:GenerationJSON.string(policy["manifest_algorithm"],maximumBytes:64),
            manifestSHA256:GenerationJSON.hex(policy["manifest_sha256"],count:64),
            backend:GenerationJSON.string(policy["backend"],maximumBytes:64),load:GenerationJSON.string(policy["load"],maximumBytes:64),
            text:GenerationJSON.string(policy["text"],maximumBytes:64),planning:GenerationJSON.string(policy["planning"],maximumBytes:64))
        guard namespace.version == 1, namespace.manifestAlgorithm == "toklet-full-manifest-v1",
              namespace.manifestSHA256 == manifestSHA256, namespace.backend == "metal_native_dense_v1",
              namespace.load == "qwen3_audited_v1", namespace.text == "qwen3-plain-nonthinking-v1",
              namespace.planning == "native_mlx_planning_v1" else { throw failure("Unsupported identity policy") }
        let roles: [String:[String]] = ["config.json":["source_metadata_consumed","codec_config_verified"],
            "model.safetensors":["source_bytes_delivered_to_assembled_model"],
            "model.safetensors.index.json":["source_metadata_consumed"],"tokenizer.json":["tokenizer_parser_consumed"],
            "tokenizer_config.json":["fixed_text_policy_verified"],"special_tokens_map.json":["ancillary_verified"]]
        guard let rows = object["assets"] as? [Any], rows.count == 6 else { throw failure("Incomplete identity asset table") }
        var assets: [Asset] = [], names = Set<String>(), total: UInt64 = 0
        for row in rows {
            let value = try GenerationJSON.dictionary(row); try GenerationJSON.keys(value,["name","bytes","sha256","roles"])
            let name = try GenerationJSON.string(value["name"],maximumBytes:64)
            let bytes = try GenerationJSON.positive(value["bytes"])
            let digest = try GenerationJSON.hex(value["sha256"],count:64)
            guard let actualRoles = value["roles"] as? [String], let expectedRoles = roles[name], actualRoles == expectedRoles,
                  names.insert(name).inserted, bytes <= 4 * 1024 * 1024 * 1024 else { throw failure("Invalid identity asset or role") }
            let sum = total.addingReportingOverflow(bytes)
            guard !sum.overflow, sum.partialValue <= 6 * 1024 * 1024 * 1024 else { throw failure("Identity inventory exceeds bounds") }; total = sum.partialValue
            assets.append(Asset(name:name,bytes:bytes,sha256:digest,roles:actualRoles))
        }
        guard names == Set(roles.keys), try canonical(assets) == manifestSHA256 else { throw failure("Complete consumed manifest differs") }
        let result = try Self(namespace:namespace,assets:assets.sorted(by:{ $0.name < $1.name }),
            modelInstance:GenerationJSON.positive(object["model_instance"]),loadedLimit:GenerationJSON.positive(object["loaded_position_limit"]),
            sourceRanges:GenerationJSON.integer(object["source_container_physical_ranges"]),sourceReads:GenerationJSON.integer(object["source_container_verified_reads"]))
        guard (1...512).contains(result.loadedLimit), (3...4096).contains(result.sourceRanges), result.sourceReads >= result.sourceRanges,
              result.matches(context) else { throw failure("Identity does not match the accepted model owner") }
        return result
    }
    static func inspect(model: OpaquePointer, chat: OpaquePointer, context: NativeModelContext) throws -> Self {
        var owner: OpaquePointer?, error = TokletError()
        func checked(_ code: UInt32) throws {
            guard code != TOKLET_OK else { return }
            let message = withUnsafeBytes(of:&error.message) { String(decoding:$0.prefix { $0 != 0 },as:UTF8.self) }
            throw failure(message.isEmpty ? "Accepted identity query failed" : message,code:code)
        }
        try checked(toklet_model_chat_identity_create(model,chat,&owner,&error))
        guard let owner else { throw failure("Missing accepted identity snapshot") }
        defer { _ = toklet_mlx_inspect_release(owner,nil) }
        var count = 0; try checked(toklet_mlx_inspect_json(owner,nil,0,&count,&error))
        guard (1...Int(TOKLET_MAX_COMPATIBILITY_JSON_BYTES)).contains(count) else { throw failure("Invalid identity snapshot length") }
        var bytes = [UInt8](repeating:0,count:count)
        try bytes.withUnsafeMutableBufferPointer { try checked(toklet_mlx_inspect_json(owner,$0.baseAddress,$0.count,&count,&error)) }
        guard count == bytes.count else { throw failure("Identity snapshot length changed") }
        return try decode(Data(bytes),context:context)
    }
}

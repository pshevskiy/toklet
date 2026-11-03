import Foundation
import CryptoKit
import CToklet

// Requested route only. A catalog selection is not consumed-byte evidence.
// Ordinary remains the default for Expert, external folders and other profiles.
enum NativeLoadIdentity: String, Sendable, Equatable {
    case ordinary
    case managedQwen2 = "managed_qwen2_audited_v1"
    case managedQwen3 = "managed_qwen3_audited_v1"
    case managedSmol = "managed_smol_audited_v1"
    case managedMistral = "managed_mistral_audited_v1"
    case managedLlama3 = "managed_llama3_audited_v1"
    var identityProfile: NativeAcceptedIdentity.Profile? {
        switch self { case .ordinary: nil; case .managedQwen2: .qwen2; case .managedQwen3: .qwen3; case .managedSmol: .smol; case .managedMistral: .mistral; case .managedLlama3: .llama3 }
    }
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
    // A concrete engine-policy allowlist, shared by wire decoding and preferences.
    // Profile recognition selects a route; only the native report proves pairing.
    enum Profile: CaseIterable, Equatable, Sendable {
        case qwen2, qwen3, smol, mistral, llama3
        var repository: String {
            switch self { case .qwen2: "mlx-community/Qwen2.5-0.5B-Instruct-8bit"; case .qwen3: "mlx-community/Qwen3-0.6B-8bit"; case .smol: "mlx-community/SmolLM2-135M-Instruct-8bit"; case .mistral: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"; case .llama3: "mlx-community/Llama-3.2-1B-Instruct-8bit" }
        }
        var revision: String {
            switch self { case .qwen2: "6474d4c1a05bd7bd634a9c993560e4e7272eda3d"; case .qwen3: "11de96878523501bcaa86104e3c186de07ff9068"; case .smol: "0f0d9b8218915bc34d401e1a340b8c049d300d5e"; case .mistral: "a4b8f870474b0eb527f466a03fbc187830d271f5"; case .llama3: "d48cdf0a4ea22d893b7c63a99d6a693e24822795" }
        }
        var manifestSHA256: String {
            switch self { case .qwen2: "e92e6d359697d8ada4714ff4f5ff618ba8d9fc69d7abc24bfb4ec905eae6c59b"; case .qwen3: "5e9aafa305e0e48182a856cafe3ae2b597599033e61b16664f42b09a4af5ff51"; case .smol: "16902411a6306407a5d330e28bc64edb3cbec3961c9db0dbb513176e8e1cb9ce"; case .mistral: "0f543bc8679995fe028857674e6e327d5baa6832453587adda92b0c45e4f4a26"; case .llama3: "61bd958bbd437cdce22122d01297318a672c605f000fedd6bb53af9ddd8ce5ff" }
        }
        var namespace: Namespace {
            let load: String, text: String
            switch self {
            case .qwen2: load = "qwen2_audited_v1"; text = "qwen-chatml-v1"
            case .qwen3: load = "qwen3_audited_v1"; text = "qwen3-plain-nonthinking-v1"
            case .smol: load = "smol_witness_audited_v1"; text = "smol-chatml-v1"
            case .mistral: load = "mistral_audited_v1"; text = "mistral-instruct-v1"
            case .llama3: load = "llama3_audited_v1"; text = "llama3-plain-20240726-v1"
            }
            return Namespace(version:1,manifestAlgorithm:"toklet-full-manifest-v1",manifestSHA256:manifestSHA256,
                backend:"metal_native_dense_v1",load:load,text:text,planning:"native_mlx_planning_v1")
        }
        var modelRoles: [String] {
            switch self {
            case .qwen2, .qwen3, .mistral, .llama3: ["source_bytes_delivered_to_assembled_model"]
            case .smol: ["source_bytes_delivered_to_assembled_model","reference_container_digest_anchor"]
            }
        }
        func admits(loaded: UInt64, ranges: UInt64, reads: UInt64) -> Bool {
            // Wire-policy bounds only; actual loaded context must also match.
            // Qwen2 uses the global ceiling, not the Qwen3/Smol/Mistral factory cap.
            let upper: UInt64 = self == .qwen2 ? 32768 : 512
            return (1...upper).contains(loaded) && reads >= ranges
                && (self == .smol ? ranges == 694 : (3...4096).contains(ranges))
        }
        static func matching(_ namespace: Namespace) -> Self? { allCases.first { $0.namespace == namespace } }
    }
    // Keep the original Qwen constants for existing witnesses/key regressions.
    static let repository = Profile.qwen3.repository
    static let revision = Profile.qwen3.revision
    static let manifestSHA256 = Profile.qwen3.manifestSHA256
    static let requiredApplicationAllowance = UInt64(TOKLET_COMPATIBILITY_JSON_ALLOWANCE_BYTES)
    var profile: Profile? { Profile.matching(namespace) }
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
    private static func canonical(_ assets: [Asset], profile: Profile) throws -> String {
        var data = Data("toklet-full-manifest-v1\0".utf8)
        func integer(_ value: UInt64) { var value = value.littleEndian; withUnsafeBytes(of:&value) { data.append(contentsOf:$0) } }
        func text(_ value: String) { integer(UInt64(value.utf8.count)); data.append(contentsOf:value.utf8) }
        text(profile.repository); text(profile.revision); integer(UInt64(assets.count))
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
              try GenerationJSON.string(object["report_kind"],maximumBytes:64) == "accepted_model_chat_identity" else { throw failure("Unsupported accepted identity schema") }
        let repository = try GenerationJSON.string(object["repository"],maximumBytes:128)
        let revision = try GenerationJSON.string(object["revision"],maximumBytes:40)
        guard let profile = Profile.allCases.first(where:{ $0.repository == repository && $0.revision == revision }) else {
            throw failure("Unsupported accepted identity artifact")
        }
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
        guard namespace == profile.namespace else { throw failure("Unsupported identity policy") }
        let roles: [String:[String]] = ["config.json":["source_metadata_consumed","codec_config_verified"],
            "model.safetensors":profile.modelRoles,
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
        guard names == Set(roles.keys), try canonical(assets,profile:profile) == profile.manifestSHA256 else { throw failure("Complete consumed manifest differs") }
        let result = try Self(namespace:namespace,assets:assets.sorted(by:{ $0.name < $1.name }),
            modelInstance:GenerationJSON.positive(object["model_instance"]),loadedLimit:GenerationJSON.positive(object["loaded_position_limit"]),
            sourceRanges:GenerationJSON.integer(object["source_container_physical_ranges"]),sourceReads:GenerationJSON.integer(object["source_container_verified_reads"]))
        guard profile.admits(loaded:result.loadedLimit,ranges:result.sourceRanges,reads:result.sourceReads),
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

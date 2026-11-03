import Foundation
import CryptoKit

// Explicit native half/MLX8 scores aligned to original-reference raw corpus IDs.
@main
struct LlamaOriginalScoreMain {
    struct Case: Decodable { let prefixTokens, logitRow, targetPosition: Int; let targetId: UInt32 }
    struct Inputs: Decodable { let format, tokensSha256: String; let tokenBytes: Int; let cases: [Case] }
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func sha(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    static func main() throws {
        guard CommandLine.arguments.count == 5 else { throw NativeContextFailure("Expected model, inputs, manifest SHA and fresh report") }
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), input = URL(fileURLWithPath:CommandLine.arguments[2])
        let manifest = try Data(contentsOf:input.appendingPathComponent("manifest.json"))
        try require(sha(manifest) == CommandLine.arguments[3],"Frozen manifest differs")
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let specification = try decoder.decode(Inputs.self,from:manifest)
        let raw = try Data(contentsOf:input.appendingPathComponent("tokens.u32le"))
        try require(specification.format == "toklet-llama3-original-short-v1" && raw.count == 72
            && specification.tokenBytes == raw.count && sha(raw) == specification.tokensSha256,"Frozen input bytes differ")
        let ids: [UInt32] = stride(from:0,to:raw.count,by:4).map { i in
            UInt32(raw[i]) | UInt32(raw[i+1]) << 8 | UInt32(raw[i+2]) << 16 | UInt32(raw[i+3]) << 24
        }
        try require(ids.allSatisfy { $0 < 128256 } && specification.cases.map(\.prefixTokens) == [1,4,8,17],"Frozen cases differ")
        let worker = try NativeWorker(captureSavedPrefix:true)
        var rows: [[String:Any]] = [], failure: Error?
        func execute(_ command: NativeCommand) throws -> NativeSnapshot {
            try require(worker.submit(command),"Native worker busy")
            guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Native scoring deadline") }
            try require(state.status == 0,"Native scoring failed: \(state.message)")
            return state
        }
        do {
            var settings = NativeSettings(directory:model.path)
            settings.metal = true; settings.mlx = true; settings.chat = true; settings.loadIdentity = .managedLlama3
            settings.capacity = 512; settings.chunk = 32; settings.numericalBudget = 2*1024*1024*1024; settings.textBudget = 256*1024*1024
            let loaded = try execute(.load(settings))
            try require(loaded.acceptedIdentity?.profile == .llama3 && loaded.acceptedIdentity?.loadedLimit == 512,"Expected audited Llama owner")
            for item in specification.cases {
                let n = item.prefixTokens
                try require(item.logitRow == n-1 && item.targetPosition == n && item.targetId == ids[n],"Frozen target alignment differs")
                let prefix = Array(ids.prefix(n))
                let committed = try execute(.generate(operation:3,tokens:prefix,maximum:0,stops:[]))
                try require(committed.evaluated == UInt32(n) && committed.pending == 0 && committed.savedPrefix == prefix,"Incomplete prefix")
                let measured = try execute(.score(target:item.targetId))
                guard let score = measured.score else { throw NativeContextFailure("Missing native score") }
                try require(score.target == item.targetId && score.evaluated == UInt32(n) && score.vocabulary == 128256,"Mismatched native score")
                rows.append(["prefix_tokens":n,"target_position":n,"target_id":item.targetId,
                             "score":try JSONSerialization.jsonObject(with:JSONEncoder().encode(score))])
            }
        } catch { failure = error }
        worker.close(); try require(worker.waitUntilClosed(seconds:30),"Native close deadline")
        if let failure { throw failure }
        let memory = try NativeFootprint.read()
        try require(memory.lifetimePeakPhysicalBytes <= 6*1024*1024*1024,"Native scoring peak exceeds6GiB")
        let report: [String:Any] = ["format":"toklet-llama3-original-short-native-v1","passed":true,
            "input_manifest_sha256":CommandLine.arguments[3],"cases":rows,"full_quality_qualified":false,
            "profile":"native MLX8/Half/MlxAffine/Compensated/Reference/capacity512",
            "memory":try JSONSerialization.jsonObject(with:JSONEncoder().encode(memory))]
        let output = URL(fileURLWithPath:CommandLine.arguments[4])
        try JSONSerialization.data(withJSONObject:report,options:[.sortedKeys,.prettyPrinted]).write(to:output,options:.withoutOverwriting)
        print("LLAMA_ORIGINAL_NATIVE scores=\(rows.count)")
    }
}

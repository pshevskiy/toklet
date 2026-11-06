import Foundation
import CryptoKit
import Darwin

struct HubFailure: LocalizedError, Sendable {
    let message: String
    var errorDescription: String? { message }
    init(_ message: String) { self.message = message }
}

struct HubFile: Codable, Equatable, Sendable {
    let name: String
    let bytes: Int64
    let sha256: String
}

struct HubProfile: Codable, Equatable, Sendable {
    let repository: String
    let revision: String
    let files: [HubFile]
    // Bounded 64-bit inventory admission for the selected multi-GB Mac candidate.
    static let maximumFileBytes: Int64 = 4 * 1024 * 1024 * 1024
    static let maximumTotalBytes: Int64 = 6 * 1024 * 1024 * 1024
    var totalBytes: Int64 { files.reduce(0) { $0 + $1.bytes } }
    var folder: String { repository.replacingOccurrences(of:"/",with:"--") + "--" + revision }
    static let smol = HubProfile(repository:"mlx-community/SmolLM2-135M-Instruct-8bit",
        revision:"0f0d9b8218915bc34d401e1a340b8c049d300d5e",files:[
            HubFile(name:"config.json",bytes:1097,sha256:"5da91905efdfde905f566d76d4031e6ca7c4cb2a01dc4e3812a70b855188fc73"),
            HubFile(name:"model.safetensors",bytes:143030401,sha256:"ec569c771e56424636be99625aba9031abc525a68a5f4ab16d678eca96fef12c"),
            HubFile(name:"model.safetensors.index.json",bytes:48979,sha256:"d016e7514e8bfed162eb9d0716c73244c1b73792db3143e92e54138c1772db44"),
            HubFile(name:"tokenizer.json",bytes:3522656,sha256:"7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7"),
            HubFile(name:"tokenizer_config.json",bytes:3764,sha256:"a27f638bd2831f5c3dea654a75838930f2b11fbe550c4d4e1d5d7bd07157b2ee"),
            HubFile(name:"special_tokens_map.json",bytes:655,sha256:"2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52")])
    static let qwen = HubProfile(repository:"mlx-community/Qwen2.5-0.5B-Instruct-8bit",
        revision:"6474d4c1a05bd7bd634a9c993560e4e7272eda3d",files:[
            HubFile(name:"config.json",bytes:783,sha256:"4dae7c9622e41e50752ba422149df0694044266c06cd851aff630799701c83c6"),
            HubFile(name:"model.safetensors",bytes:525045902,sha256:"3dd0b6c2983ac5fe35f60ba260b1c7c35e4c38f17f3d5139d0bb477924e7aef4"),
            HubFile(name:"model.safetensors.index.json",bytes:44209,sha256:"e48752ffebac15ee35d1d86e96c9b87f2efd7783b19ebf82a26955dbd43e0476"),
            HubFile(name:"tokenizer.json",bytes:7031673,sha256:"a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf"),
            HubFile(name:"tokenizer_config.json",bytes:7308,sha256:"f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7"),
            HubFile(name:"special_tokens_map.json",bytes:613,sha256:"76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd")])
    static let qwen3 = HubProfile(repository:"mlx-community/Qwen3-0.6B-8bit",
        revision:"11de96878523501bcaa86104e3c186de07ff9068",files:[
            HubFile(name:"config.json",bytes:937,sha256:"9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240"),
            HubFile(name:"model.safetensors",bytes:633442994,sha256:"3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397"),
            HubFile(name:"model.safetensors.index.json",bytes:49731,sha256:"45addf7779cdbd5af18d92f9f7b79c188f6ebbf32cff7b4ae7dbe68a91340ebc"),
            HubFile(name:"tokenizer.json",bytes:11422654,sha256:"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            HubFile(name:"tokenizer_config.json",bytes:10219,sha256:"8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6"),
            HubFile(name:"special_tokens_map.json",bytes:613,sha256:"76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd")])
    static let mistral = HubProfile(repository:"mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        revision:"a4b8f870474b0eb527f466a03fbc187830d271f5",files:[
            HubFile(name:"config.json",bytes:721,sha256:"efc145d2f53d6d6ef2639be0556b4ecf6faac6cf8a7199ad2b7bb5ebcf443bdd"),
            HubFile(name:"model.safetensors",bytes:4077480215,sha256:"d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d"),
            HubFile(name:"model.safetensors.index.json",bytes:52381,sha256:"525940738b16d210fdfd692a24752d050c15a50cbe208dd82b06204d640d687b"),
            HubFile(name:"tokenizer.json",bytes:1961548,sha256:"e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49"),
            HubFile(name:"tokenizer_config.json",bytes:138039,sha256:"e8fbcfef34d855dba2c7f1d91a2c5470cf493d00ecfd06488ce21e45325bc8ae"),
            HubFile(name:"special_tokens_map.json",bytes:414,sha256:"6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0")])
    static let llama3 = HubProfile(repository:"mlx-community/Llama-3.2-1B-Instruct-8bit",
        revision:"d48cdf0a4ea22d893b7c63a99d6a693e24822795",files:[
            HubFile(name:"config.json",bytes:1041,sha256:"847794a29e25927b4b98f35f5e66829e1b8bfd78f7743a9582780b4fd4a8ef9d"),
            HubFile(name:"model.safetensors",bytes:1313157436,sha256:"cd3582a78766654e3e2107e3b4d51d9cb18e6754b659bfb44a34328ef9d7fa52"),
            HubFile(name:"model.safetensors.index.json",bytes:26160,sha256:"c444df144fe9d86b42f8b1d797432c1f8379225316e41b1b9551e90d63136155"),
            HubFile(name:"tokenizer.json",bytes:9085657,sha256:"79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"),
            HubFile(name:"tokenizer_config.json",bytes:54528,sha256:"9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e"),
            HubFile(name:"special_tokens_map.json",bytes:296,sha256:"6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec")])
    static let qwen15 = HubProfile(repository:"mlx-community/Qwen2.5-1.5B-Instruct-8bit",
        revision:"c7e1ff25efda4a2c8f7ee312380557fb42e145a2",files:[
            HubFile(name:"config.json",bytes:784,sha256:"a939169f5c833fd4c1d872dc07f521c6e5098075ac53bf007b8595343e980937"),
            HubFile(name:"model.safetensors",bytes:1640414038,sha256:"8b76d08a5a70261b9ba4da5c06ac28bf09ee848e2322bae901e84764b7bba8e8"),
            HubFile(name:"model.safetensors.index.json",bytes:51570,sha256:"b885e07bdf11ca668b338f4bceed9ac0000dfcab56abe4b92670b748291d3643"),
            HubFile(name:"tokenizer.json",bytes:7031673,sha256:"a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf"),
            HubFile(name:"tokenizer_config.json",bytes:7308,sha256:"f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7"),
            HubFile(name:"special_tokens_map.json",bytes:613,sha256:"76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd")])
    static let gemmaBF16Shard = HubFile(name:"model.safetensors",bytes:10_246_621_918,
        sha256:"2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550")
    private static let gemmaEngineBudget: UInt64 = 5_632 * 1024 * 1024
    // Additional headroom beyond the engine and its included codec. The
    // existing Mac process policy already subtracts the current footprint.
    private static let gemmaApplicationAllowance: UInt64 = 272 * 1024 * 1024
    static let gemmaBF16 = HubProfile(repository:"google/gemma-4-E2B-it",
        revision:"70af34e20bd4b7a91f0de6b22675850c43922a03",files:[
            HubFile(name:"chat_template.jinja",bytes:17_336,sha256:"2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082"),
            HubFile(name:"config.json",bytes:4_954,sha256:"1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"),
            HubFile(name:"generation_config.json",bytes:208,sha256:"d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de"),
            gemmaBF16Shard,
            HubFile(name:"tokenizer.json",bytes:32_169_626,sha256:"cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
            HubFile(name:"tokenizer_config.json",bytes:2_095,sha256:"90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df")
        ])
    // Mac qualification is bounded to512 positions; iOS awaits physical validation.
    static let catalog: [HubProfile] = {
        var profiles: [HubProfile] = [.smol,.qwen,.qwen3,.mistral]
        #if os(macOS)
        profiles.append(contentsOf:[.llama3,.qwen15])
        #if arch(arm64)
        profiles.append(.gemmaBF16)
        #endif
        #endif
        return profiles
    }()
    static let qwen3AccuracyNotice = "The pinned Qwen3-0.6B 8-bit profile is experimental and has not passed Toklet’s accuracy checks."
    var supportNotice: String? {
        if self == Self.qwen3 { return Self.qwen3AccuracyNotice }
        if self == Self.gemmaBF16 { return "Mac basic-text chat up to 512 tokens. Downloads 10.28 GB." }
        if self == Self.llama3 || self == Self.qwen15 { return "Mac chat window: 512 tokens." }
        return nil
    }
    static func matching(_ query: String) -> [HubProfile] {
        let words = query.split(whereSeparator:{ $0.isWhitespace }).map(String.init)
        return catalog.filter { profile in
            words.allSatisfy { (profile.repository + " " + profile.name).localizedCaseInsensitiveContains($0) }
        }
    }
    var name: String {
        switch repository {
        case Self.qwen.repository: "Qwen2.5-0.5B · 8-bit"
        case Self.qwen15.repository: "Qwen2.5-1.5B · 8-bit"
        case Self.qwen3.repository: "Qwen3-0.6B · 8-bit · non-thinking"
        case Self.mistral.repository: "Mistral-7B-Instruct-v0.3 · 4-bit"
        case Self.llama3.repository: "Llama-3.2-1B · 8-bit"
        case Self.gemmaBF16.repository: "Gemma 4 E2B · BF16"
        default: "SmolLM2-135M · 8-bit (Llama family)"
        }
    }
    var numericalBudget: UInt64 {
        switch repository {
        case Self.qwen.repository: 1024 * 1024 * 1024
        case Self.qwen3.repository: 1536 * 1024 * 1024
        case Self.mistral.repository: 5 * 1024 * 1024 * 1024
        case Self.gemmaBF16.repository: Self.gemmaEngineBudget
        case Self.llama3.repository, Self.qwen15.repository: 2 * 1024 * 1024 * 1024
        default: 512 * 1024 * 1024
        }
    }
    var textBudget: UInt64 {
        switch repository {
        case Self.qwen.repository, Self.qwen15.repository: 128 * 1024 * 1024
        case Self.qwen3.repository, Self.llama3.repository: 256 * 1024 * 1024
        case Self.gemmaBF16.repository: 0
        default: 64 * 1024 * 1024
        }
    }
    var processAllowance: UInt64 {
        switch repository {
        case Self.qwen.repository: 1536 * 1024 * 1024
        case Self.qwen3.repository: 2304 * 1024 * 1024
        case Self.mistral.repository: 5632 * 1024 * 1024
        case Self.gemmaBF16.repository: Self.gemmaEngineBudget + Self.gemmaApplicationAllowance
        case Self.llama3.repository, Self.qwen15.repository: 3 * 1024 * 1024 * 1024
        default: 768 * 1024 * 1024
        }
    }
    var vocabulary: UInt32 {
        switch repository {
        case Self.qwen.repository, Self.qwen15.repository: 151936
        case Self.qwen3.repository: 151936
        case Self.mistral.repository: 32768
        case Self.llama3.repository: 128256
        case Self.gemmaBF16.repository: 262144
        default: 49152
        }
    }
    var key: String {
        switch repository {
        case Self.qwen.repository: "qwen"
        case Self.qwen15.repository: "qwen15"
        case Self.qwen3.repository: "qwen3"
        case Self.mistral.repository: "mistral"
        case Self.llama3.repository: "llama3"
        case Self.gemmaBF16.repository: "gemma-bf16"
        default: "smol"
        }
    }
    var searchQuery: String {
        switch repository {
        case Self.qwen.repository: "Qwen2.5-0.5B"
        case Self.qwen15.repository: "Qwen2.5-1.5B-Instruct-8bit"
        case Self.qwen3.repository: "Qwen3-0.6B-8bit"
        case Self.mistral.repository: "Mistral-7B-Instruct-v0.3-4bit"
        case Self.llama3.repository: "Llama-3.2-1B-Instruct-8bit"
        case Self.gemmaBF16.repository: repository
        default: "SmolLM2-135M"
        }
    }
    static func named(_ key: String) -> HubProfile? { catalog.first { $0.key == key } }
    static func identify(_ directory: URL) throws -> HubProfile {
        let bytes = try ModelHub.withRegular(directory.appendingPathComponent("config.json"),maximum:16 * 1024) { handle,size in
            let data = try handle.read(upToCount:16 * 1024 + 1) ?? Data()
            guard data.count == size else { throw HubFailure("Model configuration changed while opening.") }
            return data
        }
        let hash = SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()
        guard let profile = catalog.first(where:{ $0.files.first(where:{ $0.name == "config.json" })?.sha256 == hash }) else {
            throw HubFailure("This folder does not contain a supported pinned chat model configuration.")
        }
        return profile
    }
    func validate() throws {
        guard HubAPI.validRepository(repository), revision.count == 40, revision.allSatisfy({ $0.isHexDigit && $0.isASCII }),
              !files.isEmpty, files.count <= 16, Set(files.map(\.name)).count == files.count else { throw HubFailure("Invalid download profile.") }
        let largeOfficial = self == Self.gemmaBF16 && Self.catalog.contains(Self.gemmaBF16)
        let totalLimit = largeOfficial ? Self.gemmaBF16.totalBytes : Self.maximumTotalBytes
        var total: Int64 = 0
        for file in files {
            let fileLimit = largeOfficial && file == Self.gemmaBF16Shard ? file.bytes : Self.maximumFileBytes
            guard HubAPI.safeComponent(file.name), file.name != "snapshot.json", file.bytes > 0, file.bytes <= fileLimit,
                  file.sha256.count == 64, file.sha256.allSatisfy({ $0.isHexDigit && $0.isASCII }) else { throw HubFailure("Invalid file inventory.") }
            let sum = total.addingReportingOverflow(file.bytes)
            guard !sum.overflow, sum.partialValue <= totalLimit else { throw HubFailure("Download inventory is too large.") }
            total = sum.partialValue
        }
    }
}

struct HubModel: Decodable, Identifiable, Sendable {
    let id: String
    var profile: HubProfile? { HubProfile.catalog.first { $0.repository == id } }
    var url: URL { URL(string:"https://huggingface.co/" + id)! }
}

struct HubDiscovery: Sendable {
    let models: [HubModel]
    let failedRequests: Int
}

enum HubAPI {
    static let metadataLimit: Int64 = 2 * 1024 * 1024
    static func safeComponent(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 128 && name != "." && name != ".." &&
        name.utf8.allSatisfy { (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45,46,95].contains($0) }
    }
    static func validRepository(_ name: String) -> Bool {
        let parts = name.split(separator:"/",omittingEmptySubsequences:false)
        return parts.count == 2 && parts.allSatisfy { safeComponent(String($0)) }
    }
    static func searchURL(_ query: String) throws -> URL {
        let query = query.trimmingCharacters(in:.whitespacesAndNewlines)
        guard !query.isEmpty, query.utf8.count <= 128 else { throw HubFailure("Enter up to128 bytes of model name.") }
        var url = URLComponents(string:"https://huggingface.co/api/models")!
        let official = HubProfile.gemmaBF16
        let exactOfficial = HubProfile.catalog.contains(official) && query.caseInsensitiveCompare(official.repository) == .orderedSame
        url.queryItems = [URLQueryItem(name:"search",value:exactOfficial ? official.repository : query),
            URLQueryItem(name:"limit",value:"20")]
        if !exactOfficial { url.queryItems?.append(URLQueryItem(name:"filter",value:"mlx")) }
        return url.url!
    }
    // One popular MLX listing plus the supported families. A popular listing
    // alone can omit small checkpoints that are the best fit for an iPhone.
    static var discoveryURLs: [URL] {
        var popular = URLComponents(string:"https://huggingface.co/api/models")!
        popular.queryItems = [URLQueryItem(name:"filter",value:"mlx"),URLQueryItem(name:"pipeline_tag",value:"text-generation"),
            URLQueryItem(name:"sort",value:"downloads"),URLQueryItem(name:"direction",value:"-1"),URLQueryItem(name:"limit",value:"20")]
        return [popular.url!] + HubProfile.catalog.map { try! searchURL($0.searchQuery) }
    }
    static func searchResults(_ data: Data) throws -> [HubModel] {
        guard data.count <= metadataLimit else { throw HubFailure("Search response is too large.") }
        let rows = try JSONDecoder().decode([HubModel].self,from:data)
        guard rows.count <= 20, rows.allSatisfy({ validRepository($0.id) }), Set(rows.map(\.id)).count == rows.count else { throw HubFailure("Invalid search results.") }
        return rows
    }
    static func inventoryURL(_ profile: HubProfile) throws -> URL {
        try profile.validate()
        return URL(string:"https://huggingface.co/api/models/\(profile.repository)/revision/\(profile.revision)?blobs=true")!
    }
    static func fileURL(_ profile: HubProfile,_ file: HubFile) -> URL {
        URL(string:"https://huggingface.co/\(profile.repository)/resolve/\(profile.revision)/\(file.name)")!
    }
    static func admitInventory(_ data: Data,profile: HubProfile) throws {
        struct Inventory: Decodable {
            struct Entry: Decodable {
                struct LFS: Decodable { let sha256: String; let size: Int64 }
                let rfilename: String; let size: Int64?; let lfs: LFS?
            }
            let sha: String; let siblings: [Entry]
        }
        try profile.validate()
        guard data.count <= metadataLimit else { throw HubFailure("Model inventory is too large.") }
        let inventory = try JSONDecoder().decode(Inventory.self,from:data)
        guard inventory.sha == profile.revision, inventory.siblings.count <= 1024,
              Set(inventory.siblings.map(\.rfilename)).count == inventory.siblings.count else { throw HubFailure("Model revision or inventory changed.") }
        for file in profile.files {
            guard let entry = inventory.siblings.first(where:{ $0.rfilename == file.name }), entry.size == file.bytes else { throw HubFailure("Required model file is missing or changed.") }
            if profile == .gemmaBF16, file == HubProfile.gemmaBF16Shard, entry.lfs == nil {
                throw HubFailure("Official model weight identity is missing.")
            }
            if let lfs = entry.lfs, lfs.size != file.bytes || lfs.sha256 != file.sha256 { throw HubFailure("Model weight identity changed.") }
        }
    }
}

struct HubAdmission: Sendable {
    let unifiedGPU: Bool
    let physicalMemory: UInt64
    let availableMemory: UInt64
    func check(_ profile: HubProfile = .smol) throws {
        guard unifiedGPU else { throw HubFailure("This device cannot run Toklet’s Metal model.") }
        guard physicalMemory >= 2 * 1024 * 1024 * 1024, availableMemory >= profile.processAllowance else {
            throw HubFailure("Not enough memory available. Unload the current model or close other apps, then retry.")
        }
    }
}

struct HubSnapshotReceipt: Codable, Equatable, Sendable {
    let version: Int
    let profile: HubProfile
}

typealias HubFetch = @Sendable (URL, URL, Int64, @escaping @Sendable (Int64) -> Void) async throws -> Void

// Coarse app boundary only. Actor reentrancy is guarded explicitly while a transfer
// awaits URLSession. No download, hashing or filesystem operation uses the engine worker.
actor ModelHub {
    let root: URL
    private let fetch: HubFetch
    private let freeBytes: @Sendable (URL) throws -> Int64
    private let removeMetadata: @Sendable (URL) throws -> Void
    private var busy = false
    static let reserve: Int64 = 64 * 1024 * 1024
    init(root: URL,fetch: @escaping HubFetch = { try await HubTransfer.fetch($0,$1,$2,$3) },
         freeBytes: @escaping @Sendable (URL) throws -> Int64 = { try ModelHub.diskSpace($0) },
         removeMetadata: @escaping @Sendable (URL) throws -> Void = {
             if FileManager.default.fileExists(atPath:$0.path) { try FileManager.default.removeItem(at:$0) }
         }) {
        self.root = root; self.fetch = fetch; self.freeBytes = freeBytes; self.removeMetadata = removeMetadata
    }
    static func diskSpace(_ directory: URL) throws -> Int64 {
        let values = try directory.resourceValues(forKeys:[.volumeAvailableCapacityForImportantUsageKey])
        guard let value = values.volumeAvailableCapacityForImportantUsage else { throw HubFailure("Could not check available storage.") }
        return value
    }
    private func prepareRoot() throws {
        // Inspect existing ancestors before mkdir. /var and /tmp are OS-owned aliases on
        // Apple systems; no user-controlled link within the container is followed.
        var path = root.standardizedFileURL
        while path.path != "/" {
            var attributes = stat()
            if lstat(path.path,&attributes) == 0 {
                let kind = attributes.st_mode & S_IFMT
                let systemAlias = ["/var","/tmp"].contains(path.path) &&
                    (try? FileManager.default.destinationOfSymbolicLink(atPath:path.path)) == "private" + path.path
                guard kind == S_IFDIR || (kind == S_IFLNK && systemAlias) else { throw HubFailure("Model storage must be a regular directory.") }
            } else if errno != ENOENT { throw HubFailure("Could not inspect model storage.") }
            path.deleteLastPathComponent()
        }
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    }
    private func space(_ needed: Int64) throws {
        try Task.checkCancellation()
        guard try freeBytes(root) >= needed + Self.reserve else { throw HubFailure("Not enough storage. Free space on this device, then retry.") }
    }
    func search(_ query: String) async throws -> [HubModel] {
        guard !busy else { throw HubFailure("A model operation is already running.") }
        busy = true; defer { busy = false }
        let url = try HubAPI.searchURL(query)
        try prepareRoot(); try space(HubAPI.metadataLimit)
        let temporary = root.appendingPathComponent(".search-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:temporary) }
        try await fetch(url,temporary,HubAPI.metadataLimit,{ _ in })
        try Task.checkCancellation()
        return try HubAPI.searchResults(Data(contentsOf:temporary))
    }
    func discover() async throws -> HubDiscovery {
        guard !busy else { throw HubFailure("A model operation is already running.") }
        busy = true; defer { busy = false }
        let urls = HubAPI.discoveryURLs, fetch = self.fetch, root = self.root, remove = self.removeMetadata
        try prepareRoot(); try space(HubAPI.metadataLimit * Int64(urls.count))
        // Four bounded metadata requests run concurrently; no model payload is
        // fetched. Task-group scope joins cancelled URLSessions before cleanup.
        let pages = await withTaskGroup(of:(Int,Result<[HubModel],any Error>).self) { group in
            for (index,url) in urls.enumerated() {
                group.addTask {
                    let temporary = root.appendingPathComponent(".discover-" + UUID().uuidString)
                    var result: Result<[HubModel],any Error>
                    do {
                        try await fetch(url,temporary,HubAPI.metadataLimit,{ _ in })
                        try Task.checkCancellation()
                        result = .success(try HubAPI.searchResults(Data(contentsOf:temporary)))
                    } catch { result = .failure(error) }
                    do { try remove(temporary) }
                    catch {
                        let original: (any Error)?
                        if case .failure(let failure) = result { original = failure } else { original = nil }
                        result = .failure(HubTransfer.metadataCleanupFailure(original,"metadata temporary removal: " + error.localizedDescription))
                    }
                    return (index,result)
                }
            }
            var pages = [Result<[HubModel],any Error>](repeating:.failure(HubFailure("Metadata request did not finish.")),count:urls.count)
            // Drain all children even after an error. A later sibling can carry
            // a close/removal failure that an early throwing iteration would lose.
            for await (index,page) in group {
                pages[index] = page
                if case .failure(let error) = page, HubTransfer.mergeCleanupFailure(nil,error) != nil { group.cancelAll() }
            }
            return pages
        }
        var fatal: (any Error)? = Task.isCancelled ? CancellationError() : nil
        var failures = 0
        for page in pages {
            if case .failure(let error) = page {
                failures += 1
                if let cleanup = HubTransfer.mergeCleanupFailure(fatal,error) { fatal = cleanup }
            }
        }
        if let fatal { throw fatal }
        guard failures < urls.count else { throw HubFailure("Cannot reach Hugging Face. Check your connection and tap Find best model to retry.") }
        var rows: [HubModel] = [], seen = Set<String>()
        for page in pages {
            if case .success(let models) = page {
                for row in models where seen.insert(row.id).inserted { rows.append(row) }
            }
        }
        return HubDiscovery(models:rows,failedRequests:failures)
    }
    func verified(_ profile: HubProfile) throws -> URL? {
        guard !busy else { throw HubFailure("A model operation is already running.") }
        try prepareRoot(); try profile.validate()
        let folder = root.appendingPathComponent(profile.folder,isDirectory:true)
        guard FileManager.default.fileExists(atPath:folder.path) else { return nil }
        try Self.verify(folder,profile:profile)
        return folder
    }
    // Presence only, for button labels. Full SHA/type/receipt validation still
    // runs once when the user chooses Use; opening this screen must not hash GBs.
    func savedProfiles() throws -> Set<String> {
        try prepareRoot()
        return Set(HubProfile.catalog.filter { profile in
            FileManager.default.fileExists(atPath:root.appendingPathComponent(profile.folder)
                .appendingPathComponent("snapshot.json").path)
        }.map(\.repository))
    }
    static func verify(_ folder: URL,profile: HubProfile) throws {
        try profile.validate(); try Task.checkCancellation()
        let directory = Darwin.open(folder.path,O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK)
        guard directory >= 0 else { throw HubFailure("Invalid downloaded folder.") }
        defer { Darwin.close(directory) }
        let names = try FileManager.default.contentsOfDirectory(atPath:folder.path)
        guard Set(names) == Set(profile.files.map(\.name) + ["snapshot.json"]) else { throw HubFailure("Downloaded files changed. Remove this model folder in Files and download again.") }
        let receipt = folder.appendingPathComponent("snapshot.json")
        let data = try withRegular(receipt,maximum:65536,directory:directory) { handle,size in
            var data = Data()
            while let chunk = try handle.read(upToCount:4096), !chunk.isEmpty {
                try Task.checkCancellation()
                guard data.count + chunk.count <= 65536 else { throw HubFailure("Snapshot receipt is too large.") }
                data.append(chunk)
            }
            guard data.count == size else { throw HubFailure("Snapshot receipt changed while reading.") }
            return data
        }
        let stored = try JSONDecoder().decode(HubSnapshotReceipt.self,from:data)
        guard stored == HubSnapshotReceipt(version:1,profile:profile) else { throw HubFailure("Downloaded snapshot identity changed.") }
        for file in profile.files { try verifyFile(folder.appendingPathComponent(file.name),file:file,directory:directory) }
    }
    // Admit type/size and read through the same descriptor. O_NONBLOCK prevents
    // malicious FIFO/device replacement from blocking before fstat rejects it.
    static func withRegular<T>(_ url: URL,maximum: Int64,directory: Int32? = nil,
                                       body: (FileHandle,Int64) throws -> T) throws -> T {
        let flags = O_RDONLY | O_NOFOLLOW | O_NONBLOCK
        let descriptor = directory.map { openat($0,url.lastPathComponent,flags) } ?? Darwin.open(url.path,flags)
        guard descriptor >= 0 else { throw HubFailure("Could not open a regular downloaded file.") }
        let handle = FileHandle(fileDescriptor:descriptor,closeOnDealloc:true)
        defer { try? handle.close() }
        var before = stat()
        guard fstat(descriptor,&before) == 0, before.st_mode & S_IFMT == S_IFREG,
              before.st_size >= 0, before.st_size <= maximum else { throw HubFailure("Invalid or oversized downloaded file.") }
        let result = try body(handle,before.st_size)
        var after = stat()
        guard fstat(descriptor,&after) == 0, before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec else { throw HubFailure("Downloaded file changed while reading.") }
        return result
    }
    static func verifyFile(_ url: URL,file: HubFile,directory: Int32? = nil) throws {
        try withRegular(url,maximum:file.bytes,directory:directory) { handle,size in
            guard size == file.bytes else { throw HubFailure("Downloaded file has the wrong size.") }
            var digest = SHA256(); var count: Int64 = 0
            while let data = try autoreleasepool(invoking:{ try handle.read(upToCount:256 * 1024) }), !data.isEmpty {
                try Task.checkCancellation(); count += Int64(data.count)
                guard count <= file.bytes else { throw HubFailure("Downloaded file is too large.") }
                digest.update(data:data)
            }
            guard count == file.bytes, digest.finalize().map({ String(format:"%02x",$0) }).joined() == file.sha256 else { throw HubFailure("Downloaded file failed its size or SHA-256 check. Remove a damaged saved folder in Files and retry.") }
        }
    }
    func download(_ profile: HubProfile,admission: HubAdmission,
                  progress: @escaping @Sendable (Int64,Int64) -> Void) async throws -> URL {
        guard !busy else { throw HubFailure("A model operation is already running.") }
        busy = true; defer { busy = false }
        try admission.check(profile); try profile.validate(); try prepareRoot()
        let destination = root.appendingPathComponent(profile.folder,isDirectory:true)
        if FileManager.default.fileExists(atPath:destination.path) { try Self.verify(destination,profile:profile); return destination }
        // Includes one staging copy and a conservative additional full-copy allowance.
        try space(profile.totalBytes * 2 + HubAPI.metadataLimit)
        let stage = root.appendingPathComponent(".download-" + UUID().uuidString)
        guard Darwin.mkdir(stage.path,0o700) == 0 else { throw HubFailure("Could not create exclusive download staging directory.") }
        defer { try? FileManager.default.removeItem(at:stage) }
        let inventory = stage.appendingPathComponent(".inventory")
        try await fetch(HubAPI.inventoryURL(profile),inventory,HubAPI.metadataLimit,{ _ in })
        try HubAPI.admitInventory(Data(contentsOf:inventory),profile:profile)
        try FileManager.default.removeItem(at:inventory)
        var completed: Int64 = 0
        for file in profile.files {
            try space(profile.totalBytes - completed)
            let before = completed
            let url = stage.appendingPathComponent(file.name)
            try await fetch(HubAPI.fileURL(profile,file),url,file.bytes,{ bytes in progress(before + bytes,profile.totalBytes) })
            try Self.verifyFile(url,file:file)
            completed += file.bytes
        }
        try Task.checkCancellation(); try space(0)
        let receipt = try JSONEncoder().encode(HubSnapshotReceipt(version:1,profile:profile))
        try receipt.write(to:stage.appendingPathComponent("snapshot.json"),options:.withoutOverwriting)
        try Self.verify(stage,profile:profile)
        try Task.checkCancellation()
        // Synchronous same-root rename is the publication linearization point.
        // Cancellation observed before it publishes nothing; after it, a complete
        // valid snapshot may remain. Existing destinations are never replaced.
        guard !FileManager.default.fileExists(atPath:destination.path) else { throw HubFailure("A model folder already exists. Retry to verify it.") }
        guard renamex_np(stage.path,destination.path,UInt32(RENAME_EXCL)) == 0 else { throw HubFailure("Could not publish the download exclusively. Existing models are preserved.") }
        progress(profile.totalBytes,profile.totalBytes)
        return destination
    }
}

import Foundation

@main
@MainActor
struct HubProfileMain {
    static func require(_ condition: @autoclosure () -> Bool,_ label: String) throws {
        guard condition() else { throw HubFailure("Profile check failed: " + label) }
    }
    static func main() async throws {
        let fixtures = URL(fileURLWithPath:FileManager.default.currentDirectoryPath).appendingPathComponent("tests/fixtures/hub-profiles-v1")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TokletProfileCheck-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        defer { try? FileManager.default.removeItem(at:root) }
        for profile in HubProfile.catalog {
            let folder = root.appendingPathComponent(profile.key,isDirectory:true)
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:false)
            let data = try Data(contentsOf:fixtures.appendingPathComponent(profile.key + "-config.json"))
            let config = folder.appendingPathComponent("config.json")
            try data.write(to:config)
            let found = try HubProfile.identify(folder)
            try require(found == profile,"pinned config identifies selected profile")
            try (data + Data(" ".utf8)).write(to:config)
            var rejected = false
            do { _ = try HubProfile.identify(folder) } catch { rejected = true }
            try require(rejected,"valid but changed config rejected")
            try FileManager.default.removeItem(at:config)
            try FileManager.default.createSymbolicLink(at:config,withDestinationURL:fixtures.appendingPathComponent(profile.key + "-config.json"))
            rejected = false
            do { _ = try HubProfile.identify(folder) } catch { rejected = true }
            try require(rejected,"linked config rejected")
        }
        let old = try JSONDecoder().decode(HubSnapshotReceipt.self,from:Data(contentsOf:fixtures.appendingPathComponent("old-smol-snapshot.json")))
        try require(old == HubSnapshotReceipt(version:1,profile:.smol),"previous Smol receipt remains compatible")
        actor Seen {
            var profiles: [HubProfile] = []
            func verify(_ profile: HubProfile) -> URL {
                profiles.append(profile)
                return URL(fileURLWithPath:"/verified-" + profile.key)
            }
            func values() -> [HubProfile] { profiles }
        }
        let seen = Seen()
        let store = HubStore(profile:.qwen,hub:ModelHub(root:root),admission:{
            HubAdmission(unifiedGPU:true,physicalMemory:6 * 1024 * 1024 * 1024,availableMemory:6 * 1024 * 1024 * 1024)
        },verify:{ await seen.verify($0) })
        store.refresh(); await store.waitForIdle()
        try require(store.localURL?.path == "/verified-qwen" && store.deviceAllowed,"selected Qwen refresh")
        var opened: URL?
        store.open { opened = $0 }; await store.waitForIdle()
        try require(opened?.path == "/verified-qwen","selected Qwen open")
        store.select(.mistral); await store.waitForIdle()
        try require(store.localURL?.path == "/verified-mistral" && store.deviceAllowed,"selected Mistral refresh")
        store.open { opened = $0 }; await store.waitForIdle()
        try require(opened?.path == "/verified-mistral","selected Mistral open")
        store.select(.smol); await store.waitForIdle()
        try require(store.localURL?.path == "/verified-smol","profile change refresh")
        let values = await seen.values()
        try require(values == [.qwen,.qwen,.mistral,.mistral,.smol],"verification captured correct profiles")
        print("TOKLET_HUB_PROFILE_PASS")
    }
}

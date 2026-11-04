import Foundation

@main
@MainActor
struct HubJourneyArgumentsMain {
    static func main() throws {
        var checks = 0
        func check(_ value: Bool, _ message: String) throws {
            checks += 1
            if !value { throw NativeContextFailure(message) }
        }
        let root = "/isolated-test-root"
        let ordinary = try HubOfflineJourneyMain.request(["journey","download",root])
        try check(ordinary.online && ordinary.profile == .smol && ordinary.expected == .smol,"Default target changed")
        var cases: [(String,HubProfile,NativeAcceptedIdentity.Profile)] = [
            ("smol",.smol,.smol),("qwen",.qwen,.qwen2),("qwen3",.qwen3,.qwen3),("mistral",.mistral,.mistral)]
        #if os(macOS)
        cases.append(("llama3",.llama3,.llama3))
        let legacy = try HubOfflineJourneyMain.request(["journey","offline",root,"--llama3"])
        try check(!legacy.online && legacy.profile == .llama3 && legacy.expected == .llama3,"Legacy target changed")
        #endif
        for (key,profile,expected) in cases {
            let request = try HubOfflineJourneyMain.request(["journey","offline",root,"--profile",key])
            try check(!request.online && request.root.path == root && request.profile == profile && request.expected == expected,"Requested pinned model changed")
        }
        for args in [[],["journey"],["journey","bad",root],["journey","download",root,"--profile"],
                     ["journey","download",root,"--wrong","qwen"],["journey","download",root,"--profile","unknown"],
                     ["journey","download",root,"--profile","qwen","--llama3"]] {
            var rejected = false
            do { _ = try HubOfflineJourneyMain.request(args) } catch { rejected = true }
            try check(rejected,"Invalid request must fail before any journey")
        }
        print("HUB_JOURNEY_ARGUMENTS checks=\(checks) no_models_loaded=true")
    }
}

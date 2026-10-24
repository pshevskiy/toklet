import Foundation

@main
struct CompatibilityProfilesMain {
    static func main() throws {
        let root = URL(fileURLWithPath:CommandLine.arguments[1])
        let folders = ["smol":"MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8","qwen":"MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c","mistral":"MLX-Mistral-7B-Instruct-v0.3-4bit-a4b8f87"]
        var results: [String:NativeChatCheck] = [:]
        for profile in HubProfile.catalog {
            let path = root.appendingPathComponent("models/" + folders[profile.key]!).path
            let metadata = try NativeCompatibility.inspect(directory:path,context:512,budget:profile.numericalBudget)
            guard metadata.mayAttemptLoad, !metadata.textProfileChecked, !metadata.qualificationEvaluated,
                  profile.numericalBudget + profile.textBudget + NativeCompatibility.requiredApplicationAllowance <= profile.processAllowance else {
                throw HubFailure("Pinned profile metadata or separate application allowance changed")
            }
            results[profile.key] = try runNativeChatCheck(directory:path,profile:profile)
        }
        print(String(decoding:try JSONEncoder().encode(results),as:UTF8.self))
    }
}

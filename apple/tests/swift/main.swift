import Foundation
import Darwin

do {
    guard CommandLine.arguments.count == 2 || CommandLine.arguments.count == 3 else {
        throw NSError(domain:"TokletChecks",code:1,userInfo:[NSLocalizedDescriptionKey:"Expected tiny artifact path and optional metal or mlx"])
    }
    let mode = CommandLine.arguments.count == 3 ? CommandLine.arguments[2] : "cpu"
    guard ["cpu","metal","mlx"].contains(mode) else {
        throw NSError(domain:"TokletChecks",code:1,userInfo:[NSLocalizedDescriptionKey:"Unknown backend"])
    }
    let result = try runWorkerChecks(directory:CommandLine.arguments[1],metal:mode != "cpu",mlx:mode == "mlx")
    let data = try JSONEncoder().encode(result)
    print(String(decoding:data,as:UTF8.self))
} catch {
    fputs("Swift worker check: \(error)\n",stderr)
    exit(1)
}

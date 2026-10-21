import Foundation
import Darwin

do {
    guard CommandLine.arguments.count == 2 || CommandLine.arguments.count == 3 else {
        throw NSError(domain:"TokletChecks",code:1,userInfo:[NSLocalizedDescriptionKey:"Expected tiny artifact path and optional metal"])
    }
    let result = try runWorkerChecks(directory:CommandLine.arguments[1],metal:CommandLine.arguments.count == 3)
    let data = try JSONEncoder().encode(result)
    print(String(decoding:data,as:UTF8.self))
} catch {
    fputs("Swift worker check: \(error)\n",stderr)
    exit(1)
}

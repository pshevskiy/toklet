import Foundation
let base = URL(fileURLWithPath:"/tmp/toklet-m7-path-alias-probe-v1",isDirectory:true)
try FileManager.default.createDirectory(at:base.appendingPathComponent("Model"),withIntermediateDirectories:true)
try Data([1]).write(to:base.appendingPathComponent("Model/file"))
let alias = base.appendingPathComponent("alias")
try FileManager.default.createSymbolicLink(atPath:alias.path,withDestinationPath:base.path)
for root in [base,alias] {
 let dir = root.appendingPathComponent("Model")
 let iterator = FileManager.default.enumerator(at:dir,includingPropertiesForKeys:nil)!
 let entries = iterator.allObjects.compactMap { ($0 as? URL)?.pathComponents }
 print(["root":root.pathComponents,"canonicalRoot":root.resolvingSymlinksInPath().pathComponents,"entries":entries] as [String:Any])
}

import SwiftUI
#if os(iOS)
import UIKit
#else
import AppKit
#endif

// Presentation only: never feed rendered Markdown back into tokenization.
enum ChatBlock: Equatable {
    case prose(String)
    case code(language: String, text: String)

    static func parse(_ text: String) -> [Self] {
        let lines = text.components(separatedBy:"\n")
        var blocks: [Self] = []
        var pending: [String] = []
        var fence: Character?
        var width = 0
        var language = ""
        for (index,line) in lines.enumerated() {
            let trimmed = line.drop(while:{ $0 == " " })
            let indent = line.count - trimmed.count
            if let marker = fence {
                let count = trimmed.prefix(while:{ $0 == marker }).count
                if indent <= 3, count >= width, trimmed.dropFirst(count).trimmingCharacters(in:.whitespaces).isEmpty {
                    blocks.append(.code(language:language,text:pending.joined(separator:"\n")))
                    pending = []; fence = nil
                } else { pending.append(line) }
            } else if indent <= 3, let marker = trimmed.first, marker == "`" || marker == "~",
                      trimmed.prefix(while:{ $0 == marker }).count >= 3 {
                if !pending.isEmpty { blocks.append(.prose(pending.joined(separator:"\n"))); pending = [] }
                fence = marker; width = trimmed.prefix(while:{ $0 == marker }).count
                language = String(trimmed.dropFirst(width).split(whereSeparator:{ $0.isWhitespace }).first ?? "").lowercased()
            } else { pending.append(line) }
            if index == lines.count - 1, fence != nil {
                // Interrupted or length-limited replies often have no closing fence.
                blocks.append(.code(language:language,text:pending.joined(separator:"\n")))
                pending = []
            }
        }
        if !pending.isEmpty { blocks.append(.prose(pending.joined(separator:"\n"))) }
        return blocks
    }
}

// Bound individual text-layout/highlighting work without changing saved text.
// Slices end on Character boundaries, including multi-scalar emoji.
enum ChatPages {
    static let maximumCharacters = 8192
    static func split(_ text: String) -> [String] {
        guard !text.isEmpty else { return [""] }
        var pages: [String] = []
        var start = text.startIndex
        while start < text.endIndex {
            var end = text.index(start,offsetBy:maximumCharacters,limitedBy:text.endIndex) ?? text.endIndex
            if end < text.endIndex,
               let newline = text[start..<end].lastIndex(of:"\n") {
                // Keep ordinary code lines/paragraphs together. Very long lines
                // still split at the character bound and remain fully copyable.
                let next = text.index(after:newline)
                if text.distance(from:start,to:next) >= maximumCharacters / 2 { end = next }
            }
            pages.append(String(text[start..<end])); start = end
        }
        return pages
    }
}

enum CodeHighlight {
    enum Kind { case plain, comment, string, number, keyword }
    struct Span { let text: String; let kind: Kind }
    private static let keywords = Set("actor async await as break case catch class const continue def defer do else enum extension false fn for from func guard if impl import in init let loop match mod mut nil None null private protocol pub public return self Self static struct super switch throw throws trait true try type typealias use val var void where while with yield".split(separator:" ").map(String.init))
    private static let slash = try! NSRegularExpression(pattern:#"//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z_0-9]*\b"#)
    private static let hash = try! NSRegularExpression(pattern:##"#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z_0-9]*\b"##)
    static func spans(_ text: String,language: String) -> [Span] {
        let regex = ["python","py","bash","sh","shell","ruby","rb","yaml","yml"].contains(language) ? hash : slash
        let source = text as NSString
        var result: [Span] = []
        var offset = 0
        for match in regex.matches(in:text,range:NSRange(location:0,length:source.length)) {
            if match.range.location > offset { result.append(Span(text:source.substring(with:NSRange(location:offset,length:match.range.location - offset)),kind:.plain)) }
            let value = source.substring(with:match.range)
            let kind: Kind
            if value.hasPrefix("//") || value.hasPrefix("/*") || (regex === hash && value.hasPrefix("#")) { kind = .comment }
            else if value.hasPrefix("\"") || value.hasPrefix("'") { kind = .string }
            else if value.first?.isNumber == true { kind = .number }
            else if keywords.contains(value) { kind = .keyword }
            else { kind = .plain }
            result.append(Span(text:value,kind:kind)); offset = NSMaxRange(match.range)
        }
        if offset < source.length { result.append(Span(text:source.substring(from:offset),kind:.plain)) }
        return result
    }
    static func attributed(_ text: String,language: String) -> AttributedString {
        var result = AttributedString()
        for span in spans(text,language:language) {
            var part = AttributedString(span.text)
            switch span.kind {
            case .plain: break
            case .comment: part[AttributeScopes.SwiftUIAttributes.ForegroundColorAttribute.self] = .secondary
            case .string: part[AttributeScopes.SwiftUIAttributes.ForegroundColorAttribute.self] = .green
            case .number: part[AttributeScopes.SwiftUIAttributes.ForegroundColorAttribute.self] = .orange
            case .keyword: part[AttributeScopes.SwiftUIAttributes.ForegroundColorAttribute.self] = .purple
            }
            result.append(part)
        }
        return result
    }
}

struct ChatContent: View, Equatable {
    let text: String
    var body: some View {
        LazyVStack(alignment:.leading,spacing:12) {
            ForEach(Array(ChatBlock.parse(text).enumerated()),id:\.offset) { _,block in
                switch block {
                case .prose(let prose):
                    ForEach(Array(ChatPages.split(prose).enumerated()),id:\.offset) { _,page in
                        Text(markdown(page)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                    }
                case .code(let language,let code): CodeSnippet(code:code,language:language)
                }
            }
        }
    }
    private func markdown(_ text: String) -> AttributedString {
        var value = (try? AttributedString(markdown:text,options:.init(interpretedSyntax:.inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
        for run in value.runs where run[AttributeScopes.FoundationAttributes.InlinePresentationIntentAttribute.self]?.contains(.code) == true {
            value[run.range][AttributeScopes.SwiftUIAttributes.FontAttribute.self] = .system(.body,design:.monospaced)
            value[run.range][AttributeScopes.SwiftUIAttributes.ForegroundColorAttribute.self] = .indigo
        }
        return value
    }
}

struct CodeSnippet: View {
    let code: String
    var language = ""
    @State private var copied = false
    @State private var page = 0
    var body: some View {
        let pages = ChatPages.split(code)
        let selected = min(page,pages.count - 1)
        VStack(alignment:.leading,spacing:0) {
            HStack {
                Text(language.isEmpty ? "Code" : language).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button(copied ? "Copied" : "Copy",systemImage:copied ? "checkmark" : "doc.on.doc") {
                    #if os(iOS)
                    UIPasteboard.general.string = code
                    #else
                    NSPasteboard.general.clearContents(); NSPasteboard.general.setString(code,forType:.string)
                    #endif
                    copied = true
                }.font(.caption).buttonStyle(.borderless).frame(minHeight:44).accessibilityLabel("Copy code")
            }.padding(.horizontal,12)
            Divider()
            ScrollView(.horizontal) {
                Text(CodeHighlight.attributed(pages[selected],language:language))
                    .font(.system(.callout,design:.monospaced)).textSelection(.enabled)
                    .fixedSize(horizontal:true,vertical:false).padding(12)
            }
            if pages.count > 1 {
                Divider()
                HStack {
                    Button("Previous",systemImage:"chevron.left") { page = selected - 1 }
                        .disabled(selected == 0)
                    Spacer()
                    Text("Part \(selected + 1) of \(pages.count)").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Next",systemImage:"chevron.right") { page = selected + 1 }
                        .disabled(selected == pages.count - 1)
                }.font(.caption).buttonStyle(.borderless).frame(minHeight:44).padding(.horizontal,12)
                Text("Copy includes the entire code block.").font(.caption).foregroundStyle(.secondary)
                    .padding(.horizontal,12).padding(.bottom,8)
            }
        }.background(.quaternary.opacity(0.4),in:RoundedRectangle(cornerRadius:12))
            .overlay(RoundedRectangle(cornerRadius:12).stroke(.quaternary,lineWidth:1))
            .onChange(of:code) { _,_ in copied = false; page = 0 }
    }
}

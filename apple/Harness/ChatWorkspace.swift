import Foundation

struct ChatMessage: Codable, Identifiable {
    var id = UUID()
    let role: String
    var text: String
    var writing = false
    var interrupted = false
    var notice = ""
}

struct LocalConversation: Codable, Identifiable {
    var id = UUID()
    var modelKey = ""
    var draft = ""
    var messages: [ChatMessage] = []
    var title: String { messages.first(where: { $0.role == "user" }).map { String($0.text.prefix(48)) } ?? "New chat" }
}

// Bounded UI history is independent of numerical KV and worker handles.
struct ChatWorkspace: Codable {
    static let maximumMessageBytes = 8 * 1024 * 1024
    static let maximumBytes = 64 * 1024 * 1024
    static let outputLimits: [UInt32] = [32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]
    // JSON can escape one input byte as six bytes. Reserve a whole bounded
    // completion plus metadata before beginning; no response is silently cut.
    static let completionReserve = maximumMessageBytes * 6 + 16 * 1024
    var version = 1
    var conversations: [LocalConversation]
    var selected: UUID
    var outputLimit: UInt32 = 128

    init() {
        let first = LocalConversation()
        conversations = [first]; selected = first.id
    }
    var selectedIndex: Int { conversations.firstIndex { $0.id == selected }! }
    var current: LocalConversation { conversations[selectedIndex] }
    mutating func edit(_ text: String) {
        if text.utf8.count <= 16 * 1024 { conversations[selectedIndex].draft = text }
    }
    mutating func newChat(modelKey: String? = nil) throws {
        guard conversations.count < 20 else { throw chatInputError("Your history has 20 chats. Delete a chat before starting another.") }
        var value = LocalConversation(); value.modelKey = modelKey ?? current.modelKey
        conversations.append(value); selected = value.id
    }
    mutating func remove(_ id: UUID) {
        conversations.removeAll { $0.id == id }
        if conversations.isEmpty { let first = LocalConversation(); conversations = [first] }
        if !conversations.contains(where: { $0.id == selected }) { selected = conversations[0].id }
    }
    mutating func removeAll(modelKey: String? = nil) {
        var first = LocalConversation(); first.modelKey = modelKey ?? current.modelKey
        conversations = [first]; selected = first.id
    }
    mutating func begin() throws -> (UUID, Data) {
        let input = current.draft
        guard !input.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else { throw chatInputError("Write a message first") }
        guard current.messages.count <= 62 else { throw chatInputError("This conversation is full. Start a new chat.") }
        var messages = current.messages.map { ["role":$0.role,"content":$0.text] }
        messages.append(["role":"user","content":input])
        let json = try JSONSerialization.data(withJSONObject:messages,options:[.sortedKeys])
        guard json.count <= 64 * 1024,
              try JSONEncoder().encode(self).count <= Self.maximumBytes - Self.completionReserve else {
            throw chatInputError("This history is full. Start a new chat or delete an older chat.")
        }
        let response = ChatMessage(role:"assistant",text:"",writing:true)
        conversations[selectedIndex].messages.append(ChatMessage(role:"user",text:input))
        conversations[selectedIndex].messages.append(response)
        conversations[selectedIndex].draft = ""
        return (response.id,json)
    }
    @discardableResult
    mutating func finish(_ result: NativeChatResult,restoreRejectedDraft: Bool = true) -> Bool {
        for index in conversations.indices {
            if let message = conversations[index].messages.firstIndex(where: { $0.id == result.id && $0.writing }) {
                if result.rejected {
                    guard message > 0, message == conversations[index].messages.count - 1,
                          conversations[index].messages[message - 1].role == "user" else { return false }
                    let input = conversations[index].messages[message - 1].text
                    conversations[index].messages.removeLast(2)
                    if restoreRejectedDraft { conversations[index].draft = input }
                    return true
                }
                conversations[index].messages[message].text = result.text
                conversations[index].messages[message].writing = false
                conversations[index].messages[message].interrupted = result.reason == 3 || !result.error.isEmpty
                conversations[index].messages[message].notice = result.error.isEmpty && result.reason == 1
                    ? "Response limit reached. Reply limits are available in Settings; you can also ask the model to continue."
                    : result.error
                return true
            }
        }
        return false // Late or repeated results cannot populate another chat.
    }
    static func restore(_ data: Data) throws -> Self {
        guard data.count <= maximumBytes else { throw chatInputError("Saved history is too large") }
        var value = try JSONDecoder().decode(Self.self,from:data)
        guard value.version == 1, (1...20).contains(value.conversations.count),
              Set(value.conversations.map(\.id)).count == value.conversations.count,
              value.conversations.contains(where: { $0.id == value.selected }),
              Self.outputLimits.contains(value.outputLimit) else { throw chatInputError("Invalid saved chat history") }
        var ids = Set<UUID>()
        for index in value.conversations.indices {
            let chat = value.conversations[index]
            guard chat.draft.utf8.count <= 16384, chat.modelKey.utf8.count <= 16384,
                  chat.messages.count <= 64, chat.messages.count.isMultiple(of:2) else { throw chatInputError("Invalid saved conversation") }
            for message in chat.messages.indices {
                let item = chat.messages[message]
                guard ids.insert(item.id).inserted, item.role == (message.isMultiple(of:2) ? "user" : "assistant"),
                      item.text.utf8.count <= Self.maximumMessageBytes, item.notice.utf8.count <= 2048 else { throw chatInputError("Invalid saved message") }
                if item.writing {
                    value.conversations[index].messages[message].writing = false
                    value.conversations[index].messages[message].interrupted = true
                }
            }
        }
        return value
    }
}

func chatInputError(_ text: String) -> NSError {
    NSError(domain:"TokletChat",code:1,userInfo:[NSLocalizedDescriptionKey:text])
}

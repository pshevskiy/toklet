import SwiftUI
import AppKit
import Combine
import Darwin

@MainActor
final class MacApplicationDelegate: NSObject, NSApplicationDelegate {
    var coordinator: MacCoordinator?
    private var terminating = false
    func applicationDidFinishLaunching(_ notification: Notification) {
        NotificationCenter.default.addObserver(self,selector:#selector(windowWillClose(_:)),name:NSWindow.willCloseNotification,object:nil)
        coordinator?.record("app.launched")
    }
    // AppKit sends NSWindow notifications on the UI thread. No owner titles or
    // user text enter this optional diagnostic stream.
    @objc private func windowWillClose(_ notification: Notification) {
        if let window = notification.object as? NSWindow, !window.isSheet, window.canBecomeMain {
            coordinator?.record("window.willClose")
        }
    }
    func applicationWillTerminate(_ notification: Notification) {
        coordinator?.record("app.terminated")
        NotificationCenter.default.removeObserver(self)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let coordinator else { return .terminateNow }
        guard !terminating else { return .terminateLater }
        terminating = true
        Task { @MainActor in
            let closed = await coordinator.shutdown(terminating:true)
            if !closed { terminating = false }
            sender.reply(toApplicationShouldTerminate:closed)
        }
        return .terminateLater
    }
}

@main
@MainActor
struct TokletMacApp: App {
    @NSApplicationDelegateAdaptor(MacApplicationDelegate.self) private var delegate
    @StateObject private var coordinator: MacCoordinator
    private let timer = Timer.publish(every:0.1,on:.main,in:.common).autoconnect()

    init() {
        // A separate root makes explicit lab UI runs independent of owner chats
        // and model caches. Normal launches keep the stable Application Support
        // location; no prior Documents cache is moved or deleted.
        let standard = FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("Toklet",isDirectory:true)
        let root: URL
        do { root = try MacCoordinator.dataRoot(arguments:ProcessInfo.processInfo.arguments,standard:standard) }
        catch {
            try? FileHandle.standardError.write(contentsOf:Data((error.localizedDescription + "\n").utf8))
            Darwin.exit(64)
        }
        let tracing = ProcessInfo.processInfo.arguments.contains("--trace-ui")
        let started = ContinuousClock.now
        let owner = MacCoordinator(root:root,trace:{ event in
            guard tracing else { return }
            let line = "TOKLET_MAC_EVENT elapsed=\(started.duration(to:.now)) event=\(event)\n"
            try? FileHandle.standardOutput.write(contentsOf:Data(line.utf8))
        })
        _coordinator = StateObject(wrappedValue:owner)
        delegate.coordinator = owner
    }
    var body: some Scene {
        Window("Toklet",id:"main") {
            MacRootView(coordinator:coordinator)
                .onReceive(timer) { _ in coordinator.harness.poll() }
        }.defaultSize(width:760,height:780)
            .commands { MacWindowCommands() }
    }
}

private struct MacWindowCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    var body: some Commands {
        CommandGroup(replacing:.newItem) {
            Button("Open Toklet") { openWindow(id:"main") }.keyboardShortcut("n",modifiers:[.command,.shift])
        }
    }
}

private struct MacRootView: View {
    @ObservedObject var coordinator: MacCoordinator
    @Environment(\.openWindow) private var openWindow
    var body: some View {
        FriendlyHarnessView(store:coordinator.harness,chat:coordinator.chat,hub:coordinator.hub)
            .frame(minWidth:560,minHeight:600)
            .disabled(coordinator.shuttingDown)
            .onAppear { coordinator.record("window.appear") }
            .task { await coordinator.reopen() }
            .onDisappear {
                coordinator.record("window.disappear")
                coordinator.windowClosed()
                Task { @MainActor in
                    if !(await coordinator.shutdown()) {
                        coordinator.record("window.recovery.afterFailedSaveOrClose")
                        openWindow(id:"main")
                    }
                }
            }
    }
}

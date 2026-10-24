import XCTest

@MainActor
final class PhoneUITests: XCTestCase {
    func capture(_ name: String, app: XCUIApplication) {
        let image = XCTAttachment(screenshot:app.screenshot()); image.name = name; image.lifetime = .keepAlways; add(image)
        let tree = XCTAttachment(string:app.debugDescription); tree.name = name + " tree"; tree.lifetime = .keepAlways; add(tree)
    }
    func visible(_ element: XCUIElement, app: XCUIApplication) {
        for _ in 0..<6 {
            if element.exists && element.isHittable { return }
            app.swipeUp()
        }
        XCTAssertTrue(element.exists && element.isHittable)
    }
    func choose(_ key: String, name: String, app: XCUIApplication) {
        let picker = app.buttons["chat-model-picker"]
        XCTAssertTrue(picker.waitForExistence(timeout:15)); picker.tap()
        XCTAssertTrue(app.navigationBars["Models"].waitForExistence(timeout:10))
        let use = app.buttons["hub-use-" + key].firstMatch
        XCTAssertTrue(use.waitForExistence(timeout:15)); visible(use,app:app)
        let ready = XCTNSPredicateExpectation(predicate:NSPredicate(format:"enabled == true"),object:use)
        XCTAssertEqual(XCTWaiter.wait(for:[ready],timeout:30),.completed)
        capture("Choose-" + key,app:app); use.tap()
        let loaded = XCTNSPredicateExpectation(predicate:NSPredicate(format:"label CONTAINS %@",name),object:picker)
        XCTAssertEqual(XCTWaiter.wait(for:[loaded],timeout:120),.completed)
        let closed = XCTNSPredicateExpectation(predicate:NSPredicate(format:"exists == false"),object:app.navigationBars["Models"])
        XCTAssertEqual(XCTWaiter.wait(for:[closed],timeout:10),.completed)
        capture("Active-" + key,app:app)
    }
    func reply(_ text: String, app: XCUIApplication) {
        let draft = app.descendants(matching:.any)["chat-draft"].firstMatch
        XCTAssertTrue(draft.waitForExistence(timeout:10)); draft.tap(); draft.typeText(text)
        let send = app.buttons["chat-send"]; XCTAssertTrue(send.isEnabled); send.tap()
        let done = XCTNSPredicateExpectation(predicate:NSPredicate(format:"label == %@","Send message"),object:send)
        XCTAssertEqual(XCTWaiter.wait(for:[done],timeout:180),.completed)
    }
    func expectCount(_ query: XCUIElementQuery,_ value: Int) {
        let changed = XCTNSPredicateExpectation(predicate:NSPredicate(format:"count == %d",value),object:query)
        XCTAssertEqual(XCTWaiter.wait(for:[changed],timeout:10),.completed)
    }
    func cancelConfirmation(app: XCUIApplication) {
        let cancel = app.sheets.buttons["Cancel"]
        if cancel.exists { cancel.tap(); return }
        // iOS can present a popover without a Cancel row. Dismiss it by
        // tapping an observed point outside its actual frame.
        let panel = app.sheets.matching(NSPredicate(format:"label BEGINSWITH %@","Delete")).firstMatch
        XCTAssertTrue(panel.exists)
        let window = app.windows.firstMatch
        let outside = CGPoint(x:window.frame.minX + 10,y:window.frame.midY)
        XCTAssertFalse(panel.frame.contains(outside))
        window.coordinate(withNormalizedOffset:CGVector(dx:0,dy:0.5)).withOffset(CGVector(dx:10,dy:0)).tap()
    }
    func testModelSwitching() throws {
        continueAfterFailure = false; executionTimeAllowance = 600
        let app = XCUIApplication(bundleIdentifier:"dev.toklet.harness")
        app.launchArguments = ["--ui-test-history"]; app.launch()
        choose("smol",name:"SmolLM2",app:app)
        reply("Hello. Reply with one short sentence.",app:app)
        choose("qwen",name:"Qwen2.5",app:app)
        reply("Write a Python function add(a, b). Put it in a fenced python code block.",app:app)
        let copies = app.buttons.matching(identifier:"Copy code")
        XCTAssertTrue(copies.firstMatch.waitForExistence(timeout:10))
        XCTAssertTrue(copies.allElementsBoundByIndex.contains(where:{ $0.isHittable }),"A displayed code block has a visible Copy action")
        capture("Qwen-code",app:app)
        choose("smol",name:"SmolLM2",app:app)
        reply("Hello again. Reply briefly.",app:app)
        choose("qwen",name:"Qwen2.5",app:app)
        choose("smol",name:"SmolLM2",app:app)
        // Selecting the already active model returns immediately without reload.
        choose("smol",name:"SmolLM2",app:app)
        app.terminate()
    }
}

extension PhoneUITests {
    func testRankedRecommendations() throws {
        continueAfterFailure = false; executionTimeAllowance = 240
        let app = XCUIApplication(bundleIdentifier:"dev.toklet.harness")
        app.launchArguments = ["--ui-test-history"]; app.launch()
        app.buttons["find-best-model"].tap()
        XCTAssertTrue(app.navigationBars["Models"].waitForExistence(timeout:10))
        let first = app.staticTexts["hub-rank-1"]
        XCTAssertTrue(first.waitForExistence(timeout:10))
        let confidence = app.staticTexts["hub-ranking-confidence"]
        XCTAssertTrue(confidence.label.contains("not quality tests or measured speed"))
        let context = app.staticTexts["hub-ranking-context"]
        XCTAssertTrue(context.label.contains("512 total tokens") && context.label.contains("256 reply tokens"))
        app.buttons["Fast replies"].firstMatch.tap()
        let fast = XCTNSPredicateExpectation(predicate:NSPredicate(format:"label CONTAINS %@","SmolLM2"),object:first)
        XCTAssertEqual(XCTWaiter.wait(for:[fast],timeout:10),.completed)
        let details = app.buttons["Why this rank?"].firstMatch; visible(details,app:app); details.tap()
        XCTAssertTrue(app.staticTexts["hub-rank-breakdown-smol"].waitForExistence(timeout:10))
        capture("Ranking-fast-replies",app:app)
        app.swipeDown()
        app.buttons["Coding"].firstMatch.tap()
        let coding = XCTNSPredicateExpectation(predicate:NSPredicate(format:"label CONTAINS %@","Qwen2.5"),object:first)
        XCTAssertEqual(XCTWaiter.wait(for:[coding],timeout:10),.completed)
        capture("Ranking-coding",app:app)
        let use = app.buttons["hub-use-qwen"].firstMatch; visible(use,app:app); use.tap()
        let loaded = XCTNSPredicateExpectation(predicate:NSPredicate(format:"label CONTAINS %@","Qwen2.5"),object:app.buttons["chat-model-picker"])
        XCTAssertEqual(XCTWaiter.wait(for:[loaded],timeout:120),.completed)
        reply("Hello. Reply in one sentence.",app:app)
        capture("Ranking-model-used",app:app)
        app.terminate()
    }
    func testBenchmarkAndGemma4Search() throws {
        continueAfterFailure = false; executionTimeAllowance = 600
        let app = XCUIApplication(bundleIdentifier:"dev.toklet.harness")
        app.launchArguments = ["--ui-test-history"]; app.launch()
        choose("smol",name:"SmolLM2",app:app)
        app.buttons["local-benchmark"].tap()
        XCTAssertTrue(app.buttons["benchmark-run"].waitForExistence(timeout:10))
        XCTAssertTrue(app.buttons["benchmark-run"].isEnabled); app.buttons["benchmark-run"].tap()
        let complete = app.staticTexts["benchmark-status"]
        let finished = XCTNSPredicateExpectation(predicate:NSPredicate(format:"label BEGINSWITH %@","Complete."),object:complete)
        XCTAssertEqual(XCTWaiter.wait(for:[finished],timeout:180),.completed)
        for index in 1...3 {
            let row = app.staticTexts.matching(NSPredicate(format:"label BEGINSWITH %@","Run \(index) · ")).firstMatch
            visible(row,app:app); XCTAssertTrue(row.label.contains("tokens"))
            capture("Benchmark-run-\(index)",app:app)
        }
        app.buttons["Close"].tap()
        for key in ["monitor-memory","monitor-ttft","monitor-rate"] {
            let reading = app.descendants(matching:.any)[key].firstMatch
            XCTAssertTrue(reading.exists)
            XCTAssertNotNil(reading.label.range(of:"[0-9]",options:.regularExpression),"A measured value is shown for " + key)
        }
        capture("Measured-phone-dashboard",app:app)
        app.buttons["chat-model-picker"].tap()
        let query = app.textFields["hub-search-query"]; visible(query,app:app); query.tap(); query.typeText("gemma-4")
        app.buttons["hub-search"].tap()
        let searched = XCTNSPredicateExpectation(predicate:NSPredicate(format:"exists == false"),object:app.buttons["hub-cancel"])
        XCTAssertEqual(XCTWaiter.wait(for:[searched],timeout:45),.completed)
        let unsupported = app.staticTexts.matching(NSPredicate(format:"label BEGINSWITH %@","Unavailable for chat:"))
        for _ in 0..<16 {
            if unsupported.allElementsBoundByIndex.contains(where:{ $0.isHittable }) { break }
            app.collectionViews.firstMatch.swipeUp()
        }
        capture("Gemma4-search-state",app:app)
        XCTAssertTrue(unsupported.allElementsBoundByIndex.contains(where:{ $0.isHittable }),"Gemma 4 search must show its unsupported status")
        let gemma = app.staticTexts.matching(NSPredicate(format:"label CONTAINS[c] %@","gemma-4")).firstMatch
        XCTAssertTrue(gemma.exists)
        capture("Gemma4-search-unsupported",app:app)
        app.buttons["Close"].tap()
        XCTAssertTrue(app.buttons["chat-model-picker"].label.contains("SmolLM2"),"Unsupported search preserves the loaded model")
        app.terminate()
    }
    func testDashboardControls() throws {
        continueAfterFailure = false; executionTimeAllowance = 240
        let app = XCUIApplication(bundleIdentifier:"dev.toklet.harness")
        app.launchArguments = ["--ui-test-history"]; app.launch()
        capture("Dashboard-initial",app:app)
        XCTAssertTrue(app.descendants(matching:.any)["monitor-memory"].firstMatch.waitForExistence(timeout:10))
        XCTAssertTrue(app.descendants(matching:.any)["monitor-ttft"].firstMatch.exists)
        XCTAssertTrue(app.descendants(matching:.any)["monitor-rate"].firstMatch.exists)
        capture("Dashboard",app:app)
        app.buttons["find-best-model"].tap()
        XCTAssertTrue(app.navigationBars["Models"].waitForExistence(timeout:10))
        let explanation = app.staticTexts.matching(NSPredicate(format:"label CONTAINS %@","Toklet currently supports 3 downloadable chat models")).firstMatch
        XCTAssertTrue(explanation.waitForExistence(timeout:10))
        capture("Device-fit-shortlist",app:app); app.buttons["Close"].tap()
        app.buttons["local-benchmark"].tap()
        XCTAssertTrue(app.navigationBars["Benchmark"].waitForExistence(timeout:10))
        XCTAssertTrue(app.buttons["benchmark-run"].exists && !app.buttons["benchmark-run"].isEnabled)
        capture("Benchmark-needs-model",app:app); app.buttons["Close"].tap()
        app.buttons["New chat"].tap(); app.buttons["New chat"].tap()
        app.buttons["Conversations"].tap()
        let rows = app.buttons.matching(NSPredicate(format:"identifier BEGINSWITH %@","history-select-"))
        let trash = app.buttons.matching(NSPredicate(format:"identifier BEGINSWITH %@ AND identifier != %@","history-delete-","history-delete-all"))
        XCTAssertEqual(rows.count,3); XCTAssertEqual(trash.count,3)
        trash.firstMatch.tap(); cancelConfirmation(app:app); expectCount(rows,3)
        trash.firstMatch.tap()
        XCTAssertTrue(app.sheets.buttons["Delete conversation"].waitForExistence(timeout:5)); app.sheets.buttons["Delete conversation"].tap()
        expectCount(rows,2); capture("Single-dialog-deleted",app:app)
        let all = app.buttons["history-delete-all"]; visible(all,app:app); all.tap()
        cancelConfirmation(app:app); expectCount(rows,2)
        all.tap(); app.sheets.buttons["Delete all conversations"].tap()
        expectCount(rows,1)
        capture("Dialogs-deleted",app:app)
        app.buttons["Close"].tap(); app.buttons["Settings"].tap(); app.buttons["expert-mode"].tap()
        XCTAssertTrue(app.navigationBars["Expert mode"].waitForExistence(timeout:10))
        XCTAssertTrue(app.descendants(matching:.any)["monitor-memory"].firstMatch.waitForExistence(timeout:10))
        capture("Expert-monitoring",app:app)
        app.buttons["Back to Chat"].tap()
        XCTAssertTrue(app.buttons["chat-model-picker"].waitForExistence(timeout:10))
        app.terminate()
    }
    func testReplyLimitAndExpert() throws {
        continueAfterFailure = false; executionTimeAllowance = 600
        let app = XCUIApplication(bundleIdentifier:"dev.toklet.harness")
        app.launchArguments = ["--ui-test-history"]; app.launch()
        app.buttons["Settings"].tap()
        let limit = app.buttons.matching(NSPredicate(format:"label BEGINSWITH %@","Response limit")).firstMatch
        XCTAssertTrue(limit.waitForExistence(timeout:10))
        for _ in 0..<2 {
            if limit.label.contains("256 tokens") { break }
            let choice = app.buttons["256 tokens"]
            if !choice.exists { limit.tap() }
            let tappable = XCTNSPredicateExpectation(predicate:NSPredicate(format:"hittable == true"),object:choice)
            XCTAssertEqual(XCTWaiter.wait(for:[tappable],timeout:10),.completed); choice.tap()
            let selected = XCTNSPredicateExpectation(predicate:NSPredicate(format:"label CONTAINS %@","256 tokens"),object:limit)
            if XCTWaiter.wait(for:[selected],timeout:3) == .completed { break }
        }
        XCTAssertTrue(limit.label.contains("256 tokens"),"Reply preference is committed before closing Settings")
        capture("Reply-limit-256",app:app); app.buttons["Close"].tap()
        let settingsClosed = XCTNSPredicateExpectation(predicate:NSPredicate(format:"exists == false"),object:app.navigationBars["Settings"])
        XCTAssertEqual(XCTWaiter.wait(for:[settingsClosed],timeout:10),.completed)
        choose("qwen",name:"Qwen2.5",app:app)
        reply("Write a long adventure story about a cat exploring an old city. Include many scenes, detailed descriptions, and dialogue. Continue the story for at least 2000 words.",app:app)
        let notice = app.staticTexts.matching(NSPredicate(format:"label BEGINSWITH %@","Response limit reached.")).firstMatch
        XCTAssertTrue(notice.waitForExistence(timeout:15)); capture("Long-reply-limit",app:app)
        app.buttons["Settings"].tap(); app.buttons["expert-mode"].tap()
        XCTAssertTrue(app.navigationBars["Expert mode"].waitForExistence(timeout:20))
        app.buttons["Tiny fixture"].tap()
        let load = app.buttons["Load"]; visible(load,app:app); load.tap()
        let prompt = app.buttons["Prompt"]; visible(prompt,app:app)
        let ready = XCTNSPredicateExpectation(predicate:NSPredicate(format:"enabled == true"),object:prompt)
        XCTAssertEqual(XCTWaiter.wait(for:[ready],timeout:30),.completed); prompt.tap()
        let ids = app.staticTexts["47, 47"].firstMatch; visible(ids,app:app)
        XCTAssertTrue(ids.exists); capture("Expert-highlighted-output",app:app)
        let copy = app.buttons["Copy code"].firstMatch; visible(copy,app:app)
        XCTAssertTrue(copy.isEnabled)
        app.buttons["Back to Chat"].tap()
        XCTAssertTrue(app.buttons["chat-model-picker"].waitForExistence(timeout:20))
        app.terminate()
    }
}

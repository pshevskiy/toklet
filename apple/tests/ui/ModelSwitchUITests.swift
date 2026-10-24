import XCTest

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
    func testReplyLimitAndExpert() throws {
        continueAfterFailure = false; executionTimeAllowance = 600
        let app = XCUIApplication(bundleIdentifier:"dev.toklet.harness")
        app.launchArguments = ["--ui-test-history"]; app.launch()
        app.buttons["Settings"].tap()
        let limit = app.descendants(matching:.any).matching(NSPredicate(format:"label BEGINSWITH %@","Response limit")).firstMatch
        XCTAssertTrue(limit.waitForExistence(timeout:10)); limit.tap()
        XCTAssertTrue(app.buttons["256 tokens"].waitForExistence(timeout:10)); app.buttons["256 tokens"].tap()
        capture("Reply-limit-256",app:app); app.buttons["Close"].tap()
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

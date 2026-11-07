import Foundation

// Portable no-model tests: fixed records, clock scopes and pressure behavior.
@main
struct UIPublicationMain {
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func rows(_ store: NativeRequestSummaryStore) throws -> [[String:Any]] {
        let (records,dropped) = store.snapshot()
        return try NativeRequestSummaryStore.export(records,dropped:dropped,uiErrors:store.uiUpdateErrors)
            .split(separator:10).map { try JSONSerialization.jsonObject(with:Data($0)) as! [String:Any] }
    }
    static func main() throws {
        try require(NativeUIPublication(submitted:201,workerSubmitted:200) == nil,"Future UI start accepted")
        let store = NativeRequestSummaryStore(), id = UUID()
        var record = store.begin(id:id,maximum:12,generation:1,now:200,context:nil,configuration:nil,uiSubmitted:100)
        record.startedUptimeNanoseconds = 220
        record.observeCallback(0.01,uptime:250)
        record.finish(observation:nil,failure:false,status:0,reason:2,now:300)
        store.append(record)
        store.publishUI(id:id,submitted:100,outcome:.published,selected:true,nonempty:true,now:299)
        try require(store.uiUpdateErrors == 1 && store.snapshot().0[0].ui?.outcome == nil,"Bad UI end mutated record")
        store.publishUI(id:id,submitted:100,outcome:.published,selected:true,nonempty:true,now:500)
        let published = store.snapshot().0[0]
        try require(published.outcome == .stopToken && published.completedUptimeNanoseconds == 300 &&
                    published.requestMilliseconds == 0.0001,"UI changed native request semantics")
        try require(published.ui?.preparationMilliseconds == 0.0001 &&
                    published.ui?.callbackTTFTMilliseconds == 0.00015 &&
                    published.ui?.workerToHandlingMilliseconds == 0.0002 &&
                    published.ui?.publicationMilliseconds == 0.0004 &&
                    published.ui?.firstAnswerPublicationMilliseconds == 0.0004,"UI clock/formula mismatch")
        store.publishUI(id:id,submitted:100,outcome:.workerRejected,selected:false,nonempty:false,now:600)
        store.publishUI(id:UUID(),submitted:100,outcome:.published,selected:true,nonempty:true,now:600)
        store.publishUI(id:id,submitted:101,outcome:.published,selected:true,nonempty:true,now:600)
        try require(store.uiUpdateErrors == 4 && store.snapshot().0[0].ui?.handledUptimeNanoseconds == 500,
                    "Duplicate/stale receipt overwrote publication")
        let export = try rows(store)
        try require(export[0]["uiUpdateErrors"] as? Int == 4 && export[1]["ui"] != nil,"UI coverage/export missing")
        for outcome in [NativeUIPublication.Outcome.workerRejected,.submissionRejected,.workerClosed,.messageMissing,.published] {
            var value = NativeUIPublication(submitted:0,workerSubmitted:0)!
            try require(value.finish(outcome:outcome,selected:false,nonempty:false,workerCompleted:0,now:10),"UI outcome rejected")
            try require(value.firstNonemptyPublicationMilliseconds == nil && value.firstAnswerPublicationMilliseconds == nil,
                        "Empty/background UI became visible text")
            try require((value.publicationUptimeNanoseconds == nil) == (outcome == .messageMissing),"Missing-message publication fabricated")
        }
        var background = NativeUIPublication(submitted:1,workerSubmitted:1)!
        try require(background.finish(outcome:.published,selected:false,nonempty:true,workerCompleted:2,now:3) &&
                    background.publicationMilliseconds != nil && background.firstNonemptyPublicationMilliseconds == nil,
                    "Background conversation claimed selected text")
        var rejected = NativeUIPublication(submitted:1,workerSubmitted:1)!
        try require(!rejected.finish(outcome:.workerRejected,selected:true,nonempty:true,workerCompleted:2,now:3),
                    "Rejection claimed response text")
        let ambiguous = NativeRequestSummaryStore()
        ambiguous.append(record); ambiguous.append(record)
        ambiguous.publishUI(id:id,submitted:100,outcome:.published,selected:true,nonempty:true,now:500)
        try require(ambiguous.uiUpdateErrors == 1 && ambiguous.snapshot().0.allSatisfy({ $0.ui?.outcome == nil }),
                    "Ambiguous identities updated a record")
        let invalid = NativeRequestSummaryStore()
        let invalidStart = invalid.begin(id:UUID(),maximum:1,generation:0,now:1,context:nil,configuration:nil,uiSubmitted:2)
        try require(invalidStart.ui == nil && invalid.uiUpdateErrors == 1,"Invalid start not explicitly counted")
        let pressure = NativeRequestSummaryStore()
        pressure.append(record)
        for index in 0..<32 {
            var row = pressure.begin(id:UUID(),maximum:12,generation:UInt64(index),now:200,context:nil,configuration:nil,uiSubmitted:100)
            row.finish(observation:nil,failure:false,status:0,reason:1,now:300)
            pressure.append(row)
        }
        pressure.publishUI(id:id,submitted:100,outcome:.published,selected:true,nonempty:true,now:500)
        try require(pressure.snapshot().0.count == 32 && pressure.snapshot().1 == 1 && pressure.uiUpdateErrors == 1,
                    "UI pressure did not report lost correlation")
        for row in pressure.snapshot().0 {
            pressure.publishUI(id:row.requestID,submitted:100,outcome:.published,selected:true,nonempty:true,now:500)
        }
        let bytes = try rows(pressure)
        try require(bytes.count == 33 && MemoryLayout<NativeRequestSummary>.stride < 16*1024,
                    "UI records exceed bounded collector")
        let disabled = try NativeWorker()
        try require(disabled.requestSummaryUISubmissionTime() == nil,"Disabled mode started UI timing")
        disabled.close(); try require(disabled.waitUntilClosed(seconds:30),"Disabled worker did not close")
        print("UI_PUBLICATION_UNIT_PASS clock/formula/status/correlation/pressure/export/disabled")
    }
}

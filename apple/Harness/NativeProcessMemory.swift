import Foundation
import Darwin
#if os(iOS)
import os
#endif

private func nativeProcessMemoryError(_ message: String) -> NSError {
    NSError(domain:"TokletMeasurement",code:1,userInfo:[NSLocalizedDescriptionKey:message])
}

struct NativeFootprint: Codable, Sendable {
    let physicalBytes: UInt64
    let lifetimePeakPhysicalBytes: UInt64
    let returnedNaturalCount: UInt32

    static var macProcessBudget: UInt64 {
        min(ProcessInfo.processInfo.physicalMemory / 2,6 * 1024 * 1024 * 1024)
    }

    // The same process-headroom policy used for model admission. This is an
    // observation, not free system RAM, an allocation reservation or a peak.
    static func readHeadroom() throws -> (availableBytes: UInt64, instrument: String) {
        #if os(iOS)
        return (UInt64(os_proc_available_memory()),
                "os_proc_available_memory: current process dirty-memory allowance")
        #else
        let budget = macProcessBudget
        let footprint = try read().physicalBytes
        return (budget > footprint ? budget - footprint : 0,
                "Configured Mac process allowance min(RAM/2,6GiB) minus TASK_VM_INFO footprint; not free system RAM")
        #endif
    }

    static func read() throws -> NativeFootprint {
        var info = task_vm_info_data_t()
        let capacity = MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        var count = mach_msg_type_number_t(capacity)
        // The initialized C record owns capacity integer_t slots. task_info
        // writes at most the supplied count for this task, during this rebound
        // lifetime only. No Swift access aliases that storage during the call.
        let status = withUnsafeMutablePointer(to:&info) { pointer in
            pointer.withMemoryRebound(to:integer_t.self,capacity:capacity) {
                task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
            }
        }
        // SDK revision macros containing sizeof are not imported into Swift.
        // Derive byte coverage from this SDK's actual C layout instead of a
        // hardcoded revision/count. Both physical fields precede peakEnd.
        guard let offset = MemoryLayout<task_vm_info_data_t>.offset(of:\.ledger_phys_footprint_peak) else {
            throw nativeProcessMemoryError("Unavailable task memory layout")
        }
        let peakEnd = offset + MemoryLayout<Int64>.size
        guard status == KERN_SUCCESS, count <= capacity,
              Int(count) * MemoryLayout<integer_t>.size >= peakEnd,
              info.ledger_phys_footprint_peak >= 0 else {
            throw nativeProcessMemoryError("TASK_VM_INFO physical peak unavailable: status \(status), count \(count)")
        }
        return NativeFootprint(physicalBytes:info.phys_footprint,
                               lifetimePeakPhysicalBytes:UInt64(info.ledger_phys_footprint_peak),
                               returnedNaturalCount:count)
    }
}

// Two read-only checkpoints, not a timer or an admission mechanism. A strictly
// increased lifetime high-water mark proves that a process peak was established
// between the reads. Equal/decreasing/inconsistent counters cannot isolate it.
enum NativeMemoryObservation: Sendable {
    case available(NativeFootprint)
    case unavailable(String)
    static func capture(read: () throws -> NativeFootprint = NativeFootprint.read) -> Self {
        do { return .available(try read()) }
        catch { return .unavailable(String(error.localizedDescription.prefix(256))) }
    }
}

struct NativeLoadMemorySnapshot: Sendable {
    let attemptID: UUID
    let loadRequestID: UUID?
    let requestedCapacity, requestedLoadUpper: UInt32
    let completed: Bool
    let modelInstance, initialSessionInstance: UInt64?
    let before, after: NativeMemoryObservation
    var establishedProcessPeakBytes: UInt64? {
        guard case .available(let start) = before, case .available(let end) = after,
              start.lifetimePeakPhysicalBytes >= start.physicalBytes,
              end.lifetimePeakPhysicalBytes >= end.physicalBytes,
              end.lifetimePeakPhysicalBytes > start.lifetimePeakPhysicalBytes else { return nil }
        return end.lifetimePeakPhysicalBytes
    }
    func matches(model: NativeModelContext?, loadRequestID current: UUID?) -> Bool {
        guard completed, let model, let modelInstance, modelInstance > 0,
              let initialSessionInstance, initialSessionInstance > 0 else { return false }
        return model.modelInstance == modelInstance && current == loadRequestID
    }
    var peakExplanation: String {
        guard case .available(let start) = before, case .available(let end) = after else {
            return "Process memory reading unavailable; loading still uses normal admission checks."
        }
        guard start.lifetimePeakPhysicalBytes >= start.physicalBytes,
              end.lifetimePeakPhysicalBytes >= end.physicalBytes,
              end.lifetimePeakPhysicalBytes >= start.lifetimePeakPhysicalBytes else {
            return "Inconsistent memory counters; load-interval peak is undetermined."
        }
        return establishedProcessPeakBytes == nil
            ? "Load-interval peak undetermined; the prior process peak was not exceeded."
            : "The process established this physical peak during the load attempt. Other process activity contributes."
    }
}

struct NativeLoadMemoryProbe {
    let attemptID = UUID()
    let before: NativeMemoryObservation
    static func begin(enabled: Bool, read: () throws -> NativeFootprint = NativeFootprint.read) -> Self? {
        guard enabled else { return nil }
        return Self(before:.capture(read:read))
    }
    func finish(loadRequestID: UUID?, capacity: UInt32, upper: UInt32, completed: Bool,
                model: UInt64?, session: UInt64?, read: () throws -> NativeFootprint = NativeFootprint.read) -> NativeLoadMemorySnapshot {
        NativeLoadMemorySnapshot(attemptID:attemptID,loadRequestID:loadRequestID,requestedCapacity:capacity,
            requestedLoadUpper:upper,completed:completed,modelInstance:completed ? model : nil,
            initialSessionInstance:completed ? session : nil,before:before,after:.capture(read:read))
    }
}

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

    // The same process-headroom policy used for model admission. This is an
    // observation, not free system RAM, an allocation reservation or a peak.
    static func readHeadroom() throws -> (availableBytes: UInt64, instrument: String) {
        #if os(iOS)
        return (UInt64(os_proc_available_memory()),
                "os_proc_available_memory: current process dirty-memory allowance")
        #else
        let budget = min(ProcessInfo.processInfo.physicalMemory / 2,6 * 1024 * 1024 * 1024)
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

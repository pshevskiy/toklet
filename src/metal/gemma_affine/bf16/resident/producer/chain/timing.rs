//! Optional properties of the already completed candidate command.
use super::Metrics;

pub(super) const NOT_OBSERVED: &str = "not_observed_after_successful_wait";
pub(super) const SCOPE: &str =
    "candidate_whole_command_including_caller_work_excluding_reference_replay";

/// Called only after the existing wait reports Completed and no Metal error.
/// Bad/unavailable properties affect this observation, never publication.
pub(super) fn record(
    metrics: &mut Metrics,
    epoch: u64,
    command_identity: usize,
    start: f64,
    end: f64,
) {
    metrics.candidate_gpu_timing_completed = true;
    metrics.candidate_gpu_timing_epoch = Some(epoch);
    metrics.candidate_gpu_timing_command_identity = Some(command_identity);
    metrics.candidate_gpu_timing_scope = Some(SCOPE);
    // JSON cannot carry NaN/Inf. Preserve finite raw values, with an explicit
    // reason whenever the raw interval cannot provide a duration.
    metrics.candidate_gpu_start_seconds = start.is_finite().then_some(start);
    metrics.candidate_gpu_end_seconds = end.is_finite().then_some(end);
    let (seconds, reason) = if !start.is_finite() || !end.is_finite() {
        (None, Some("nonfinite_timestamp"))
    } else if start == 0. || end == 0. {
        (None, Some("zero_timestamp"))
    } else if start < 0. || end < 0. {
        (None, Some("nonpositive_timestamp"))
    } else if end <= start {
        (None, Some("unordered_timestamp"))
    } else {
        let seconds = end - start;
        if seconds.is_finite() && seconds > 0. {
            (Some(seconds), None)
        } else {
            (None, Some("invalid_duration"))
        }
    };
    metrics.candidate_completed_gpu_seconds = seconds;
    metrics.candidate_gpu_timing_unavailable_reason = reason;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_candidate_timing_is_bound_and_excludes_reference_work() {
        let mut m = Metrics {
            epoch: 7,
            command_identity: 91,
            candidate_commands: 1,
            candidate_waits: 1,
            encode_seconds: 0.01,
            wait_seconds: 0.02,
            complete_seconds: 0.03,
            publication_seconds: 0.04,
            ..Metrics::default()
        };
        record(&mut m, 7, 91, 12.25, 12.75);
        assert_eq!(m.candidate_gpu_start_seconds, Some(12.25));
        assert_eq!(m.candidate_gpu_end_seconds, Some(12.75));
        assert_eq!(m.candidate_completed_gpu_seconds, Some(0.5));
        assert_eq!(m.candidate_gpu_timing_epoch, Some(m.epoch));
        assert_eq!(
            m.candidate_gpu_timing_command_identity,
            Some(m.command_identity)
        );
        assert!(m.candidate_gpu_timing_completed);
        assert_eq!(m.candidate_gpu_timing_unavailable_reason, None);
        assert_eq!(m.candidate_gpu_timing_scope, Some(SCOPE));
        m.replay_count = 1;
        m.reference_commands = Some(27);
        m.reference_waits = Some(27);
        assert_eq!(m.candidate_completed_gpu_seconds, Some(0.5));
        assert_eq!((m.candidate_commands, m.candidate_waits), (1, 1));
        assert_eq!(
            (
                m.encode_seconds,
                m.wait_seconds,
                m.complete_seconds,
                m.publication_seconds
            ),
            (0.01, 0.02, 0.03, 0.04)
        );
    }

    #[test]
    fn unavailable_or_invalid_native_properties_never_fabricate_duration() {
        for (start, end, reason) in [
            (0., 0., "zero_timestamp"),
            (-0., 1., "zero_timestamp"),
            (1., 0., "zero_timestamp"),
            (-1., 1., "nonpositive_timestamp"),
            (1., -1., "nonpositive_timestamp"),
            (1., 1., "unordered_timestamp"),
            (2., 1., "unordered_timestamp"),
            (f64::NAN, 1., "nonfinite_timestamp"),
            (1., f64::INFINITY, "nonfinite_timestamp"),
            (f64::NEG_INFINITY, 1., "nonfinite_timestamp"),
        ] {
            let mut m = Metrics::default();
            record(&mut m, 7, 91, start, end);
            assert!(m.candidate_gpu_timing_completed);
            assert_eq!(m.candidate_completed_gpu_seconds, None);
            assert_eq!(m.candidate_gpu_timing_unavailable_reason, Some(reason));
            assert_eq!(m.candidate_gpu_start_seconds.is_some(), start.is_finite());
            assert_eq!(m.candidate_gpu_end_seconds.is_some(), end.is_finite());
            assert_eq!(
                (m.candidate_commands, m.candidate_waits, m.replay_count),
                (0, 0, 0)
            );
        }
    }
}

//! Constructor-only diagnostic capture of one committed evaluation. GPU stages
//! overlap host waits; their durations are never added to exclusive host scopes.
use super::*;
use crate::{
    metal::{
        gemma_affine::project::{Observation, TileAttribution, TileRecord},
        profile::{Profile, ProfileEvent, Trace},
    },
    mlx::gemma4::bf16::execution::OperationId,
};

pub(super) const ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_TILES: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct ProfileOptions {
    pub request_id: [u8; 16],
    pub evaluation_ordinal: usize,
}
impl ProfileOptions {
    pub(super) fn validate(self) -> Result<Self> {
        if !(1..=512).contains(&self.evaluation_ordinal) {
            return Err(Error::Options(
                "BF16 profile evaluation ordinal outside bounds",
            ));
        }
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureState {
    Armed,
    Recording,
    Complete,
    Invalid,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ProfileStatus {
    pub state: CaptureState,
    pub evaluations_seen: usize,
    pub tile_capacity: usize,
    pub tiles_recorded: usize,
    pub gathers_recorded: usize,
    pub overflowed: bool,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Evaluation {
    pub position: usize,
    pub input_rows: usize,
    pub output_rows: usize,
    pub committed_positions: usize,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct CpuGather {
    pub operation: OperationId,
    pub source_matrix_slot: usize,
    pub rows: usize,
    pub columns: usize,
    /// Active BF16 payload, not measured storage/DRAM traffic or page reads.
    pub active_source_bytes: usize,
    pub host_begin_seconds: f64,
    pub host_end_seconds: f64,
}
#[derive(Debug, serde::Serialize)]
pub struct AttributedTile {
    pub record: TileRecord,
    pub event: ProfileEvent,
}
#[derive(Debug, serde::Serialize)]
pub struct FrameProfile {
    pub schema_version: u32,
    pub request_id: [u8; 16],
    pub evaluation_ordinal: usize,
    pub evaluation: Evaluation,
    pub status: ProfileStatus,
    pub clocks: Profile,
    pub tiles: Vec<AttributedTile>,
    pub cpu_gathers: [CpuGather; 2],
    pub frame_wall_seconds: f64,
    pub exclusive_host_seconds: f64,
    pub clock_bracket_seconds: f64,
    pub unattributed_host_seconds: f64,
    pub diagnostic_allowance_bytes: usize,
}

pub(super) struct Bf16Trace {
    options: ProfileOptions,
    trace: Trace,
    records: Vec<TileRecord>,
    gathers: [Option<CpuGather>; 2],
    status: ProfileStatus,
    evaluation: Option<Evaluation>,
    wall: f64,
    end_bracket: f64,
}
impl Bf16Trace {
    #[cfg(test)]
    pub(crate) fn limit_events_for_test(&mut self, capacity: usize) {
        assert!(self.status.state == CaptureState::Armed && capacity <= self.status.tile_capacity);
        self.status.tile_capacity = capacity;
    }
    pub(super) fn new(source: &Bf16Weights, gpu: &Gpu, options: ProfileOptions) -> Result<Self> {
        let options = options.validate()?;
        let capacity = source
            .matrices
            .iter()
            .flatten()
            .try_fold(0usize, |n, m| add(n, m.tiles.len()))?;
        if !(1..=MAX_TILES).contains(&capacity) {
            return Err(Error::Options("BF16 profile tile capacity outside bounds"));
        }
        let records = reserve::<TileRecord>(capacity)?;
        Ok(Self {
            options,
            trace: autoreleasepool(|_| Trace::new(gpu, capacity))?,
            records,
            gathers: [None; 2],
            status: ProfileStatus {
                state: CaptureState::Armed,
                evaluations_seen: 0,
                tile_capacity: capacity,
                tiles_recorded: 0,
                gathers_recorded: 0,
                overflowed: false,
            },
            evaluation: None,
            wall: 0.,
            end_bracket: 0.,
        })
    }
    pub(super) fn begin(&mut self, gpu: &Gpu, position: usize, rows: usize, last_row_only: bool) {
        self.status.evaluations_seen = self.status.evaluations_seen.saturating_add(1);
        if self.status.state != CaptureState::Armed
            || self.status.evaluations_seen != self.options.evaluation_ordinal
        {
            return;
        }
        self.trace.reset(gpu);
        self.evaluation = Some(Evaluation {
            position,
            input_rows: rows,
            output_rows: if last_row_only { 1 } else { rows },
            committed_positions: 0,
        });
        self.status.state = CaptureState::Recording;
    }
    pub(super) fn finish(&mut self, gpu: &Gpu, committed: Option<usize>) {
        if self.status.state != CaptureState::Recording {
            return;
        }
        let valid = self
            .evaluation
            .is_some_and(|e| e.position.checked_add(e.input_rows) == committed)
            && committed.is_some()
            && !self.status.overflowed
            && self.records.len() == self.status.tile_capacity
            && self.gathers.iter().all(Option::is_some)
            && self.records.iter().all(|r| {
                r.dispatches_encoded == 1 && r.commands_submitted == 1 && r.host_wait_calls == 1
            });
        if !valid {
            self.status.state = CaptureState::Invalid;
            return;
        }
        self.end_bracket = self.trace.complete(gpu);
        self.wall = self.trace.elapsed();
        self.evaluation.as_mut().unwrap().committed_positions = committed.unwrap();
        self.status.state = CaptureState::Complete;
    }
    pub(super) fn replace_history(&mut self) {
        if self.status.evaluations_seen != 0 {
            self.status.state = CaptureState::Invalid;
        }
    }
    pub(super) fn status(&self) -> ProfileStatus {
        ProfileStatus {
            tiles_recorded: self.records.len(),
            gathers_recorded: self.gathers.iter().flatten().count(),
            ..self.status
        }
    }
    pub(super) fn observation(&mut self, attribution: TileAttribution) -> Option<Observation<'_>> {
        (self.status.state == CaptureState::Recording).then_some(Observation {
            trace: &self.trace,
            records: &mut self.records,
            capacity: self.status.tile_capacity,
            overflowed: &mut self.status.overflowed,
            attribution,
        })
    }
    pub(super) fn gather_start(&self) -> Option<f64> {
        (self.status.state == CaptureState::Recording).then(|| self.trace.elapsed())
    }
    pub(super) fn gather(
        &mut self,
        operation: OperationId,
        slot: usize,
        rows: usize,
        columns: usize,
        begin: f64,
    ) -> Result<()> {
        let record = CpuGather {
            operation,
            source_matrix_slot: slot,
            rows,
            columns,
            active_source_bytes: multiply(multiply(rows, columns)?, 2)?,
            host_begin_seconds: begin,
            host_end_seconds: self.trace.elapsed(),
        };
        let index = self
            .gathers
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| {
                self.status.overflowed = true;
                Error::Options("BF16 profile CPU gather capacity exceeded")
            })?;
        self.gathers[index] = Some(record);
        Ok(())
    }
    pub(super) fn export(&self) -> Result<FrameProfile> {
        if self.status.state != CaptureState::Complete || self.status.overflowed {
            return Err(Error::Options("BF16 committed profile unavailable"));
        }
        let clocks = autoreleasepool(|_| self.trace.resolve())?;
        if clocks.events.len() != self.records.len() {
            return Err(Error::Options("BF16 profile counter coverage differs"));
        }
        let gathers = [self.gathers[0].unwrap(), self.gathers[1].unwrap()];
        let mut tiles = reserve::<AttributedTile>(self.records.len())?;
        let mut host = gathers
            .iter()
            .map(|r| r.host_end_seconds - r.host_begin_seconds)
            .sum::<f64>();
        for (index, record) in self.records.iter().enumerate() {
            if record.counter_slot != index {
                return Err(Error::Options("BF16 profile counter slot differs"));
            }
            let event = *clocks
                .events
                .get(record.counter_slot)
                .ok_or(Error::Options("BF16 profile counter slot absent"))?;
            let durations = [
                record.host_staging_seconds,
                record.host_encoding_seconds,
                record.host_commit_seconds,
                record.host_wait_seconds,
                record.host_validation_and_copy_seconds,
            ];
            if durations.iter().any(|s| !s.is_finite() || *s < 0.) {
                return Err(Error::Options("BF16 profile host duration invalid"));
            }
            host += durations.iter().sum::<f64>();
            tiles.push(AttributedTile {
                record: *record,
                event,
            });
        }
        let bracket = clocks.host_clock_sample_bracket_seconds + self.end_bracket;
        let remaining = self.wall - host - bracket;
        if !remaining.is_finite() || remaining < -1e-6 {
            return Err(Error::Options("BF16 profile exclusive host scopes overlap"));
        }
        Ok(FrameProfile {
            schema_version: 1,
            request_id: self.options.request_id,
            evaluation_ordinal: self.options.evaluation_ordinal,
            evaluation: self.evaluation.unwrap(),
            status: self.status(),
            clocks,
            tiles,
            cpu_gathers: gathers,
            frame_wall_seconds: self.wall,
            exclusive_host_seconds: host,
            clock_bracket_seconds: bracket,
            unattributed_host_seconds: remaining.max(0.),
            diagnostic_allowance_bytes: ALLOWANCE_BYTES,
        })
    }
}

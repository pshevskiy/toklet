//! Audited storage for the fixed shared Metal LFM3 layout.
use super::*;
use crate::metal::lfm3_shared_weights::{Bank, BankBuilder, BankRef};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

const BANKS: usize = 3 * LAYERS;
const BANK_WORDS: usize = HIDDEN * INTERMEDIATE;
const BANK_BYTES: usize = 2 * BANK_WORDS;
const CONV_WORDS: usize = HIDDEN * 3 * HIDDEN;
const CONV_BANKS: usize = BANKS + CONV_LAYERS.len();
const BASE_MAX_BANKS: usize = CONV_BANKS + LAYERS;
const ATTENTION_LAYERS: usize = LAYERS - CONV_LAYERS.len();
pub(crate) const QKV_BANKS: usize = 3 * ATTENTION_LAYERS;
const QKV_WORDS: [usize; 3] = [
    HIDDEN * HIDDEN,
    HIDDEN * KV_HEADS * HEAD_DIM,
    HIDDEN * KV_HEADS * HEAD_DIM,
];
pub(crate) const QKV_BYTES: usize =
    2 * ATTENTION_LAYERS * (QKV_WORDS[0] + QKV_WORDS[1] + QKV_WORDS[2]);
const GRAPH_BANKS: usize = BASE_MAX_BANKS + QKV_BANKS;
const TIED_WORDS: usize = VOCAB * HIDDEN;
pub(crate) const TIED_BYTES: usize = 2 * TIED_WORDS;
const MAX_BANKS: usize = GRAPH_BANKS + 1;
const OUTPUT_WORDS: usize = HIDDEN * HIDDEN;
const CPU_WORDS: usize = PARAMETERS
    - BANKS * BANK_WORDS
    - CONV_LAYERS.len() * CONV_WORDS
    - LAYERS * OUTPUT_WORDS
    - QKV_BYTES / 2
    - TIED_WORDS;
const COLD_ALLOWANCE: usize = LOAD_ALLOWANCE + (1 << 20);

#[derive(Clone, Copy, Debug)]
pub(crate) struct Lfm3SharedPlan {
    pub required_bytes: usize,
    pub weights_bytes: usize,
    pub weight_padding_bytes: usize,
    pub cold_allowance_bytes: usize,
}

pub(crate) fn plan_lfm3_shared(
    device: &ProtocolObject<dyn MTLDevice>,
    capacity: usize,
    chunk: usize,
    extra_live_bytes: usize,
    memory_limit_bytes: usize,
) -> Result<Lfm3SharedPlan, Error> {
    if TIED_BYTES > device.maxBufferLength() {
        return Err(invalid("LFM3 tied bank exceeds device extent"));
    }
    if !device.hasUnifiedMemory() || BANK_BYTES > device.maxBufferLength() {
        return Err(invalid("LFM3 shared allocation device geometry"));
    }
    // SAFETY: sysconf with this selector has no pointer arguments.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .map_err(|_| invalid("unavailable shared LFM3 page size"))?;
    plan(capacity, chunk, page, extra_live_bytes, memory_limit_bytes)
}

fn plan(
    capacity: usize,
    chunk: usize,
    page: usize,
    extra_live_bytes: usize,
    memory_limit_bytes: usize,
) -> Result<Lfm3SharedPlan, Error> {
    if (capacity, chunk) != (2048, 128) || !page.is_power_of_two() {
        return Err(invalid(
            "LFM3 shared plan requires capacity2048/chunk128 and native page geometry",
        ));
    }
    let memory = Model::plan(capacity, chunk)?;
    let arithmetic = || invalid("LFM3 shared memory arithmetic overflow");
    let padding = |n: usize| {
        n.checked_add(page - 1)
            .map(|x| (x & !(page - 1)) - n)
            .ok_or_else(arithmetic)
    };
    let weight_padding_bytes = padding(BANK_BYTES)?
        .checked_mul(BANKS)
        .and_then(|x| {
            x.checked_add(padding(CPU_WORDS * 2).ok()?)?
                .checked_add(padding(TIED_BYTES).ok()?)
        })
        .and_then(|x| {
            x.checked_add(
                padding(CONV_WORDS * 2)
                    .ok()?
                    .checked_mul(CONV_LAYERS.len())?,
            )
        })
        .and_then(|x| x.checked_add(padding(OUTPUT_WORDS * 2).ok()?.checked_mul(LAYERS)?))
        .and_then(|x| {
            QKV_WORDS.into_iter().try_fold(x, |total, words| {
                total.checked_add(padding(words * 2).ok()?.checked_mul(ATTENTION_LAYERS)?)
            })
        })
        .ok_or_else(arithmetic)?;
    let required_bytes = [
        memory.total_bytes,
        weight_padding_bytes,
        COLD_ALLOWANCE,
        extra_live_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |n, item| n.checked_add(item).ok_or_else(arithmetic))?;
    let limit = memory_limit_bytes.min(INITIAL_MEMORY_LIMIT_BYTES);
    if required_bytes > limit {
        return Err(Error::Budget {
            required: required_bytes,
            limit,
        });
    }
    Ok(Lfm3SharedPlan {
        required_bytes,
        weights_bytes: memory.weights_bytes,
        weight_padding_bytes,
        cold_allowance_bytes: COLD_ALLOWANCE,
    })
}

#[derive(Clone, Copy)]
enum Place {
    Cpu(usize),
    Bank(usize),
}

struct Span {
    tensor: Tensor,
    place: Place,
}

pub(super) struct SharedWeights {
    spans: Vec<Span>,
    banks: Vec<Bank>,
}

impl SharedWeights {
    fn resolve(&self, tensor: Tensor) -> Option<&Span> {
        let index = self
            .spans
            .binary_search_by_key(&tensor.begin, |span| span.tensor.begin)
            .ok()?;
        let span = &self.spans[index];
        (span.tensor.end == tensor.end).then_some(span)
    }

    pub(super) fn get<'a>(&'a self, cpu: &'a [bf16], tensor: Tensor) -> &'a [bf16] {
        match self
            .resolve(tensor)
            .expect("exact validated LFM3 tensor handle")
            .place
        {
            Place::Cpu(start) => &cpu[start..start + (tensor.end - tensor.begin)],
            Place::Bank(slot) => self.banks[slot].words(),
        }
    }

    pub(super) fn bank(&self, tensor: Tensor) -> Option<BankRef<'_>> {
        match self.resolve(tensor)?.place {
            Place::Cpu(_) => None,
            Place::Bank(slot) => Some(self.banks[slot].view()),
        }
    }
}

fn spans(source: &Source) -> Result<Vec<Span>, Error> {
    let mut selected = reserve(MAX_BANKS)?;
    for layer in &source.layers {
        selected.extend([
            (layer.gate, BANK_WORDS),
            (layer.up, BANK_WORDS),
            (layer.down, BANK_WORDS),
        ]);
    }
    for index in CONV_LAYERS {
        let Operator::Convolution { bcx, .. } = source.layers[index].operator else {
            return Err(invalid("LFM3 selected ConvIn layer identity"));
        };
        selected.push((bcx, CONV_WORDS));
    }
    for layer in &source.layers {
        let (Operator::Convolution { output, .. } | Operator::Attention { output, .. }) =
            layer.operator;
        selected.push((output, OUTPUT_WORDS));
    }
    for layer in &source.layers {
        if let Operator::Attention { q, k, v, .. } = layer.operator {
            selected.extend([q, k, v].into_iter().zip(QKV_WORDS));
        }
    }
    if selected.len() != GRAPH_BANKS {
        return Err(invalid("incomplete LFM3 typed QKV bank roles"));
    }
    selected.push((source.embedding, TIED_WORDS));
    let keys = source.metadata.offset_keys();
    let mut result = reserve(keys.len())?;
    let mut position = 0;
    let mut cpu = 0;
    let mut seen = [false; MAX_BANKS];
    for name in keys {
        let info = source
            .metadata
            .info(&name)
            .ok_or_else(|| invalid("unknown shared LFM3 role"))?;
        let (begin, end) = info.data_offsets;
        if begin % 2 != 0
            || end % 2 != 0
            || begin / 2 != position
            || begin >= end
            || info.dtype != Dtype::BF16
        {
            return Err(invalid("invalid contiguous shared LFM3 BF16 span"));
        }
        let tensor = Tensor {
            begin: begin / 2,
            end: end / 2,
        };
        let words = tensor.end - tensor.begin;
        let place = if let Some(slot) = selected
            .iter()
            .position(|(role, _)| role.begin == tensor.begin)
        {
            if seen[slot] || words != selected[slot].1 || selected[slot].0.end != tensor.end {
                return Err(invalid("invalid selected LFM3 projection bank extent"));
            }
            seen[slot] = true;
            Place::Bank(slot)
        } else {
            let start = cpu;
            cpu += words;
            Place::Cpu(start)
        };
        result.push(Span { tensor, place });
        position = tensor.end;
    }
    if position != PARAMETERS
        || cpu != CPU_WORDS
        || !seen[..selected.len()].iter().all(|present| *present)
    {
        return Err(invalid("incomplete shared LFM3 typed layout"));
    }
    Ok(result)
}

pub(crate) fn load_lfm3_shared(
    directory: &Path,
    device: &ProtocolObject<dyn MTLDevice>,
    capacity: usize,
    chunk: usize,
    extra_live_bytes: usize,
    memory_limit_bytes: usize,
) -> Result<LoadedLfm3, Error> {
    plan_lfm3_shared(
        device,
        capacity,
        chunk,
        extra_live_bytes,
        memory_limit_bytes,
    )?;
    let memory = Model::plan(capacity, chunk)?;
    let mut source = read_source(directory)?;
    let started = Instant::now();
    let spans = spans(&source)?;
    let compact = CPU_WORDS;
    let layout_ns = source.layout_ns + started.elapsed().as_nanos();
    let started = Instant::now();
    let mut pending = reserve(MAX_BANKS)?;
    for slot in 0..MAX_BANKS {
        let words = if slot == GRAPH_BANKS {
            TIED_WORDS
        } else if slot < BANKS {
            BANK_WORDS
        } else if slot < CONV_BANKS {
            CONV_WORDS
        } else if slot < BASE_MAX_BANKS {
            OUTPUT_WORDS
        } else {
            QKV_WORDS[(slot - BASE_MAX_BANKS) % 3]
        };
        let bank = BankBuilder::new(device, words)?;
        if pending.iter().any(|other| !bank.disjoint(other)) {
            return Err(invalid("overlapping LFM3 native banks"));
        }
        pending.push(bank);
    }
    let mut data = reserve(compact)?;
    let mut i = 0;
    let mut position = 0;
    source.audit.visit(1, |mut bytes| {
        if !bytes.len().is_multiple_of(2) {
            return Err(invalid("unaligned shared LFM3 BF16 delivery"));
        }
        while !bytes.is_empty() {
            let span = spans
                .get(i)
                .ok_or_else(|| invalid("excess shared LFM3 delivery"))?;
            let start = position - span.tensor.begin;
            let take = (bytes.len() / 2).min(span.tensor.end - position);
            let (part, rest) = bytes.split_at(take * 2);
            match span.place {
                Place::Cpu(begin) => {
                    if begin + start != data.len() {
                        return Err(invalid("shared LFM3 CPU arena delivery cursor"));
                    }
                    for pair in part.chunks_exact(2) {
                        let word = bf16::from_le_bytes([pair[0], pair[1]]);
                        if !word.is_finite() {
                            return Err(invalid("nonfinite compact LFM3 CPU weight"));
                        }
                        data.push(word);
                    }
                }
                Place::Bank(slot) => pending[slot].write(start, part)?,
            }
            bytes = rest;
            position += take;
            if position == span.tensor.end {
                i += 1;
            }
        }
        Ok(())
    })?;
    let delivery = source.audit.finish()?;
    if delivery.file_sha256() != SHA256
        || delivery.bytes() != BYTES
        || data.len() != compact
        || i != spans.len()
        || position != PARAMETERS
    {
        return Err(invalid("incomplete authenticated shared LFM3 delivery"));
    }
    let mut banks: Vec<Bank> = reserve(MAX_BANKS)?;
    for builder in pending {
        let bank = builder.seal()?;
        banks.push(bank);
    }
    let delivery_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let model = Model::from_verified(
        Weights {
            data,
            shared: Some(SharedWeights { spans, banks }),
            embedding: source.embedding,
            final_norm: source.final_norm,
            layers: source.layers,
        },
        capacity,
        chunk,
        memory,
    )?;
    let rotary_ns = started.elapsed().as_nanos();
    Ok(LoadedLfm3 {
        model,
        report: Lfm3LoadReport {
            source_sha256: delivery.file_sha256(),
            source_bytes: delivery.bytes(),
            temporary_allowance_bytes: COLD_ALLOWANCE,
            authenticated_ns: source.authenticated_ns,
            layout_ns,
            delivery_ns,
            rotary_ns,
        },
    })
}

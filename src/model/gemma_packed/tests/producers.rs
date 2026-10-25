//! Small structural graph only: 35 layers / 15 producers, not E2B numerical parity.
use super::*;
use crate::model::gemma_reference::tests::Measurement;
use crate::model::window_cache::PreparedKv;
use smallvec::SmallVec;

const OWNERS: usize = 15;
const ALL_LAYERS: usize = 35;

struct Table {
    layout: crate::mlx::affine::Layout,
    codes: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}
impl Table {
    fn new(rows: usize, columns: usize) -> Self {
        let layout =
            crate::mlx::affine::Layout::new(rows, columns, 4, 64, MetadataDtype::BF16).unwrap();
        Self {
            layout,
            codes: (0..layout.weight_bytes())
                .map(|i| ((i % 7) * 17) as u8)
                .collect(),
            scales: [0x00, 0x3b].repeat(layout.metadata_bytes() / 2),
            biases: [0x00, 0xbc].repeat(layout.metadata_bytes() / 2),
        }
    }
    fn view(&self) -> Matrix<'_> {
        Matrix::from_le_bytes(self.layout, &self.codes, &self.scales, &self.biases).unwrap()
    }
}
struct Graph {
    data: fixture::Fixture,
    lookup: Table,
    projection: Table,
}
impl Graph {
    fn new() -> Self {
        Self {
            data: fixture::Fixture::new(),
            lookup: Table::new(VOCAB_SIZE, ALL_LAYERS * PLE),
            projection: Table::new(ALL_LAYERS * PLE, HIDDEN),
        }
    }
    fn layout() -> Layout {
        let mut l = Layout::witness();
        l.producers = OWNERS;
        l.layers = (0..ALL_LAYERS)
            .map(|i| {
                let full = i % 5 == 4;
                LayerLayout {
                    kind: if full { KvKind::Full } else { KvKind::Sliding },
                    head_dim: if full { 128 } else { 64 },
                    rotated_dims: if full { 32 } else { 64 },
                    intermediate: if i < OWNERS { 64 } else { 128 },
                    theta: if full { 1e6 } else { 1e4 },
                }
            })
            .collect();
        l
    }
    fn weights(&self) -> Weights<'_> {
        let mut weights = self.data.weights();
        weights.per_layer_embedding = self.lookup.view();
        weights.per_layer_projection = self.projection.view();
        weights.layers = (0..ALL_LAYERS)
            .map(|i| {
                let source = usize::from(i % 5 == 4) + if i < OWNERS { 0 } else { 2 };
                self.data.weights().layers.remove(source)
            })
            .collect();
        weights
    }
    fn model(&self) -> Model<'_> {
        Model::with_layout(self.weights(), Self::layout()).unwrap()
    }
}

#[test]
fn fifteen_producers_route_commit_and_chunk_equivalence() {
    let graph = Graph::new();
    let model = graph.model();
    assert_eq!(model.plan.producer_count(), OWNERS);
    for i in 0..ALL_LAYERS {
        let owner = if i < OWNERS {
            i
        } else if i % 5 == 4 {
            14
        } else {
            13
        };
        assert_eq!(model.plan.producer_for(i).unwrap(), owner);
    }
    let ids = &graph.data.ids;
    let mut control = model.new_session(usize::MAX).unwrap();
    let all = control.forward(ids).unwrap().to_vec();
    for schedule in [vec![1; 32], vec![3, 5, 7, 17]] {
        let mut session = model.new_session(usize::MAX).unwrap();
        let addresses = pointers(&session);
        let planned = model.planned_session_bytes().unwrap();
        assert_eq!(session.allocated_bytes().unwrap(), planned);
        let mut offset = 0;
        for n in schedule {
            close(
                session.forward(&ids[offset..offset + n]).unwrap(),
                &all[offset * VOCAB_SIZE..(offset + n) * VOCAB_SIZE]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>(),
            );
            offset += n;
            assert!(session.caches.iter().all(|c| c.offset() == offset));
            assert_eq!(pointers(&session), addresses);
        }
        // The last shared consumers cannot substitute another producer's state.
        for owner in 0..OWNERS {
            let a = session.caches[owner].committed();
            let b = control.caches[owner].committed();
            close(a.0, &b.0.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
            close(a.1, &b.1.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        }
        let full = snapshot(&session);
        assert!(session.forward(&[1]).is_err());
        assert_eq!(snapshot(&session), full);
        session.reset().unwrap();
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        assert_eq!(pointers(&session), addresses);
        assert_eq!(session.forward(ids).unwrap(), all);
    }
}

#[test]
fn all_producer_consumer_and_head_failures_are_atomic_and_allocate_nothing() {
    let graph = Graph::new();
    let model = graph.model();
    let ids = &graph.data.ids;
    let mut session = model.new_session(usize::MAX).unwrap();
    session.forward(&ids[..3]).unwrap();
    let mut control = model.new_session(usize::MAX).unwrap();
    control.forward(&ids[..3]).unwrap();
    control.forward(&ids[3..8]).unwrap();
    let after = snapshot(&control);
    for point in 0..=ALL_LAYERS + 2 {
        let mut session = model.new_session(usize::MAX).unwrap();
        session.forward(&ids[..3]).unwrap();
        let before = snapshot(&session);
        let addresses = pointers(&session);
        session.fault = Some(point);
        let measured = Measurement::start();
        let rejected = session.forward(&ids[3..8]).is_err();
        let allocations = measured.finish();
        assert!(rejected);
        assert_eq!(allocations, 0);
        assert_eq!(snapshot(&session), before);
        assert_eq!(pointers(&session), addresses);
        session.fault = None;
        let measured = Measurement::start();
        session.forward(&ids[3..8]).unwrap();
        let allocations = measured.finish();
        assert_eq!(allocations, 0);
        assert_eq!(snapshot(&session), after);
    }
    let measured = Measurement::start();
    session.reset().unwrap();
    for id in ids {
        session.forward(std::slice::from_ref(id)).unwrap();
    }
    let allocations = measured.finish();
    assert_eq!(allocations, 0);
    println!(
        "Gemma structural35/15: 38 failure/retry boundaries and32 decode steps allocate nothing; model={} session={} proposal_frame={} bytes",
        model.model_bytes(),
        session.allocated_bytes().unwrap(),
        std::mem::size_of::<SmallVec<[PreparedKv<'_>; MAX_PRODUCERS]>>()
    );
}

#[test]
fn producer_bound_and_actual_staging_capacity_are_admitted() {
    let graph = Graph::new();
    for invalid in [0, MAX_PRODUCERS + 1, ALL_LAYERS + 1] {
        let mut l = Graph::layout();
        l.producers = invalid;
        let weights = graph.weights();
        let measured = Measurement::start();
        let rejected = Model::with_layout(weights, l).is_err();
        let allocations = measured.finish();
        assert!(rejected);
        assert_eq!(allocations, 0);
    }
    // Exercise producers>layers independently of the separate inline bound.
    let mut short = Graph::layout();
    short.layers.truncate(4);
    short.producers = 5;
    let mut weights = graph.weights();
    weights.layers.truncate(4);
    let measured = Measurement::start();
    let rejected = Model::with_layout(weights, short).is_err();
    let allocations = measured.finish();
    assert!(rejected);
    assert_eq!(allocations, 0);
    let model = graph.model();
    let budget = model.model_bytes() + model.planned_session_bytes().unwrap();
    let measured = Measurement::start();
    let rejected = model.new_session(budget - 1).is_err();
    let allocations = measured.finish();
    assert!(rejected);
    assert_eq!(allocations, 0);
    let mut session = model.new_session(budget).unwrap();
    assert_eq!(
        session.allocated_bytes().unwrap(),
        model.planned_session_bytes().unwrap()
    );
    let before = session.allocated_bytes().unwrap();
    session.staging.reserve_exact(1);
    let excess = (session.staging.capacity() - OWNERS) * std::mem::size_of::<Staging>();
    assert!(excess > 0);
    assert_eq!(session.allocated_bytes().unwrap(), before + excess);
    session.staging.pop();
    let before = snapshot(&session);
    assert!(session.forward(&[1]).is_err());
    assert_eq!(snapshot(&session), before);
}

#[test]
fn physical_512_slot_rings_keep_32_position_admission_atomic() {
    let graph = Graph::new();
    let mut layout = Graph::layout();
    layout.window = 512;
    let model = Model::with_layout(graph.weights(), layout).unwrap();
    let budget = model.model_bytes() + model.planned_session_bytes().unwrap();
    let mut session = model.new_session(budget).unwrap();
    let addresses = pointers(&session);
    assert_eq!(
        session.allocated_bytes().unwrap(),
        model.planned_session_bytes().unwrap()
    );
    for (i, cache) in session.caches.iter().enumerate() {
        let (k, v) = cache.committed();
        let capacity = if i % 5 == 4 { 32 } else { 512 };
        let dims = if i % 5 == 4 { 128 } else { 64 };
        assert_eq!(k.len(), capacity * dims);
        assert_eq!(v.len(), k.len());
    }
    session.forward(&graph.data.ids[..3]).unwrap();
    let before = snapshot(&session);
    session.fault = Some(ALL_LAYERS + 2);
    let measured = Measurement::start();
    let rejected = session.forward(&graph.data.ids[3..]).is_err();
    assert_eq!(measured.finish(), 0);
    assert!(rejected);
    assert_eq!(snapshot(&session), before);
    session.fault = None;
    session.forward(&graph.data.ids[3..]).unwrap();
    let before = snapshot(&session);
    let measured = Measurement::start();
    let rejected = session.forward(&[1]).is_err();
    assert_eq!(measured.finish(), 0);
    assert!(rejected);
    assert_eq!(session.tokens().len(), 32);
    assert_eq!(snapshot(&session), before);
    assert_eq!(pointers(&session), addresses);
    assert!(session.caches.iter().all(|c| c.offset() == 32));
    session.reset().unwrap();
    let empty = snapshot(&session);
    assert!(session.forward(&[1; 33]).is_err());
    assert_eq!(snapshot(&session), empty);
    assert_eq!(pointers(&session), addresses);
}

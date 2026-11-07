//! Portable witness for coarse diagnostic hooks around authoritative commit.
use super::*;
use crate::mlx::affine::Matrix;
use crate::model::gemma_packed::operations::{BackendMemory, Operations, Scalar, sealed};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Begin(usize, usize, bool),
    Finish(Option<usize>),
    Replace,
}
struct Witness {
    events: Rc<RefCell<Vec<Event>>>,
    fail_head: bool,
}
impl sealed::Operations for Witness {}
impl Operations<Matrix<'_>> for Witness {
    type Error = Error;
    fn project(
        &mut self,
        input: &[f32],
        weight: &Matrix<'_>,
        output: &mut [f32],
        tokens: usize,
    ) -> Result<(), Error> {
        Scalar.project(input, weight, output, tokens)?;
        if self.fail_head && weight.layout().rows() == 17 {
            return Err(Error::InvalidConfig("injected late head failure"));
        }
        Ok(())
    }
    fn gather(
        &mut self,
        weight: &Matrix<'_>,
        ids: &[u32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        Scalar.gather(weight, ids, output)
    }
    fn ready(&self) -> Result<(), Error> {
        Ok(())
    }
    fn memory(&self) -> Result<BackendMemory, Error> {
        Ok(BackendMemory::default())
    }
    fn begin_generation_step(&mut self, position: usize, rows: usize, last: bool) {
        self.events
            .borrow_mut()
            .push(Event::Begin(position, rows, last));
    }
    fn finish_generation_step(&mut self, committed: Option<usize>) {
        self.events.borrow_mut().push(Event::Finish(committed));
    }
    fn generation_history_replaced(&mut self) {
        self.events.borrow_mut().push(Event::Replace);
    }
}
#[test]
fn lifecycle_hooks_follow_commits_and_ignore_invalid_or_fully_cached_work() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut observed = GenerationSession::with_backend(
        &model,
        Witness {
            events: events.clone(),
            fail_head: false,
        },
        options(),
    )
    .unwrap();
    let mut reference = model.new_generation_session(options()).unwrap();
    reference.append(&data.ids[..5]).unwrap();
    observed.append(&data.ids[..5]).unwrap();
    for count in [3, 2] {
        assert_eq!(observed.evaluate_next_chunk().unwrap(), count);
        assert_eq!(reference.evaluate_next_chunk().unwrap(), count);
        assert_eq!(observed.history(), reference.history());
        assert_eq!(observed.evaluated_len(), reference.evaluated_len());
        assert_eq!(
            observed
                .logits()
                .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
            reference
                .logits()
                .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
        );
    }
    let expected = vec![
        Event::Begin(0, 3, false),
        Event::Finish(Some(3)),
        Event::Begin(3, 2, false),
        Event::Finish(Some(5)),
    ];
    assert_eq!(*events.borrow(), expected);
    assert_eq!(observed.evaluate_next_chunk().unwrap(), 0);
    assert!(observed.append(&[17]).is_err());
    assert!(observed.set_tokens(&[17]).is_err());
    assert_eq!(*events.borrow(), expected);
    observed.set_tokens(&data.ids[..5]).unwrap();
    assert_eq!(events.borrow().last(), Some(&Event::Replace));
    observed.reset();
    assert_eq!(
        events
            .borrow()
            .iter()
            .filter(|e| **e == Event::Replace)
            .count(),
        2
    );
}
#[test]
fn lifecycle_hooks_reject_a_late_failure_without_committing_a_prefix() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut observed = GenerationSession::with_backend(
        &model,
        Witness {
            events: events.clone(),
            fail_head: true,
        },
        options(),
    )
    .unwrap();
    observed.append(&data.ids[..3]).unwrap();
    assert!(observed.evaluate_next_chunk().is_err());
    assert_eq!(
        *events.borrow(),
        [Event::Begin(0, 3, false), Event::Finish(None)]
    );
    assert_eq!(observed.history(), &data.ids[..3]);
    assert_eq!(observed.evaluated_len(), 0);
    assert!(observed.logits().is_none());
    assert!(
        observed
            .inner
            .caches
            .iter()
            .all(|cache| cache.offset() == 0)
    );
}

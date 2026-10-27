use super::*;
#[test]
fn instance_exhaustion_never_wraps_or_reuses_zero() {
    let sequence = Instances(AtomicU64::new(u64::MAX - 1));
    assert_eq!(sequence.next().unwrap(), u64::MAX - 1);
    assert!(sequence.next().is_err());
    assert!(sequence.next().is_err());
    assert_eq!(sequence.0.load(Ordering::Relaxed), u64::MAX);
    assert!(Instances(AtomicU64::new(0)).next().is_err());
}

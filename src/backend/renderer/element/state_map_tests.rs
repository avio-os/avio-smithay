use super::*;

fn state(area: usize) -> RenderElementState {
    RenderElementState {
        visible_area: area,
        presentation_state: super::super::RenderElementPresentationState::ZeroCopy,
        needs_capture: false,
    }
}

#[test]
fn held_receipts_keep_independent_maps_until_exact_drop() {
    let bank = StateMapBank::new(4, 2);
    let id = Id::new();
    let mut first = bank.acquire(1).unwrap();
    first.insert(id.clone(), state(12));
    let mut latest = bank.acquire(1).unwrap();
    latest.insert(id.clone(), state(24));
    assert!(bank.acquire(1).is_err());
    assert_eq!(first[&id].visible_area, 12);
    assert_eq!(latest[&id].visible_area, 24);
    drop(latest);
    let returned = bank.acquire(4).unwrap();
    assert!(returned.is_empty());
    assert_eq!(first[&id].visible_area, 12);
}

#[test]
fn admission_and_reserved_copy_never_steal_a_live_slot() {
    let bank = StateMapBank::new(2, 2);
    assert!(bank.acquire(3).is_err());
    let id = Id::new();
    let mut first = bank.acquire(2).unwrap();
    first.insert(id.clone(), state(1));
    let copy = first.try_clone_reserved().unwrap();
    assert!(first.try_clone_reserved().is_err());
    assert_eq!(copy[&id].visible_area, 1);
    drop(copy);
    assert!(bank.acquire(2).is_ok());
}

#[test]
fn dropped_iterator_and_cross_thread_receipt_return_the_exact_storage() {
    let bank = StateMapBank::new(2, 1);
    let mut receipt = bank.acquire(2).unwrap();
    receipt.insert(Id::new(), state(1));
    receipt.insert(Id::new(), state(2));
    let mut iterator = receipt.into_iter();
    assert!(iterator.next().is_some());
    assert!(bank.acquire(1).is_err());
    std::thread::spawn(move || drop(iterator)).join().unwrap();
    assert!(bank.acquire(2).unwrap().is_empty());
}

#[test]
fn exact_return_wake_observes_bank_receipt_arc_already_released() {
    use std::sync::{atomic::AtomicUsize, Mutex};
    let observed = Arc::new(AtomicUsize::new(usize::MAX));
    let weak = Arc::new(Mutex::new(None::<std::sync::Weak<StateMapBankInner>>));
    let callback_weak = weak.clone();
    let callback_observed = observed.clone();
    let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        callback_observed.store(
            callback_weak.lock().unwrap().as_ref().unwrap().strong_count(),
            Ordering::Relaxed,
        );
    });
    let bank = StateMapBank::new_with_wakeup(
        1,
        1,
        Some(StateReceiptReturnWakeup::with_retirement(
            Some(wake.clone()),
            Arc::new(AtomicBool::new(false)),
            None,
        )),
    );
    *weak.lock().unwrap() = Some(Arc::downgrade(&bank.0));
    let receipt = bank.acquire(1).unwrap();
    drop(receipt);
    assert_eq!(
        observed.load(Ordering::Relaxed),
        1,
        "notification never precedes final receipt bank-reference return"
    );
    assert!(bank.is_reclaimable());
}

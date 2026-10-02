use super::*;

#[test]
fn held_vector_receipts_never_share_storage_or_disappear_on_pressure() {
    let bank = VecStorageBank::new("test selected indices", 12, 2);
    let mut old = bank.acquire(12).unwrap();
    old.extend(0..12).unwrap();
    let mut latest = bank.acquire(12).unwrap();
    latest.extend(12..24).unwrap();
    assert!(bank.acquire(1).is_err());
    assert!(latest.push(24).is_err());
    assert_eq!(&*old, &(0..12).collect::<Vec<_>>());
    drop(latest);
    let mut replacement = bank.acquire(12).unwrap();
    assert!(replacement.is_empty());
    replacement.push(999).unwrap();
    assert_eq!(old[0], 0);
}

#[test]
fn cross_thread_drop_returns_only_the_exact_vector_slot() {
    let bank = VecStorageBank::new("test selected indices", 1, 2);
    let mut old = bank.acquire(1).unwrap();
    old.push(1).unwrap();
    let mut latest = bank.acquire(1).unwrap();
    latest.push(2).unwrap();
    std::thread::spawn(move || drop(latest)).join().unwrap();
    let replacement = bank.acquire(1).unwrap();
    assert!(replacement.is_empty());
    assert_eq!(old[0], 1);
    assert!(bank.acquire(1).is_err());
}

#[test]
fn reservation_copy_requires_an_independent_free_slot() {
    let bank = VecStorageBank::new("test selected indices", 2, 2);
    let mut old = bank.acquire(2).unwrap();
    old.extend([1, 2]).unwrap();
    let copy = old.try_clone_reserved().unwrap();
    assert!(old.try_clone_reserved().is_err());
    assert_eq!(&*copy, &[1, 2]);
    drop(old);
    let mut replacement = bank.acquire(2).unwrap();
    replacement.push(3).unwrap();
    assert_eq!(&*copy, &[1, 2]);
}

#[test]
fn warmed_120_numeric_receipt_cycles_do_not_allocate_or_free() {
    let bank = VecStorageBank::new("test selected indices", 12, 3);
    let mut retained = bank.acquire(12).unwrap();
    retained.extend(0..12).unwrap();
    let (_, counts) = crate::backend::renderer::storage_heap_probe::measure(|| {
        for _ in 0..120 {
            let mut work = bank.acquire(12).unwrap();
            work.extend(0..12).unwrap();
            let copy = work.try_clone_reserved().unwrap();
            assert!(bank.acquire(1).is_err());
            drop(copy);
            work.clear();
            drop(work);
            assert_eq!(retained.len(), 12);
        }
    });
    assert_eq!(counts, [0; 4]);
}

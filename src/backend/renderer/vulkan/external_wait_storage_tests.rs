//! Real production semaphore custody through ash dispatch ABI test functions.
//! No Vulkan driver or GPU is used; native acceptance remains a device gate.
use super::super::{
    device::{InFlightSubmission, RetiredCommands},
    device_handle::retirement_tests::{
        device, device_with_wait_result, drain_until_parent, external_semaphore_fd, next, set_import_result,
        Operation,
    },
    storage_heap_probe::measure,
    submission_storage::VulkanCommandStorageLimits,
    sync::import_sync_file_into_semaphore,
};
use super::*;
use ash::vk::Handle;
use std::{
    os::fd::AsRawFd,
    sync::{atomic::AtomicI32, mpsc},
    thread,
};

#[derive(Debug)]
struct DropProof(mpsc::SyncSender<thread::ThreadId>);
impl Drop for DropProof {
    fn drop(&mut self) {
        let _ = self.0.send(thread::current().id());
    }
}
impl Fence for DropProof {
    fn is_signaled(&self) -> bool {
        false
    }
    fn wait(&self) -> Result<(), Interrupted> {
        Err(Interrupted)
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<std::os::fd::OwnedFd> {
        None
    }
}
fn proof() -> (SyncPoint, mpsc::Receiver<thread::ThreadId>) {
    let (sent, received) = mpsc::sync_channel(1);
    (SyncPoint::from(DropProof(sent)), received)
}
fn take_create(events: &mpsc::Receiver<(Operation, thread::ThreadId)>) -> u64 {
    match next(events).0 {
        Operation::SemaphoreCreated(id) => id,
        other => panic!("{other:?}"),
    }
}

#[test]
fn native_wait_loan_preserves_original_proof_and_resets_once_on_existing_actor() {
    let warm = thread::current().id();
    let (device, events) = device();
    let mut bank = ExternalWaitBank::cold(device.clone(), 1).unwrap();
    let handle = take_create(&events);
    let (source, dropped) = proof();
    let same = source.clone();
    let loan = bank.take(&source).unwrap();
    assert_eq!(loan.handle.as_raw(), handle);
    assert!(bank.contains_owner(&same));
    assert!(!bank.all_returned_edge().is_reached());
    drop(source);
    drop(same);
    assert!(dropped.try_recv().is_err(), "exact imported proof stays retained");
    bank.release(loan);
    bank.all_returned_edge().wait().unwrap();
    let actor = dropped.recv_timeout(std::time::Duration::from_secs(3)).unwrap();
    assert_ne!(actor, warm);
    assert_eq!(bank.available(), 1);
    assert!(
        events.try_recv().is_err(),
        "return does not create/destroy a native semaphore"
    );
    drop(bank);
    assert_eq!(next(&events), (Operation::Semaphore(handle), actor));
    drop(device);
    drain_until_parent(&events);
}

#[test]
fn bank_exhaustion_never_steals_another_proof_or_fabricates_native_ready_fd() {
    let (device, events) = device();
    let mut bank = ExternalWaitBank::cold(device.clone(), 1).unwrap();
    take_create(&events);
    let (source, _) = proof();
    let (other, _) = proof();
    let first = bank.take(&source).unwrap();
    assert!(bank.admit_batch(1).unwrap_err().is_command_deferred());
    assert!(matches!(
        bank.admit_batch(2),
        Err(VulkanRendererError::CommandStorageLimitExceeded {
            requested: 2,
            limit: 1,
            ..
        })
    ));
    assert!(bank.contains_owner(&source));
    assert!(!bank.contains_owner(&other));
    assert!(bank.take(&other).is_none());
    let ready = bank.all_returned_edge();
    assert!(!ready.is_reached());
    assert!(!ready.is_exportable());
    assert!(ready.export().is_none());
    bank.release(first);
    ready.wait().unwrap();
    bank.admit_batch(1).unwrap();
    let second = bank.take(&other).unwrap();
    assert_eq!(first.handle, second.handle);
    bank.release(second);
    ready.wait().unwrap();
    drop((source, other, ready, bank, device));
    drain_until_parent(&events);
}

#[test]
fn actual_temporary_import_transfers_success_fd_and_closes_only_failed_duplicate() {
    let (device, events) = device();
    let mut bank = ExternalWaitBank::cold(device.clone(), 1).unwrap();
    let handle = take_create(&events);
    let fd = rustix::event::eventfd(0, rustix::event::EventfdFlags::CLOEXEC).unwrap();
    let source = SyncPoint::from(crate::backend::renderer::sync::SyncFileFence::new(fd));
    let loan = bank.take(&source).unwrap();
    let dispatch = external_semaphore_fd(&device);
    let success = source.export().unwrap();
    let success_fd = success.as_raw_fd();
    import_sync_file_into_semaphore(&device, &dispatch, loan.handle, success).unwrap();
    assert_eq!(next(&events).0, Operation::SemaphoreImported(handle));
    assert_eq!(unsafe { libc::fcntl(success_fd, libc::F_GETFD) }, -1);
    assert!(
        source.export().is_some(),
        "cached original remains live for retry"
    );
    set_import_result(&device, vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    let failed = source.export().unwrap();
    let failed_fd = failed.as_raw_fd();
    let error = import_sync_file_into_semaphore(&device, &dispatch, loan.handle, failed).unwrap_err();
    assert!(error.is_resource_allocation_failure());
    assert_eq!(unsafe { libc::fcntl(failed_fd, libc::F_GETFD) }, -1);
    assert!(source.export().is_some());
    assert!(
        events.try_recv().is_err(),
        "failed import does not destroy pooled handle"
    );
    bank.release(loan);
    bank.all_returned_edge().wait().unwrap();
    drop((source, bank, device));
    drain_until_parent(&events);
}

#[test]
fn unknown_native_completion_keeps_semaphore_bank_and_proof_until_real_success() {
    let wait = Arc::new(AtomicI32::new(vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw()));
    let (device, events) = device_with_wait_result(wait.clone());
    let mut bank = ExternalWaitBank::cold(device.clone(), 1).unwrap();
    let semaphore = take_create(&events);
    let (source, dropped) = proof();
    let loan = bank.take(&source).unwrap();
    drop(source);
    let mut submitted =
        InFlightSubmission::cold(device.clone(), None, VulkanCommandStorageLimits::default()).unwrap();
    submitted.wait_semaphores.push(loan);
    let mut commands = RetiredCommands::empty(vk::CommandPool::from_raw(987654));
    commands.external_wait_bank = Some(bank);
    commands.submissions.push_back(submitted);
    assert_eq!(
        commands.wait_complete(device.handle()),
        Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
    );
    assert!(dropped.try_recv().is_err());
    assert!(
        events.try_recv().is_err(),
        "unknown wait frees no semaphore or source proof"
    );
    wait.store(vk::Result::SUCCESS.as_raw(), Ordering::Release);
    commands.wait_complete(device.handle()).unwrap();
    commands.destroy(device.handle());
    let mut destroyed = false;
    for _ in 0..3 {
        if next(&events).0 == Operation::Semaphore(semaphore) {
            destroyed = true;
        }
    }
    assert!(destroyed);
    assert_ne!(
        dropped.recv_timeout(std::time::Duration::from_secs(3)).unwrap(),
        thread::current().id()
    );
    drop(device);
    drain_until_parent(&events);
}

#[test]
fn warm_native_wait_loans_dedup_and_actor_return_have_no_heap_operations() {
    let (device, events) = device();
    let mut bank = ExternalWaitBank::cold(device.clone(), 1).unwrap();
    take_create(&events);
    let (source, _) = proof();
    let ready = bank.all_returned_edge();
    // Initialize thread parking/condvar bookkeeping outside the probe.
    let first = bank.take(&source).unwrap();
    bank.release(first);
    ready.wait().unwrap();
    let (_, calls) = measure(|| {
        for _ in 0..4096 {
            bank.admit_batch(1).unwrap();
            let loan = bank.take(&source).unwrap();
            assert!(bank.contains_owner(&source));
            assert!(!ready.is_reached());
            bank.release(loan);
            ready.wait().unwrap();
        }
    });
    assert_eq!(calls, [0; 4]);
    assert!(events.try_recv().is_err());
    drop((source, ready, bank, device));
    drain_until_parent(&events);
}

#[test]
fn declared_batch_limit_refuses_before_extra_native_import() {
    let mut batch = ExternalWaitBatch::default();
    batch.begin(2);
    batch.check().unwrap();
    batch.imported();
    batch.check().unwrap();
    batch.imported();
    assert!(matches!(
        batch.check(),
        Err(VulkanRendererError::CommandStorageLimitExceeded {
            requested: 3,
            limit: 2,
            ..
        })
    ));
    batch.begin(1);
    batch.check().unwrap();
    batch.imported();
    assert!(batch.check().is_err());
}

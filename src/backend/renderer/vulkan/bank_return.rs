//! Cold aggregate for replacing a complete command bank, not one free slot.
use super::{device::InFlightSubmission, fence_return::FenceReaders, sync::VulkanFence};
use crate::backend::renderer::sync::{Fence, Interrupted, SyncPoint, SyncPointOwnerReturn};
use std::{
    fmt,
    os::fd::OwnedFd,
    sync::{Arc, MutexGuard},
};
struct Entry {
    native: Arc<VulkanFence>,
    readers: Arc<FenceReaders>,
}
struct BankReturn {
    entries: Vec<Entry>,
}
impl fmt::Debug for BankReturn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BankReturn")
            .field("slots", &self.entries.len())
            .finish()
    }
}
struct NativeReader<'a> {
    readers: &'a FenceReaders,
    _epoch: MutexGuard<'a, ()>,
}
impl<'a> NativeReader<'a> {
    fn new(readers: &'a FenceReaders) -> Self {
        let epoch = readers.wait_native_epoch();
        readers.acquire_reader();
        Self {
            readers,
            _epoch: epoch,
        }
    }
    fn try_new(readers: &'a FenceReaders) -> Option<Self> {
        let epoch = readers.try_native_epoch()?;
        readers.acquire_reader();
        Some(Self {
            readers,
            _epoch: epoch,
        })
    }
}
impl Drop for NativeReader<'_> {
    fn drop(&mut self) {
        self.readers.release_reader();
    }
}
impl Fence for BankReturn {
    fn is_signaled(&self) -> bool {
        self.entries.iter().all(|entry| {
            if !entry.readers.returned() {
                return false;
            }
            let Some(reader) = NativeReader::try_new(&entry.readers) else {
                return false;
            };
            let ready = (!entry.native.native_attempted() || entry.native.status().unwrap_or(false))
                && entry.readers.only_internal_reader();
            drop(reader);
            ready
        })
    }
    fn wait(&self) -> Result<(), Interrupted> {
        loop {
            for entry in &self.entries {
                // Pin this exact native epoch while its host wait executes.
                // The pin is returned before waiting for external readers,
                // so the aggregate cannot deadlock on its own custody.
                let reader = NativeReader::new(&entry.readers);
                let result = if entry.native.native_attempted() {
                    entry.native.wait_vk().map_err(|_| Interrupted)
                } else {
                    Ok(())
                };
                drop(reader);
                result?;
                entry.readers.wait_returned()?;
            }
            if self.is_signaled() {
                return Ok(());
            }
        }
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<OwnedFd> {
        None
    }
}
pub(super) fn cold_bank_return(slots: &[InFlightSubmission]) -> SyncPoint {
    SyncPoint::from_shared_fence(Arc::new(BankReturn {
        entries: slots
            .iter()
            .map(|slot| Entry {
                native: slot.native.clone(),
                readers: slot.readers.clone(),
            })
            .collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::super::{
        device_handle::retirement_tests::{device_with_wait_result, drain_until_parent},
        VulkanCommandStorageLimits,
    };
    use super::*;
    use ash::vk;
    use std::sync::atomic::{AtomicI32, Ordering};
    #[test]
    fn one_free_slot_does_not_make_whole_bank_replacement_ready() {
        let status = Arc::new(AtomicI32::new(vk::Result::SUCCESS.as_raw()));
        let (device, events) = device_with_wait_result(status);
        let slots = vec![
            InFlightSubmission::cold(device.clone(), None, VulkanCommandStorageLimits::default()).unwrap(),
            InFlightSubmission::cold(device.clone(), None, VulkanCommandStorageLimits::default()).unwrap(),
        ];
        let all = cold_bank_return(&slots);
        assert!(all.is_reached());
        let external = slots[0].fence.clone();
        assert!(slots[1].readers_returned());
        assert!(!all.is_reached());
        assert!(!all.is_exportable());
        drop(external);
        assert!(all.is_reached());
        drop(slots);
        drop(all);
        drop(device);
        drain_until_parent(&events);
    }
    #[test]
    fn bank_edge_outlives_its_slot_but_still_waits_for_the_exact_last_external_reader() {
        let status = Arc::new(AtomicI32::new(vk::Result::SUCCESS.as_raw()));
        let (device, events) = device_with_wait_result(status);
        let slots = vec![InFlightSubmission::cold(device.clone(), None, Default::default()).unwrap()];
        let all = cold_bank_return(&slots);
        let external = slots[0].fence.clone();
        drop(slots);
        assert!(!all.is_reached());
        drop(external);
        assert!(all.is_reached());
        all.wait().unwrap();
        drop(all);
        drop(device);
        drain_until_parent(&events);
    }
    #[test]
    fn native_observer_refuses_epoch_reset_without_a_driver_call() {
        let status = Arc::new(AtomicI32::new(vk::Result::SUCCESS.as_raw()));
        let (device, events) = device_with_wait_result(status);
        let slot = InFlightSubmission::cold(device.clone(), None, Default::default()).unwrap();
        let reader = NativeReader::new(&slot.readers);
        assert!(slot.prepare_for_submit().unwrap_err().is_command_deferred());
        assert!(!slot.reader_return.is_reached());
        assert!(events.try_recv().is_err(), "no reset under native observation");
        drop(reader);
        assert!(slot.reader_return.is_reached());
        slot.prepare_for_submit().unwrap();
        assert!(matches!(
            events.recv().unwrap().0,
            super::super::device_handle::retirement_tests::Operation::FenceReset(_)
        ));
        drop(slot);
        drop(device);
        drain_until_parent(&events);
    }
    #[test]
    fn actual_native_pending_or_unknown_status_never_reports_adoption_ready() {
        let status = Arc::new(AtomicI32::new(vk::Result::NOT_READY.as_raw()));
        let (device, events) = device_with_wait_result(status.clone());
        let slots =
            vec![
                InFlightSubmission::cold(device.clone(), None, VulkanCommandStorageLimits::default())
                    .unwrap(),
            ];
        slots[0].native.mark_native_attempt();
        let all = cold_bank_return(&slots);
        assert!(!all.is_reached());
        status.store(vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw(), Ordering::Release);
        assert!(!all.is_reached());
        assert!(all.wait().is_err());
        assert!(
            slots[0].readers_returned(),
            "internal native pin returned on failed wait"
        );
        status.store(vk::Result::SUCCESS.as_raw(), Ordering::Release);
        assert!(all.is_reached());
        all.wait().unwrap();
        drop(slots);
        drop(all);
        drop(device);
        drain_until_parent(&events);
    }
}

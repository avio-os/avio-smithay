//! Same-context binary semaphore loans for an admitted source-fence inventory.
//! Submitted loans return only after their exact native fence completes. The
//! original proof owner then resets on the existing native retirement actor.

use std::sync::{
    atomic::{AtomicBool, AtomicU8, Ordering},
    Arc, Condvar, Mutex,
};

use ash::vk;

use super::{
    device::DeviceHandle, device_handle::DeviceRetirement, retirement::RetirementNode, VulkanRendererError,
};
use crate::backend::renderer::sync::{Fence, Interrupted, SyncPoint};

const READY: u8 = 0;
const LOANED: u8 = 1;
const RESETTING: u8 = 2;

#[derive(Clone, Copy, Debug)]
pub(super) struct ImportedWaitSemaphore {
    pub(super) handle: vk::Semaphore,
    pub(super) pooled: bool,
    pub(super) index: usize,
}

struct WaitSlot {
    handle: vk::Semaphore,
    state: AtomicU8,
    owner: Mutex<Option<SyncPoint>>,
    node: Mutex<Option<Box<RetirementNode<DeviceRetirement>>>>,
}

struct WaitBacking {
    slots: Vec<WaitSlot>,
    closed: AtomicBool,
    wake_lock: Mutex<()>,
    wake: Condvar,
}

pub(super) struct RetiredExternalWaitSemaphores {
    handles: Vec<vk::Semaphore>,
    _backing: Option<Arc<WaitBacking>>,
}

impl RetiredExternalWaitSemaphores {
    pub(super) fn destroy(self, raw: &ash::Device) {
        for handle in &self.handles {
            // The command owner proves all submitted loans have completed.
            unsafe { raw.destroy_semaphore(*handle, None) };
        }
    }
}

/// One reusable queue node per actual native wait handle, allocated cold.
/// Its strong backing owner exists only while this node is on the actor.
/// Idle nodes therefore introduce no bank/backing ownership cycle.
pub(super) struct ExternalWaitOwnerReturn {
    backing: Option<Arc<WaitBacking>>,
    index: usize,
}

impl ExternalWaitOwnerReturn {
    pub(super) fn reset(mut node: Box<RetirementNode<DeviceRetirement>>) {
        let (backing, index) = match node.value_mut() {
            DeviceRetirement::ExternalWaitOwner(returned) => (
                returned.backing.take().expect("submitted owner-return node"),
                returned.index,
            ),
            _ => unreachable!("exact native wait-owner node"),
        };
        let slot = &backing.slots[index];
        assert_eq!(slot.state.load(Ordering::Acquire), RESETTING);
        // Last native proof/FD owner disposal belongs to this cold actor.
        let owner = slot.owner.lock().unwrap_or_else(|p| p.into_inner()).take();
        drop(owner);
        *slot.node.lock().unwrap_or_else(|p| p.into_inner()) = Some(node);
        let _wake = backing.wake_lock.lock().unwrap_or_else(|p| p.into_inner());
        slot.state.store(READY, Ordering::Release);
        backing.wake.notify_all();
    }
}

struct AllWaitOwnersReturned(Arc<WaitBacking>);
impl std::fmt::Debug for AllWaitOwnersReturned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllWaitOwnersReturned").finish_non_exhaustive()
    }
}
impl Fence for AllWaitOwnersReturned {
    fn is_signaled(&self) -> bool {
        self.0.closed.load(Ordering::Acquire)
            || self
                .0
                .slots
                .iter()
                .all(|slot| slot.state.load(Ordering::Acquire) == READY)
    }
    fn wait(&self) -> Result<(), Interrupted> {
        let mut wake = self.0.wake_lock.lock().unwrap_or_else(|p| p.into_inner());
        while !self.is_signaled() {
            wake = self.0.wake.wait(wake).unwrap_or_else(|p| p.into_inner());
        }
        Ok(())
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<std::os::fd::OwnedFd> {
        None
    }
}

pub(super) struct ExternalWaitBank {
    device: Arc<DeviceHandle>,
    backing: Arc<WaitBacking>,
    returned: SyncPoint,
    retirement: Option<Box<RetirementNode<DeviceRetirement>>>,
}

impl ExternalWaitBank {
    pub(super) fn cold(device: Arc<DeviceHandle>, count: usize) -> Result<Self, VulkanRendererError> {
        let mut retirement = scopeguard::guard(
            Some(RetirementNode::new(DeviceRetirement::ExternalWaitSemaphores(
                RetiredExternalWaitSemaphores {
                    handles: Vec::with_capacity(count),
                    _backing: None,
                },
            ))),
            |node| {
                if let Some(node) = node {
                    device.retire_resource(node);
                }
            },
        );
        let storage = match retirement.as_mut().unwrap().value_mut() {
            DeviceRetirement::ExternalWaitSemaphores(storage) => storage,
            _ => unreachable!(),
        };
        for _ in 0..count {
            storage.handles.push(device.observe_result(unsafe {
                device
                    .handle()
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })?);
        }
        let backing = Arc::new(WaitBacking {
            slots: storage
                .handles
                .iter()
                .enumerate()
                .map(|(index, handle)| WaitSlot {
                    handle: *handle,
                    state: AtomicU8::new(READY),
                    owner: Mutex::new(None),
                    node: Mutex::new(Some(RetirementNode::new(DeviceRetirement::ExternalWaitOwner(
                        ExternalWaitOwnerReturn { backing: None, index },
                    )))),
                })
                .collect(),
            closed: AtomicBool::new(false),
            wake_lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        storage._backing = Some(backing.clone());
        let returned = SyncPoint::from_shared_fence(Arc::new(AllWaitOwnersReturned(backing.clone())));
        let node = retirement.take();
        drop(retirement);
        Ok(Self {
            device,
            backing,
            returned,
            retirement: node,
        })
    }

    pub(super) fn capacity(&self) -> usize {
        self.backing.slots.len()
    }
    pub(super) fn available(&self) -> usize {
        // The renderer is the sole borrower; the actor only publishes READY.
        // Count the actual states rather than racing a separate ready counter
        // against its slot publication in either order.
        self.backing
            .slots
            .iter()
            .filter(|slot| slot.state.load(Ordering::Acquire) == READY)
            .count()
    }
    pub(super) fn all_returned_edge(&self) -> SyncPoint {
        self.returned.clone()
    }

    pub(super) fn admit_batch(&self, required: usize) -> Result<(), VulkanRendererError> {
        if required > self.capacity() {
            return Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "prepared external waits",
                requested: required,
                limit: self.capacity(),
            });
        }
        if self.available() < required {
            return Err(VulkanRendererError::CommandCapacityExhausted {
                slots: self.capacity(),
            });
        }
        Ok(())
    }

    /// Identity comparison is the original shared proof Arc, never its FD.
    /// ALL_COMMANDS imports make this exact queue dependency reusable by all
    /// later reads in the selected batch, including partial render segments.
    pub(super) fn contains_owner(&self, proof: &SyncPoint) -> bool {
        self.backing.slots.iter().any(|slot| {
            slot.state.load(Ordering::Acquire) == LOANED
                && slot
                    .owner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .as_ref()
                    .is_some_and(|owner| owner.same_native_owner(proof))
        })
    }

    pub(super) fn take(&mut self, owner: &SyncPoint) -> Option<ImportedWaitSemaphore> {
        let (index, slot) = self.backing.slots.iter().enumerate().find(|(_, slot)| {
            slot.state
                .compare_exchange(READY, LOANED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        })?;
        assert!(slot.node.lock().unwrap_or_else(|p| p.into_inner()).is_some());
        let mut proof = slot.owner.lock().unwrap_or_else(|p| p.into_inner());
        assert!(proof.is_none());
        *proof = Some(owner.clone());
        Some(ImportedWaitSemaphore {
            handle: slot.handle,
            pooled: true,
            index,
        })
    }

    /// Only an unsubmitted import owner or an exactly completed submission
    /// can return this loan. Native handle reuse waits for actor-side owner
    /// reset too, so final proof disposal cannot move onto a frame thread.
    pub(super) fn release(&mut self, semaphore: ImportedWaitSemaphore) {
        let slot = &self.backing.slots[semaphore.index];
        assert!(semaphore.pooled && semaphore.handle == slot.handle);
        assert_eq!(slot.state.swap(RESETTING, Ordering::AcqRel), LOANED);
        let mut node = slot
            .node
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .expect("unique wait return node");
        match node.value_mut() {
            DeviceRetirement::ExternalWaitOwner(returned) => returned.backing = Some(self.backing.clone()),
            _ => unreachable!(),
        }
        self.device.retire_resource(node);
    }
}

impl Drop for ExternalWaitBank {
    fn drop(&mut self) {
        // RetiredCommands retains this bank until native proof, including
        // unknown-completion teardown. This wake is a CPU lifecycle edge.
        let _wake = self.backing.wake_lock.lock().unwrap_or_else(|p| p.into_inner());
        self.backing.closed.store(true, Ordering::Release);
        self.backing.wake.notify_all();
        if let Some(node) = self.retirement.take() {
            self.device.retire_resource(node);
        }
    }
}

#[derive(Default)]
pub(super) struct ExternalWaitBatch {
    limit: usize,
    imported: usize,
}
impl ExternalWaitBatch {
    pub(super) fn begin(&mut self, limit: usize) {
        self.limit = limit;
        self.imported = 0;
    }
    pub(super) fn check(&self) -> Result<(), VulkanRendererError> {
        if self.imported < self.limit {
            Ok(())
        } else {
            Err(VulkanRendererError::CommandStorageLimitExceeded {
                resource: "admitted external waits",
                requested: self.imported.saturating_add(1),
                limit: self.limit,
            })
        }
    }
    pub(super) fn imported(&mut self) {
        self.imported += 1;
    }
}

#[cfg(test)]
#[path = "external_wait_storage_tests.rs"]
mod tests;

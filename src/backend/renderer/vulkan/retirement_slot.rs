//! Cold-reserved custody for foreign output resources released on the GPU actor.

use std::{marker::PhantomData, sync::Arc};

use super::{
    device_handle::{DeviceHandle, DeviceRetirement},
    retirement::RetirementNode,
};

/// One cold-reserved retirement publication. Fill only with resources whose
/// reader completion has already been proven by their owning protocol. Healthy
/// disposal runs on the existing device actor; device loss quarantines the exact
/// owner because its native destructor may otherwise call a lost device.
pub struct VulkanRetirementSlot<T: Send + Sync + 'static> {
    device: Arc<DeviceHandle>,
    node: Option<Box<RetirementNode<DeviceRetirement>>>,
    _value: PhantomData<T>,
}

impl<T: Send + Sync + 'static> std::fmt::Debug for VulkanRetirementSlot<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanRetirementSlot")
            .field("reserved", &self.node.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::super::device_handle::retirement_tests::{device, next, Operation};
    use super::*;

    struct Owner(std::sync::mpsc::Sender<std::thread::ThreadId>);
    impl Drop for Owner {
        fn drop(&mut self) {
            let _ = self.0.send(std::thread::current().id());
        }
    }

    #[test]
    fn reserved_output_owner_drops_on_device_actor_before_device_parent() {
        let (device, events) = device();
        let slot = VulkanRetirementSlot::new(device.clone());
        let (destroyed, received) = std::sync::mpsc::channel();
        let caller = std::thread::spawn(move || {
            let caller = std::thread::current().id();
            slot.retire(Owner(destroyed));
            caller
        })
        .join()
        .unwrap();
        let executor = received.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert_ne!(caller, executor);
        drop(device);
        assert_eq!(next(&events), (Operation::Device, executor));
        assert_eq!(next(&events), (Operation::Parent, executor));
    }

    #[test]
    fn lost_device_quarantines_foreign_native_owner_instead_of_dropping_it() {
        let (device, events) = device();
        let slot = VulkanRetirementSlot::new(device.clone());
        let (destroyed, received) = std::sync::mpsc::channel();
        device.mark_lost();
        slot.retire(Owner(destroyed));
        drop(device);
        assert_eq!(next(&events).0, Operation::Parent);
        assert!(received.try_recv().is_err());
    }
}

impl<T: Send + Sync + 'static> VulkanRetirementSlot<T> {
    pub(super) fn new(device: Arc<DeviceHandle>) -> Self {
        let node = RetirementNode::new(DeviceRetirement::OpaqueCustody(Box::new(None::<T>)));
        Self {
            device,
            node: Some(node),
            _value: PhantomData,
        }
    }

    /// Move an exact owner into the reserved publication without allocation,
    /// locking, GPU calls or completion waits on the releasing thread.
    pub fn retire(mut self, value: T) {
        let node = self.node.as_mut().expect("retirement publication is unused");
        let DeviceRetirement::OpaqueCustody(custody) = node.value_mut() else {
            unreachable!()
        };
        *custody
            .downcast_mut::<Option<T>>()
            .expect("retirement custody type is immutable") = Some(value);
        // Drop publishes the filled node. The caller keeps no shadow owner.
    }
}

impl<T: Send + Sync + 'static> Drop for VulkanRetirementSlot<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            self.device.retire_resource(node);
        }
    }
}

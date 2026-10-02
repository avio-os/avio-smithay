//! Queue global priority (`VK_KHR_global_priority`).
//!
//! A compositor shares the GPU with every other client. When a build, a game or another
//! compositor client keeps the GPU busy, the kernel's GPU scheduler decides whose work runs
//! next, and a compositor frame that waits behind that work misses its vblank. Drivers that
//! implement `VK_KHR_global_priority` (or its predecessor `VK_EXT_global_priority`) let a
//! logical device's queues ask that scheduler for a priority above the default.
//!
//! A priority above [`QueueGlobalPriority::Medium`] is usually privileged. Linux DRM drivers
//! check the caller's `CAP_SYS_NICE` when the queue is created (Mesa's ANV on `xe` maps
//! [`QueueGlobalPriority::High`] to `DRM_SCHED_PRIORITY_HIGH`, which the kernel grants only with
//! that capability), and the driver then fails `vkCreateDevice` with
//! `VK_ERROR_NOT_PERMITTED_KHR`. The Vulkan specification allows exactly that answer for a
//! request the caller may not make.
//!
//! [`create_device_with_queue_priority`] keeps the request optional and recoverable: it asks
//! once, creates the device once more at the default priority if the driver answers
//! `VK_ERROR_NOT_PERMITTED_KHR`, and reports what the device's queues were granted in a
//! [`QueuePriorityGrant`]. Any other error is the caller's, unchanged.

use std::ffi::CStr;

use ash::vk;

use super::PhysicalDevice;

/// A queue global priority level (`VkQueueGlobalPriorityKHR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueueGlobalPriority {
    /// Below the default.
    Low,
    /// The default priority of every queue that does not ask for one.
    Medium,
    /// Above the default; usually needs a privilege such as `CAP_SYS_NICE`.
    High,
    /// The highest level a driver offers; usually needs a privilege.
    Realtime,
}

impl QueueGlobalPriority {
    /// The priority a queue has when nothing was requested or a request was refused.
    pub const DEFAULT: Self = Self::Medium;

    /// The Vulkan value of this priority.
    pub fn as_vk(self) -> vk::QueueGlobalPriorityKHR {
        match self {
            Self::Low => vk::QueueGlobalPriorityKHR::LOW,
            Self::Medium => vk::QueueGlobalPriorityKHR::MEDIUM,
            Self::High => vk::QueueGlobalPriorityKHR::HIGH,
            Self::Realtime => vk::QueueGlobalPriorityKHR::REALTIME,
        }
    }

    /// A short lowercase name for logs: `low`, `medium`, `high` or `realtime`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Realtime => "realtime",
        }
    }
}

/// What became of a queue global priority request at device creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueuePriorityOutcome {
    /// Nothing was requested; the queues run at the default priority.
    NotRequested,
    /// The device supports neither global-priority extension, so nothing was requested and the
    /// queues run at the default priority.
    Unsupported,
    /// The driver granted the requested priority.
    Granted,
    /// The driver refused the request with `VK_ERROR_NOT_PERMITTED_KHR`, and the device was
    /// created once more at the default priority.
    NotPermitted,
}

impl QueuePriorityOutcome {
    /// A short lowercase name for logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Unsupported => "unsupported",
            Self::Granted => "granted",
            Self::NotPermitted => "not_permitted",
        }
    }
}

/// The priority a logical device's queues were created with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueuePriorityGrant {
    /// The priority that was asked for, if any.
    pub requested: Option<QueueGlobalPriority>,
    /// What the driver made of the request.
    pub outcome: QueuePriorityOutcome,
}

impl QueuePriorityGrant {
    /// A device created without a request.
    pub const NOT_REQUESTED: Self = Self {
        requested: None,
        outcome: QueuePriorityOutcome::NotRequested,
    };

    /// The priority the device's queues run at.
    pub fn granted(&self) -> QueueGlobalPriority {
        match (self.outcome, self.requested) {
            (QueuePriorityOutcome::Granted, Some(priority)) => priority,
            _ => QueueGlobalPriority::DEFAULT,
        }
    }
}

/// One queue global priority request, handed to the `vkCreateDevice` call that makes it.
///
/// The caller enables [`QueuePriorityRequest::extension`] on the device and chains
/// [`QueuePriorityRequest::create_info`] onto each `VkDeviceQueueCreateInfo` that should run at
/// the requested priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueuePriorityRequest {
    priority: QueueGlobalPriority,
    extension: &'static CStr,
}

impl QueuePriorityRequest {
    /// The requested priority.
    pub fn priority(&self) -> QueueGlobalPriority {
        self.priority
    }

    /// The device extension that carries the request; it must be enabled on the device.
    pub fn extension(&self) -> &'static CStr {
        self.extension
    }

    /// The structure to chain onto a `VkDeviceQueueCreateInfo`.
    pub fn create_info(&self) -> vk::DeviceQueueGlobalPriorityCreateInfoKHR<'static> {
        vk::DeviceQueueGlobalPriorityCreateInfoKHR::default().global_priority(self.priority.as_vk())
    }
}

impl PhysicalDevice {
    /// The device extension that carries a queue global priority request on this device,
    /// preferring `VK_KHR_global_priority` over `VK_EXT_global_priority`, or `None` when the
    /// device supports neither.
    pub fn queue_global_priority_extension(&self) -> Option<&'static CStr> {
        [ash::khr::global_priority::NAME, ash::ext::global_priority::NAME]
            .into_iter()
            .find(|extension| self.has_device_extension(extension))
    }
}

/// Creates a logical device whose queues ask for `requested`, recovering once from a refusal.
///
/// `create` makes one `vkCreateDevice` call. With `Some(request)` it enables
/// [`QueuePriorityRequest::extension`] and chains [`QueuePriorityRequest::create_info`] onto its
/// queue create infos; with `None` it creates the device exactly as it would without a request.
///
/// - Without `requested`, or on a device that supports neither global-priority extension,
///   `create` runs once with `None`.
/// - A device the driver creates with the request reports [`QueuePriorityOutcome::Granted`].
/// - If the driver answers `VK_ERROR_NOT_PERMITTED_KHR`, `create` runs once more with `None`,
///   and the device reports [`QueuePriorityOutcome::NotPermitted`]. That second call is the only
///   retry; its error, like any other error of the first call, is returned unchanged.
pub fn create_device_with_queue_priority<T>(
    physical_device: &PhysicalDevice,
    requested: Option<QueueGlobalPriority>,
    create: impl FnMut(Option<&QueuePriorityRequest>) -> Result<T, vk::Result>,
) -> Result<(T, QueuePriorityGrant), vk::Result> {
    request_with_fallback(
        requested,
        physical_device.queue_global_priority_extension(),
        create,
    )
}

fn request_with_fallback<T>(
    requested: Option<QueueGlobalPriority>,
    extension: Option<&'static CStr>,
    mut create: impl FnMut(Option<&QueuePriorityRequest>) -> Result<T, vk::Result>,
) -> Result<(T, QueuePriorityGrant), vk::Result> {
    let Some(priority) = requested else {
        return create(None).map(|device| (device, QueuePriorityGrant::NOT_REQUESTED));
    };
    let grant = |outcome| QueuePriorityGrant {
        requested: Some(priority),
        outcome,
    };
    let Some(extension) = extension else {
        return create(None).map(|device| (device, grant(QueuePriorityOutcome::Unsupported)));
    };
    let request = QueuePriorityRequest { priority, extension };
    match create(Some(&request)) {
        Ok(device) => Ok((device, grant(QueuePriorityOutcome::Granted))),
        Err(vk::Result::ERROR_NOT_PERMITTED_KHR) => {
            create(None).map(|device| (device, grant(QueuePriorityOutcome::NotPermitted)))
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use ash::vk;

    use super::{
        request_with_fallback, QueueGlobalPriority, QueuePriorityGrant, QueuePriorityOutcome,
        QueuePriorityRequest,
    };

    const KHR: &CStr = ash::khr::global_priority::NAME;

    /// Runs the fallback against scripted `vkCreateDevice` answers and records the request each
    /// call carried, if any.
    fn run(
        requested: Option<QueueGlobalPriority>,
        extension: Option<&'static CStr>,
        answers: Vec<Result<u32, vk::Result>>,
    ) -> (
        Result<(u32, QueuePriorityGrant), vk::Result>,
        Vec<Option<QueuePriorityRequest>>,
    ) {
        let mut answers = answers.into_iter();
        let mut calls = Vec::new();
        let result = request_with_fallback(requested, extension, |request| {
            calls.push(request.copied());
            answers.next().expect("no more create calls were expected")
        });
        (result, calls)
    }

    #[test]
    fn granted_request_creates_the_device_once_with_the_request() {
        let (result, calls) = run(Some(QueueGlobalPriority::High), Some(KHR), vec![Ok(7)]);
        let (device, grant) = result.unwrap();
        assert_eq!(device, 7);
        assert_eq!(grant.outcome, QueuePriorityOutcome::Granted);
        assert_eq!(grant.granted(), QueueGlobalPriority::High);
        assert_eq!(calls.len(), 1);
        let request = calls[0].expect("the first call carries the request");
        assert_eq!(request.priority(), QueueGlobalPriority::High);
        assert_eq!(request.extension(), KHR);
        assert_eq!(
            request.create_info().global_priority,
            vk::QueueGlobalPriorityKHR::HIGH
        );
    }

    #[test]
    fn refused_request_falls_back_to_the_default_exactly_once() {
        let (result, calls) = run(
            Some(QueueGlobalPriority::High),
            Some(KHR),
            vec![Err(vk::Result::ERROR_NOT_PERMITTED_KHR), Ok(9)],
        );
        let (device, grant) = result.unwrap();
        assert_eq!(device, 9);
        assert_eq!(grant.requested, Some(QueueGlobalPriority::High));
        assert_eq!(grant.outcome, QueuePriorityOutcome::NotPermitted);
        assert_eq!(grant.granted(), QueueGlobalPriority::DEFAULT);
        assert_eq!(calls.len(), 2);
        assert!(calls[0].is_some(), "the first call asks");
        assert!(
            calls[1].is_none(),
            "the fallback creates the device without a request"
        );
    }

    #[test]
    fn a_failing_fallback_is_not_retried_again() {
        let (result, calls) = run(
            Some(QueueGlobalPriority::High),
            Some(KHR),
            vec![
                Err(vk::Result::ERROR_NOT_PERMITTED_KHR),
                Err(vk::Result::ERROR_NOT_PERMITTED_KHR),
            ],
        );
        assert_eq!(result.unwrap_err(), vk::Result::ERROR_NOT_PERMITTED_KHR);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn other_errors_are_returned_without_a_fallback() {
        let (result, calls) = run(
            Some(QueueGlobalPriority::High),
            Some(KHR),
            vec![Err(vk::Result::ERROR_INITIALIZATION_FAILED)],
        );
        assert_eq!(result.unwrap_err(), vk::Result::ERROR_INITIALIZATION_FAILED);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn unsupported_devices_are_created_without_a_request() {
        let (result, calls) = run(Some(QueueGlobalPriority::High), None, vec![Ok(3)]);
        let (_, grant) = result.unwrap();
        assert_eq!(grant.outcome, QueuePriorityOutcome::Unsupported);
        assert_eq!(grant.granted(), QueueGlobalPriority::DEFAULT);
        assert_eq!(calls, vec![None]);
    }

    #[test]
    fn no_request_creates_the_device_once_without_one() {
        let (result, calls) = run(None, Some(KHR), vec![Ok(1)]);
        let (_, grant) = result.unwrap();
        assert_eq!(grant, QueuePriorityGrant::NOT_REQUESTED);
        assert_eq!(grant.granted(), QueueGlobalPriority::DEFAULT);
        assert_eq!(calls, vec![None]);
    }

    #[test]
    fn priorities_map_to_their_vulkan_values() {
        assert_eq!(QueueGlobalPriority::Low.as_vk(), vk::QueueGlobalPriorityKHR::LOW);
        assert_eq!(
            QueueGlobalPriority::Medium.as_vk(),
            vk::QueueGlobalPriorityKHR::MEDIUM
        );
        assert_eq!(
            QueueGlobalPriority::High.as_vk(),
            vk::QueueGlobalPriorityKHR::HIGH
        );
        assert_eq!(
            QueueGlobalPriority::Realtime.as_vk(),
            vk::QueueGlobalPriorityKHR::REALTIME
        );
        assert_eq!(QueueGlobalPriority::DEFAULT, QueueGlobalPriority::Medium);
    }
}

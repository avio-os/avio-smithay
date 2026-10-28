use super::FrameError;
use crate::backend::drm::{error::AccessError, DrmError};
use drm::control::from_u32;
use std::convert::Infallible;

type Error = FrameError<Infallible, Infallible, Infallible>;

#[test]
fn queue_evidence_distinguishes_local_and_test_refusal_from_physical_ioctl() {
    for error in [
        Error::EmptyFrame,
        Error::DrmError(DrmError::DeviceInactive),
        Error::DrmError(DrmError::AtomicRequestBusy),
        Error::DrmError(DrmError::TestFailed(from_u32(1).unwrap())),
        Error::DrmError(DrmError::AtomicRequestCapacity {
            resource: "test request",
            required: 2,
            capacity: 1,
        }),
    ] {
        assert!(!error.queue_submission_attempted());
    }
    let rejected = Error::DrmError(DrmError::Access(AccessError {
        errmsg: "Page flip commit failed",
        dev: None,
        source: std::io::Error::from_raw_os_error(libc::EBUSY),
    }));
    assert!(rejected.queue_submission_attempted());
}

//! GBM allocation through one declared API with exact layout verification.
use super::*;

/// An allocator that preserves the requested usages and explicit layout.
///
/// A singleton linear request uses `gbm_bo_create` with `GBM_BO_USE_LINEAR`.
/// Other requests use `gbm_bo_create_with_modifiers2`. Failure never retries
/// with different usages or changes an unknown modifier into a known one.
#[derive(Debug, Clone)]
pub struct ExplicitGbmAllocator<A: AsFd + 'static> {
    device: GbmDevice<A>,
    flags: GbmBufferFlags,
}

impl<A: AsFd + 'static> ExplicitGbmAllocator<A> {
    /// Select the device and the complete set of intended usages.
    pub fn new(device: GbmDevice<A>, flags: GbmBufferFlags) -> Self {
        Self { device, flags }
    }
}

impl<A: AsFd + 'static> AsRef<GbmDevice<A>> for ExplicitGbmAllocator<A> {
    fn as_ref(&self) -> &GbmDevice<A> {
        &self.device
    }
}

fn valid_layout(modifiers: &[Modifier], actual: Modifier, planes: u32) -> bool {
    !modifiers.is_empty()
        && !modifiers.contains(&Modifier::Invalid)
        && actual != Modifier::Invalid
        && modifiers.contains(&actual)
        && (actual != Modifier::Linear || planes == 1)
}

impl<A: AsFd + 'static> Allocator for ExplicitGbmAllocator<A> {
    type Buffer = GbmBuffer;
    type Error = std::io::Error;

    fn create_buffer(
        &mut self,
        width: u32,
        height: u32,
        fourcc: Fourcc,
        modifiers: &[Modifier],
    ) -> Result<Self::Buffer, Self::Error> {
        if modifiers.is_empty() || modifiers.contains(&Modifier::Invalid) {
            return Err(std::io::Error::other("explicit GBM layout required"));
        }
        let bo = if modifiers == [Modifier::Linear] {
            self.device
                .create_buffer_object::<()>(width, height, fourcc, self.flags | GbmBufferFlags::LINEAR)
        } else {
            #[cfg(feature = "backend_gbm_has_create_with_modifiers2")]
            {
                self.device.create_buffer_object_with_modifiers2::<()>(
                    width,
                    height,
                    fourcc,
                    modifiers.iter().copied(),
                    self.flags,
                )
            }
            #[cfg(not(feature = "backend_gbm_has_create_with_modifiers2"))]
            return Err(std::io::Error::other(
                "explicit modifier allocation with usages is unavailable",
            ));
        }?;
        crate::backend::allocator::note_gpu_allocation(
            crate::backend::allocator::GpuAllocationKind::GbmBuffer,
        );
        if !valid_layout(modifiers, bo.modifier(), bo.plane_count())
            || bo.width() != width
            || bo.height() != height
            || bo.format() != fourcc
        {
            return Err(std::io::Error::other(
                "GBM returned a different explicit allocation contract",
            ));
        }
        Ok(GbmBuffer::from_bo(bo, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_allocation_requires_reported_single_plane_linear_layout() {
        assert!(valid_layout(&[Modifier::Linear], Modifier::Linear, 1));
        assert!(!valid_layout(&[Modifier::Linear], Modifier::Invalid, 1));
        assert!(!valid_layout(&[Modifier::Linear], Modifier::Linear, 2));
        assert!(!valid_layout(&[Modifier::Invalid], Modifier::Linear, 1));
        assert!(!valid_layout(&[], Modifier::Linear, 1));
    }
}

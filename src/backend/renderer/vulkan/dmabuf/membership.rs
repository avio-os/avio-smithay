//! Explicit owner pins and source-identity retirement.
use super::*;

impl DmabufState {
    pub(crate) fn pin_texture(
        &mut self,
        device: &DeviceState,
        formats: &FormatCapabilities,
        dmabuf: &Dmabuf,
    ) -> Result<VulkanTexture, VulkanRendererError> {
        let imported = self.import_or_reuse(device, formats, dmabuf, DmabufRole::Texture, true)?;
        Ok(VulkanTexture::from_dmabuf_import(
            imported,
            dmabuf.size(),
            Some(dmabuf.format().code),
            dmabuf.y_inverted(),
        ))
    }

    pub(crate) fn unpin(&mut self, dmabuf: &Dmabuf) -> bool {
        self.unpin_weak(&dmabuf.weak())
    }

    pub(crate) fn unpin_weak(&mut self, key: &WeakDmabuf) -> bool {
        let Some(cached) = self.cache.get_mut(key) else {
            return false;
        };
        let Some(pins) = cached.pins.checked_sub(1) else {
            return false;
        };
        cached.pins = pins;
        true
    }

    /// Remove owner residency even while readers retain the exact old image.
    /// Submitted work's Arc custody, not cache membership, guards GPU safety.
    pub(crate) fn retire_sampled(&mut self, buffers: &[WeakDmabuf]) -> usize {
        let mut retired = 0usize;
        for buffer in buffers {
            let retire = self.cache.get(buffer).is_some_and(|cached| {
                cached.pins == 0
                    && cached
                        .imported
                        .upgrade()
                        .is_some_and(|image| idle_evictable_usage(image.usage()))
            });
            if retire {
                self.cache.shift_remove(buffer);
                retired += 1;
            }
        }
        self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(retired as u64);
        retired
    }

    pub(crate) fn prepare_frame_client_sources(&mut self, sources: &[WeakDmabuf]) {
        self.frame_client_sources.clear();
        self.frame_client_sources.extend(sources.iter().cloned());
    }

    pub(crate) fn set_frame_client_scope(&mut self, active: bool) {
        self.frame_client_scope = active;
    }

    pub(crate) fn client_first_imports_on_frame(&self) -> u64 {
        self.client_first_imports_on_frame.total()
    }

    pub(crate) fn client_import_observer(&self) -> super::super::VulkanClientImportObserver {
        self.client_first_imports_on_frame.observer()
    }
}

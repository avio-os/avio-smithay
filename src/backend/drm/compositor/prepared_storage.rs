//! Cold factory products and exact retirement of DRM CPU workspace.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// CPU storage prepared by an untagged owner for one DRM output.
///
/// Adoption moves storage, not output/scene authority. The displaced product
/// must return to that same cold owner and remain alive until every independent
/// state-map, selection and native plane receipt has returned. No fence or age
/// counter substitutes for those actual reader lifetimes.
pub struct PreparedDrmFrameStorage<B: Buffer, F: Framebuffer> {
    pub(super) selection: Option<DrmFrameStorage<B, F>>,
    pub(super) states: Option<StateMapBank>,
    pub(super) primary_damage: OutputDamageTracker,
    pub(super) layer_damage: OutputDamageTracker,
    pub(super) element_states: IndexMap<Id, ElementState<F>>,
    pub(super) previous_element_states: IndexMap<Id, ElementState<F>>,
    pub(super) old_current: Option<FrameState<B, F>>,
    pub(super) retirement_armed: Option<Arc<AtomicBool>>,
}

impl<B: Buffer, F: Framebuffer> std::fmt::Debug for PreparedDrmFrameStorage<B, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedDrmFrameStorage")
            .field("state_receipts", &self.state_receipt_capacity())
            .field(
                "element_capacity",
                &self.states.as_ref().map(StateMapBank::entries),
            )
            .field(
                "retirement_armed",
                &self.retirement_armed.as_ref().map(|a| a.load(Ordering::Acquire)),
            )
            .finish_non_exhaustive()
    }
}

impl<B: Buffer, F: Framebuffer> PreparedDrmFrameStorage<B, F> {
    /// Allocate the admitted CPU inventories only on the cold owner.
    ///
    /// `elements` is the actual root/native retained-ID bound, `plane_capacity`
    /// is the actual native roster bound, and synthetic overlays are the
    /// output's configured physical overlay count. `receipts` is an explicit
    /// finite Main result-lane admission, independent of KMS queue depth.
    /// Return callbacks must only enqueue/signal a bounded owning control and
    /// remain independently retained by the cold owner through worker teardown.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode_source: OutputModeSource,
        elements: usize,
        rectangles: usize,
        receipts: usize,
        policy: DamageStoragePolicy,
        plane_capacity: usize,
        synthetic_overlay_elements: usize,
        receipt_returned: Option<Arc<dyn Fn() + Send + Sync>>,
        retired_storage_returned: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Result<Self, FrameWorkspaceError> {
        if rectangles == 0 || receipts == 0 || plane_capacity == 0 {
            return Err(FrameWorkspaceError {
                resource: "cold DRM frame storage",
                required: 1,
                capacity: 0,
            });
        }
        let damage_elements =
            elements
                .checked_add(synthetic_overlay_elements)
                .ok_or(FrameWorkspaceError {
                    resource: "DRM damage element capacity",
                    required: usize::MAX,
                    capacity: elements,
                })?;
        let armed = Arc::new(AtomicBool::new(false));
        let retirement =
            StateReceiptReturnWakeup::with_retirement(None, armed.clone(), retired_storage_returned.clone());
        let returned = StateReceiptReturnWakeup::with_retirement(
            receipt_returned,
            armed.clone(),
            retired_storage_returned,
        );
        let mut primary_damage = OutputDamageTracker::from_mode_source(mode_source.clone());
        let mut layer_damage = OutputDamageTracker::from_mode_source(mode_source);
        primary_damage.prepare_frame_storage_with_return_wakeup(
            damage_elements,
            rectangles,
            2,
            policy,
            Some(retirement.clone()),
        )?;
        layer_damage.prepare_frame_storage_with_return_wakeup(
            elements,
            rectangles,
            2,
            policy,
            Some(retirement.clone()),
        )?;
        Ok(Self {
            selection: Some(DrmFrameStorage::new_with_wakeup(
                elements,
                plane_capacity,
                rectangles,
                receipts,
                policy,
                Some(retirement),
            )?),
            states: Some(StateMapBank::new_with_wakeup(elements, receipts, Some(returned))),
            primary_damage,
            layer_damage,
            element_states: IndexMap::with_capacity(elements),
            previous_element_states: IndexMap::with_capacity(elements),
            old_current: None,
            retirement_armed: Some(armed),
        })
    }

    /// Actual admitted independent returned state-map slots.
    pub fn state_receipt_capacity(&self) -> usize {
        self.states.as_ref().map_or(0, StateMapBank::receipt_capacity)
    }

    /// Arm displaced storage on its untagged owner, dropping the old current
    /// duplicate there. Check `is_reclaimable` immediately after this call;
    /// subsequent exact last-reader returns signal the registered cold owner.
    pub fn arm_retirement(&mut self) {
        self.old_current = None;
        if let Some(armed) = &self.retirement_armed {
            armed.store(true, Ordering::Release);
        }
    }

    /// Every storage lease has actually returned. Active products and parked
    /// current snapshots are never reclaimable even when GPU work has ended.
    pub fn is_reclaimable(&self) -> bool {
        self.old_current.is_none()
            && self
                .retirement_armed
                .as_ref()
                .is_none_or(|a| a.load(Ordering::Acquire))
            && self.states.as_ref().is_none_or(StateMapBank::is_reclaimable)
            && self
                .selection
                .as_ref()
                .is_none_or(DrmFrameStorage::is_reclaimable)
            && self.primary_damage.storage_receipts_reclaimable()
            && self.layer_damage.storage_receipts_reclaimable()
    }
}

/// Rejected adoption returns the original cold product to its owner.
pub struct FrameStorageAdoptionError<B: Buffer, F: Framebuffer> {
    /// Exact capacity or roster mismatch; no native submission occurred.
    pub error: FrameWorkspaceError,
    /// Unchanged rejected product, disposed or retried by the cold owner.
    pub storage: PreparedDrmFrameStorage<B, F>,
}
impl<B: Buffer, F: Framebuffer> std::fmt::Debug for FrameStorageAdoptionError<B, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameStorageAdoptionError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<A, F, U, G> DrmCompositor<A, F, U, G>
where
    A: Allocator,
    A::Error: std::error::Error + Send + Sync,
    A::Buffer: AsDmabuf,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + std::fmt::Debug,
    F: ExportFramebuffer<A::Buffer>,
    F::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    F::Error: std::error::Error + Send + Sync,
    G: AsFd + Clone,
{
    /// Adopt an already allocated cold product without reallocating workspace.
    /// All rejection checks precede changing current/native/cache state. The
    /// returned displaced product must be retained by its cold owner until its
    /// exact reader leases return; do not discard it on the render thread.
    #[allow(clippy::result_large_err)]
    pub fn adopt_frame_storage(
        &mut self,
        mut prepared: PreparedDrmFrameStorage<A::Buffer, F::Framebuffer>,
    ) -> Result<
        PreparedDrmFrameStorage<A::Buffer, F::Framebuffer>,
        FrameStorageAdoptionError<A::Buffer, F::Framebuffer>,
    > {
        let roster = self.frame_storage_plane_capacity();
        let check = (|| {
            let selection = prepared.selection.as_ref().ok_or(FrameWorkspaceError {
                resource: "DRM prepared selection storage",
                required: 1,
                capacity: 0,
            })?;
            if prepared.old_current.is_some()
                || prepared
                    .retirement_armed
                    .as_ref()
                    .is_none_or(|a| a.load(Ordering::Acquire))
            {
                return Err(FrameWorkspaceError {
                    resource: "retired DRM storage adoption",
                    required: 1,
                    capacity: 0,
                });
            }
            if selection.plane_capacity() < roster {
                return Err(FrameWorkspaceError {
                    resource: "DRM native plane roster",
                    required: roster,
                    capacity: selection.plane_capacity(),
                });
            }
            let entries = prepared
                .states
                .as_ref()
                .ok_or(FrameWorkspaceError {
                    resource: "DRM prepared returned state maps",
                    required: 1,
                    capacity: 0,
                })?
                .entries();
            for (map, required) in [
                (&prepared.element_states, self.element_states.len()),
                (
                    &prepared.previous_element_states,
                    self.previous_element_states.len(),
                ),
            ] {
                if entries < required || map.capacity() < required {
                    return Err(FrameWorkspaceError {
                        resource: "DRM retained framebuffer state",
                        required,
                        capacity: entries.min(map.capacity()),
                    });
                }
            }
            // Retained native configuration metadata is copied into a
            // distinct admitted lane. Current/pending/queued keep their own
            // strong native resources and exact reset claims.
            let mut current = self.current_frame.copy_for_update(Some(selection))?;
            current.opaque_black = self.current_frame.opaque_black;
            Ok(current)
        })();
        let current = match check {
            Ok(current) => current,
            Err(error) => {
                return Err(FrameStorageAdoptionError {
                    error,
                    storage: prepared,
                })
            }
        };
        prepared
            .primary_damage
            .set_mode_source_preserving_storage(self.output_mode_source.clone());
        prepared
            .layer_damage
            .set_mode_source_preserving_storage(self.output_mode_source.clone());
        prepared.element_states.extend(self.element_states.drain(..));
        prepared
            .previous_element_states
            .extend(self.previous_element_states.drain(..));
        std::mem::swap(&mut self.element_states, &mut prepared.element_states);
        std::mem::swap(
            &mut self.previous_element_states,
            &mut prepared.previous_element_states,
        );
        std::mem::swap(&mut self.damage_tracker, &mut prepared.primary_damage);
        std::mem::swap(&mut self.output_layer_damage_tracker, &mut prepared.layer_damage);
        std::mem::swap(&mut self.frame_state_bank, &mut prepared.states);
        std::mem::swap(&mut self.selection_storage, &mut prepared.selection);
        std::mem::swap(&mut self.storage_retirement_armed, &mut prepared.retirement_armed);
        prepared.old_current = Some(std::mem::replace(&mut self.current_frame, current));
        self.reset_pending = true;
        Ok(prepared)
    }
}

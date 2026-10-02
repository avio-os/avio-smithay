//! Exercise the real damage error exits with a segmented CPU Frame test double.
//! Native Vulkan submission itself is covered by the ash ABI custody fixtures.
use super::*;
use crate::backend::renderer::{
    element::solid::{SolidColorBuffer, SolidColorRenderElement},
    test::{DummyFramebuffer, DummyRenderer, DummyTexture},
    vulkan::VulkanRendererError,
    ContextId, DebugFlags, RendererSuper, TextureFilter,
};
use crate::utils::Buffer;

#[derive(Debug)]
struct SegmentedRenderer {
    fail_after: usize,
    fail_finish: bool,
    successful_segments: usize,
}
#[derive(Debug)]
struct SegmentedFrame<'a> {
    renderer: &'a mut SegmentedRenderer,
    submitted: bool,
}
impl RendererSuper for SegmentedRenderer {
    type Error = VulkanRendererError;
    type TextureId = DummyTexture;
    type Framebuffer<'a> = DummyFramebuffer;
    type Frame<'a, 'b>
        = SegmentedFrame<'a>
    where
        'b: 'a,
        Self: 'a;
}
impl Renderer for SegmentedRenderer {
    fn context_id(&self) -> ContextId<DummyTexture> {
        DummyRenderer.context_id()
    }
    fn downscale_filter(&mut self, _: TextureFilter) -> Result<(), Self::Error> {
        Ok(())
    }
    fn upscale_filter(&mut self, _: TextureFilter) -> Result<(), Self::Error> {
        Ok(())
    }
    fn set_debug_flags(&mut self, _: DebugFlags) {}
    fn debug_flags(&self) -> DebugFlags {
        DebugFlags::empty()
    }
    fn render<'a, 'b>(
        &'a mut self,
        _: &'a mut DummyFramebuffer,
        _: Size<i32, Physical>,
        _: Transform,
    ) -> Result<SegmentedFrame<'a>, Self::Error>
    where
        'b: 'a,
    {
        Ok(SegmentedFrame {
            renderer: self,
            submitted: false,
        })
    }
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        sync.wait()
            .map_err(|_| VulkanRendererError::CommandCompletionUnavailable)
    }
}
impl Frame for SegmentedFrame<'_> {
    type Error = VulkanRendererError;
    type TextureId = DummyTexture;
    fn context_id(&self) -> ContextId<DummyTexture> {
        self.renderer.context_id()
    }
    fn clear(&mut self, _: Color32F, _: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        Ok(())
    }
    fn draw_solid(
        &mut self,
        _: Rectangle<i32, Physical>,
        _: &[Rectangle<i32, Physical>],
        _: Color32F,
    ) -> Result<(), Self::Error> {
        if self.renderer.successful_segments >= self.renderer.fail_after {
            return Err(VulkanRendererError::CommandCapacityExhausted { slots: 1 });
        }
        self.renderer.successful_segments += 1;
        self.submitted = true;
        Ok(())
    }
    fn render_texture_from_to(
        &mut self,
        _: &DummyTexture,
        _: Rectangle<f64, Buffer>,
        _: Rectangle<i32, Physical>,
        _: &[Rectangle<i32, Physical>],
        _: &[Rectangle<i32, Physical>],
        _: Transform,
        _: f32,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn transformation(&self) -> Transform {
        Transform::Normal
    }
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.wait(sync)
    }
    fn completion_unobservable_on_error(&self) -> bool {
        self.submitted
    }
    fn finish(self) -> Result<SyncPoint, Self::Error> {
        if self.renderer.fail_finish {
            Err(VulkanRendererError::CommandCapacityExhausted { slots: 1 })
        } else {
            Ok(SyncPoint::signaled())
        }
    }
}
fn elements() -> [SolidColorRenderElement; 2] {
    std::array::from_fn(|_| {
        let buffer = SolidColorBuffer::new((20, 20), Color32F::new(0.5, 0.5, 0.5, 0.5));
        SolidColorRenderElement::from_buffer(&buffer, (10, 10), 1.0, 1.0, Kind::Unspecified)
    })
}
#[test]
fn pressure_after_first_submitted_draw_is_typed_unobservable_and_resets_history() {
    let mut renderer = SegmentedRenderer {
        fail_after: 1,
        fail_finish: false,
        successful_segments: 0,
    };
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(2, 64, 2).unwrap();
    let error = tracker
        .render_output(
            &mut renderer,
            &mut DummyFramebuffer,
            0,
            &elements(),
            Color32F::BLACK,
        )
        .unwrap_err();
    assert_eq!(renderer.successful_segments, 1);
    assert!(matches!(
        error,
        Error::PartialFrameCompletionUnobservable(VulkanRendererError::CommandCapacityExhausted { slots: 1 })
    ));
    assert!(error.is_completion_unobservable());
    assert!(!error.is_device_lost());
    assert!(tracker.last_state.size.is_none());
}
#[test]
fn pressure_before_any_submitted_draw_keeps_ordinary_retry_classification() {
    let mut renderer = SegmentedRenderer {
        fail_after: 0,
        fail_finish: false,
        successful_segments: 0,
    };
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(2, 64, 2).unwrap();
    let error = tracker
        .render_output(
            &mut renderer,
            &mut DummyFramebuffer,
            0,
            &elements(),
            Color32F::BLACK,
        )
        .unwrap_err();
    assert_eq!(renderer.successful_segments, 0);
    assert!(matches!(
        error,
        Error::Rendering(VulkanRendererError::CommandCapacityExhausted { slots: 1 })
    ));
    assert!(!error.is_completion_unobservable());
}
#[test]
fn final_finish_pressure_preserves_prior_segment_custody() {
    let mut renderer = SegmentedRenderer {
        fail_after: 2,
        fail_finish: true,
        successful_segments: 0,
    };
    let mut tracker = OutputDamageTracker::new((100, 100), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(2, 64, 2).unwrap();
    let error = tracker
        .render_output(
            &mut renderer,
            &mut DummyFramebuffer,
            0,
            &elements(),
            Color32F::BLACK,
        )
        .unwrap_err();
    assert_eq!(renderer.successful_segments, 2);
    assert!(matches!(error, Error::PartialFrameCompletionUnobservable(_)));
    assert!(error.is_completion_unobservable());
}
#[test]
fn partial_wrap_retains_positive_device_loss_and_workspace_cause_without_allocation() {
    let error =
        Error::from_frame_rendering(VulkanRendererError::Vk(ash::vk::Result::ERROR_DEVICE_LOST), true);
    assert!(error.is_completion_unobservable());
    assert!(error.is_device_lost());
    let refusal = FrameWorkspaceError {
        resource: "damage",
        required: 2,
        capacity: 1,
    };
    let (_, operations) = crate::backend::renderer::storage_heap_probe::measure(|| {
        let error = Error::<VulkanRendererError>::from_frame_workspace(refusal, true);
        assert!(error.is_completion_unobservable());
        assert!(matches!(
            error,
            Error::PartialFrameWorkspaceCompletionUnobservable(_)
        ));
        assert!(!error.is_device_lost());
    });
    assert_eq!(operations, [0; 4]);
}

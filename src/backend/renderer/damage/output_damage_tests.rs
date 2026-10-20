use super::*;
use crate::backend::renderer::{
    element::solid::{SolidColorBuffer, SolidColorRenderElement},
    test::{DummyError, DummyFramebuffer, DummyRenderer, DummyTexture},
    ContextId, DebugFlags, RendererSuper, TextureFilter,
};
use crate::utils::Buffer;

#[derive(Debug, Default)]
struct RecordingRenderer {
    declared: Vec<Rectangle<i32, Physical>>,
    cleared: Vec<Rectangle<i32, Physical>>,
    draws: usize,
    refuse: bool,
}

struct RecordingFrame<'a>(&'a mut RecordingRenderer);

impl RendererSuper for RecordingRenderer {
    type Error = DummyError;
    type TextureId = DummyTexture;
    type Framebuffer<'a> = DummyFramebuffer;
    type Frame<'a, 'b>
        = RecordingFrame<'a>
    where
        'b: 'a,
        Self: 'a;
}

impl Renderer for RecordingRenderer {
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
    ) -> Result<RecordingFrame<'a>, Self::Error>
    where
        'b: 'a,
    {
        self.declared.clear();
        self.cleared.clear();
        self.draws = 0;
        Ok(RecordingFrame(self))
    }
    fn wait(&mut self, _: &SyncPoint) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl Frame for RecordingFrame<'_> {
    type Error = DummyError;
    type TextureId = DummyTexture;
    fn context_id(&self) -> ContextId<DummyTexture> {
        self.0.context_id()
    }
    fn set_output_damage(&mut self, damage: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        assert!(self.0.cleared.is_empty());
        assert_eq!(self.0.draws, 0);
        if self.0.refuse {
            return Err(DummyError::SyncInterrupted);
        }
        self.0.declared.extend_from_slice(damage);
        Ok(())
    }
    fn clear(&mut self, _: Color32F, damage: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        assert!(!self.0.declared.is_empty());
        self.0.cleared.extend_from_slice(damage);
        Ok(())
    }
    fn draw_solid(
        &mut self,
        _: Rectangle<i32, Physical>,
        _: &[Rectangle<i32, Physical>],
        _: Color32F,
    ) -> Result<(), Self::Error> {
        assert!(!self.0.declared.is_empty());
        self.0.draws += 1;
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
        unreachable!()
    }
    fn transformation(&self) -> Transform {
        Transform::Normal
    }
    fn wait(&mut self, _: &SyncPoint) -> Result<(), Self::Error> {
        Ok(())
    }
    fn finish(self) -> Result<SyncPoint, Self::Error> {
        Ok(SyncPoint::signaled())
    }
}

#[test]
fn terminal_damage_is_authoritative_before_opaque_subtraction() {
    let mut renderer = RecordingRenderer::default();
    let mut tracker = OutputDamageTracker::new((100, 80), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(1, 64, 4).unwrap();
    let mut buffer = SolidColorBuffer::new((20, 20), Color32F::BLACK);
    for (age, color) in [(0, Color32F::BLACK), (1, Color32F::new(1.0, 1.0, 1.0, 1.0))] {
        buffer.set_color(color);
        let element = SolidColorRenderElement::from_buffer(&buffer, (10, 10), 1.0, 1.0, Kind::Unspecified);
        let result = tracker
            .render_output(
                &mut renderer,
                &mut DummyFramebuffer,
                age,
                &[element],
                Color32F::BLACK,
            )
            .unwrap();
        assert_eq!(result.damage.unwrap(), renderer.declared.as_slice());
        assert!(renderer.declared.iter().any(|rect| rect.contains((15, 15))));
        assert!(renderer.cleared.iter().all(|rect| !rect.contains((15, 15))));
        assert_eq!(renderer.draws, 1);
    }
}

#[test]
fn output_damage_refusal_stops_before_clear_or_draw_and_resets_history() {
    let mut renderer = RecordingRenderer {
        refuse: true,
        ..Default::default()
    };
    let mut tracker = OutputDamageTracker::new((100, 80), 1.0, Transform::Normal);
    tracker.prepare_frame_storage(0, 64, 4).unwrap();
    let error = tracker
        .render_output::<SolidColorRenderElement, _>(
            &mut renderer,
            &mut DummyFramebuffer,
            0,
            &[],
            Color32F::BLACK,
        )
        .unwrap_err();
    assert!(matches!(error, Error::Rendering(DummyError::SyncInterrupted)));
    assert!(renderer.cleared.is_empty());
    assert_eq!(renderer.draws, 0);
    assert!(tracker.last_state.size.is_none());
}

use super::*;
use crate::backend::renderer::utils::CommitCounter;

#[derive(Debug)]
struct StreamSource {
    id: Id,
}
impl StreamSource {
    fn regions() -> impl Iterator<Item = Rectangle<i32, Physical>> {
        (0..96).map(|i| Rectangle::new(((i % 12) * 4, (i / 12) * 4).into(), (3, 3).into()))
    }
}
impl Element for StreamSource {
    fn id(&self) -> &Id {
        &self.id
    }
    fn current_commit(&self) -> CommitCounter {
        CommitCounter::from(1)
    }
    fn src(&self) -> Rectangle<f64, Buffer> {
        Rectangle::from_size((64.0, 64.0).into())
    }
    fn geometry(&self, _: Scale<f64>) -> Rectangle<i32, Physical> {
        Rectangle::from_size((64, 64).into())
    }
    fn damage_since(&self, _: Scale<f64>, _: Option<CommitCounter>) -> DamageSet<i32, Physical> {
        panic!("adapter collected legacy damage")
    }
    fn opaque_regions(&self, _: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        panic!("adapter collected legacy opaque proof")
    }
    fn visit_damage_since(
        &self,
        _: Scale<f64>,
        _: Option<CommitCounter>,
        visit: &mut dyn FnMut(Rectangle<i32, Physical>),
    ) {
        Self::regions().for_each(visit);
    }
    fn visit_opaque_regions(&self, _: Scale<f64>, visit: &mut dyn FnMut(Rectangle<i32, Physical>)) {
        Self::regions().for_each(visit);
    }
}

#[test]
fn adapter_chain_streams_every_rectangle_with_original_rounding() {
    let source = StreamSource { id: Id::new() };
    let resized = RescaleRenderElement::from_element(source, (0, 0).into(), 1.25);
    let placed = RelocateRenderElement::from_element(resized, (7, -3), Relocate::Relative);
    let crop = Rectangle::new((13, 2).into(), (42, 28).into());
    let element = CropRenderElement::from_element(placed, 1.0, crop).unwrap();
    let expected = |opaque: bool| {
        StreamSource::regions()
            .filter_map(|rect| {
                let scaled = if opaque {
                    rect.to_f64().upscale(1.25).to_i32_round()
                } else {
                    rect.to_f64().upscale(1.25).to_i32_up()
                };
                let local_crop = Rectangle::new((6, 5).into(), crop.size);
                scaled.intersection(local_crop).map(|mut rect| {
                    rect.loc -= local_crop.loc;
                    rect
                })
            })
            .collect::<Vec<_>>()
    };
    let mut actual = Vec::new();
    element.visit_damage_since(Scale::from(1.0), None, &mut |rect| actual.push(rect));
    assert_eq!(actual, expected(false));
    assert!(actual.len() > 32);
    actual.clear();
    element.visit_opaque_regions(Scale::from(1.0), &mut |rect| actual.push(rect));
    assert_eq!(actual, expected(true));
    assert!(actual.len() > 16);
}

#[cfg(feature = "renderer_vulkan")]
#[test]
fn repeated_borrowed_and_wrap_adapter_visits_have_no_heap_operations() {
    use crate::backend::renderer::{element::Wrap, vulkan::storage_heap_probe};
    let source = StreamSource { id: Id::new() };
    let wrapped = Wrap::from(&source);
    let (count, calls) = storage_heap_probe::measure(|| {
        let mut count = 0;
        for _ in 0..120 {
            wrapped.visit_damage_since(Scale::from(1.0), None, &mut |_| count += 1);
            wrapped.visit_opaque_regions(Scale::from(1.0), &mut |_| count += 1);
        }
        count
    });
    assert_eq!(count, 120 * 96 * 2);
    assert_eq!(calls, [0; 4]);
}

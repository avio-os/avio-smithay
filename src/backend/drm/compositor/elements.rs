use crate::{
    backend::renderer::{
        element::{Element, ElementSource, Id, RenderElement, WorkspaceVec},
        utils::{CommitCounter, DamageSet, OpaqueRegions},
        Color32F, Frame, Renderer,
    },
    render_elements,
    utils::{Buffer, Physical, Rectangle, Scale, Transform},
};

render_elements! {
    pub DrmRenderElements<'a, R, E>;
    Plane=BorrowedPlaneElement<'a, E>,
    Other=&'a E,
}

#[derive(Debug)]
pub(super) struct PlaneElementDescriptor {
    pub id: Id,
    pub commit: CommitCounter,
    pub geometry: Rectangle<i32, Physical>,
    pub source_index: usize,
    pub opaque_offset: crate::utils::Point<i32, Physical>,
    pub holepunch: bool,
}

pub struct BorrowedPlaneElement<'a, E> {
    pub(super) descriptor: &'a PlaneElementDescriptor,
    pub(super) source: &'a E,
    pub(super) scale: Scale<f64>,
}

impl<E: Element> Element for BorrowedPlaneElement<'_, E> {
    fn id(&self) -> &Id {
        &self.descriptor.id
    }
    fn current_commit(&self) -> CommitCounter {
        self.descriptor.commit
    }
    fn src(&self) -> Rectangle<f64, Buffer> {
        Rectangle::default()
    }
    fn geometry(&self, _scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.descriptor.geometry
    }
    fn transform(&self) -> Transform {
        Transform::Normal
    }
    fn damage_since(&self, _scale: Scale<f64>, commit: Option<CommitCounter>) -> DamageSet<i32, Physical> {
        if self.descriptor.commit.distance(commit) != Some(0) {
            DamageSet::from_slice(&[Rectangle::from_size(self.descriptor.geometry.size)])
        } else {
            DamageSet::default()
        }
    }
    fn opaque_regions(&self, _scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        if self.descriptor.holepunch {
            return OpaqueRegions::from_slice(&[Rectangle::from_size(self.descriptor.geometry.size)]);
        }
        self.source
            .opaque_regions(self.scale)
            .into_iter()
            .map(|mut region| {
                region.loc -= self.descriptor.opaque_offset;
                region
            })
            .collect()
    }
    fn visit_opaque_regions(&self, _scale: Scale<f64>, visit: &mut dyn FnMut(Rectangle<i32, Physical>)) {
        if self.descriptor.holepunch {
            visit(Rectangle::from_size(self.descriptor.geometry.size));
        } else {
            self.source.visit_opaque_regions(self.scale, &mut |mut region| {
                region.loc -= self.descriptor.opaque_offset;
                visit(region);
            });
        }
    }
}

impl<R: Renderer, E: Element> RenderElement<R> for BorrowedPlaneElement<'_, E> {
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque: &[Rectangle<i32, Physical>],
    ) -> Result<(), R::Error> {
        if self.descriptor.holepunch {
            // clear already visits independent damage rectangles; retaining
            // one stack rectangle avoids a relocated heap vector.
            for rect in damage {
                let mut rect = *rect;
                rect.loc += dst.loc;
                frame.clear(Color32F::TRANSPARENT, &[rect])?;
            }
        }
        Ok(())
    }
}

pub(super) struct PrimaryElementSource<'a, R, E> {
    pub elements: &'a [E],
    pub primary: &'a WorkspaceVec<usize>,
    pub fake: &'a WorkspaceVec<PlaneElementDescriptor>,
    pub scale: Scale<f64>,
    pub disabled: bool,
    pub renderer: std::marker::PhantomData<R>,
}

impl<R: Renderer, E: RenderElement<R>> ElementSource for PrimaryElementSource<'_, R, E>
where
    R::TextureId: 'static,
{
    type Element<'a>
        = DrmRenderElements<'a, R, E>
    where
        Self: 'a;
    fn len(&self) -> usize {
        if self.disabled {
            0
        } else {
            self.fake.len() + self.primary.len()
        }
    }
    fn element(&self, index: usize) -> Self::Element<'_> {
        if index < self.fake.len() {
            let descriptor = &self.fake[index];
            DrmRenderElements::Plane(BorrowedPlaneElement {
                descriptor,
                source: &self.elements[descriptor.source_index],
                scale: self.scale,
            })
        } else {
            DrmRenderElements::Other(&self.elements[self.primary[index - self.fake.len()]])
        }
    }
}

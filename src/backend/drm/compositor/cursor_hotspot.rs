//! Frame-local cursor metadata; a presented cursor keeps its own hotspot.
use crate::backend::renderer::{element::Id, utils::CommitCounter};
use crate::utils::{Physical, Point, Size, Transform};

/// The hotspot of one exact cursor element in one frame.
///
/// Coordinates are pixels relative to the untransformed element's top-left,
/// at the output scale used to build that element. The compositor transforms
/// these coordinates with the same padded bitmap transform used for scanout.
/// A different element or commit cannot borrow this image's hotspot.
#[derive(Debug, Clone)]
pub struct CursorPlaneHotspot {
    element_id: Id,
    element_commit: CommitCounter,
    hotspot: Point<i32, Physical>,
}

impl CursorPlaneHotspot {
    /// Bind an image's hotspot to its actual element and content generation.
    pub fn new(element_id: Id, element_commit: CommitCounter, hotspot: Point<i32, Physical>) -> Self {
        Self {
            element_id,
            element_commit,
            hotspot,
        }
    }

    pub(super) fn for_bitmap(
        &self,
        id: &Id,
        commit: CommitCounter,
        transform: Transform,
        padded_bitmap_size: Size<i32, Physical>,
    ) -> Option<Point<i32, Physical>> {
        (self.element_id == *id && self.element_commit == commit)
            .then(|| transform.transform_point_in(self.hotspot, &padded_bitmap_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_bitmap_or_commit_cannot_borrow_a_hotspot() {
        let id = Id::new();
        let hotspot = CursorPlaneHotspot::new(id.clone(), 4usize.into(), (7, 12).into());
        let size = (64, 64).into();
        assert_eq!(
            hotspot.for_bitmap(&id, 4usize.into(), Transform::Normal, size),
            Some((7, 12).into())
        );
        assert_eq!(
            hotspot.for_bitmap(&Id::new(), 4usize.into(), Transform::Normal, size),
            None
        );
        assert_eq!(
            hotspot.for_bitmap(&id, 5usize.into(), Transform::Normal, size),
            None
        );
    }

    #[test]
    fn cursor_anchor_uses_the_transformed_padded_bitmap() {
        let id = Id::new();
        let hotspot = CursorPlaneHotspot::new(id.clone(), 4usize.into(), (7, 12).into());
        let size = (64, 64).into();
        for (transform, expected) in [
            (Transform::Normal, (7, 12)),
            (Transform::_90, (52, 7)),
            (Transform::_180, (57, 52)),
            (Transform::_270, (12, 57)),
            (Transform::Flipped, (57, 12)),
            (Transform::Flipped90, (12, 7)),
            (Transform::Flipped180, (7, 52)),
            (Transform::Flipped270, (52, 57)),
        ] {
            assert_eq!(
                hotspot.for_bitmap(&id, 4usize.into(), transform, size),
                Some(expected.into())
            );
        }
    }

    #[test]
    fn hotspot_coordinates_remain_signed() {
        let id = Id::new();
        let hotspot = CursorPlaneHotspot::new(id.clone(), 0usize.into(), (-3, 70).into());
        assert_eq!(
            hotspot.for_bitmap(&id, 0usize.into(), Transform::Normal, (64, 64).into()),
            Some((-3, 70).into())
        );
    }
}

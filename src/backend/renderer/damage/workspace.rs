//! Rectangle operations over owner-admitted storage; no reference erasure.
use crate::{
    backend::renderer::element::{FrameWorkspaceError, WorkspaceVec},
    utils::{Physical, Rectangle},
};

pub(super) fn extend<T>(
    out: &mut Vec<T>,
    values: impl IntoIterator<Item = T>,
    limit: Option<usize>,
) -> Result<(), FrameWorkspaceError> {
    for value in values {
        if let Some(capacity) = limit {
            if out.len() >= capacity {
                return Err(FrameWorkspaceError {
                    resource: "damage rectangles",
                    required: out.len().saturating_add(1),
                    capacity,
                });
            }
        }
        out.push(value);
    }
    Ok(())
}

/// Preserve the first bounded sink refusal while a borrowed source finishes
/// its visit. Nothing after refusal is appended and the caller propagates
/// failure before drawing/submitting the partial numeric workspace.
pub(crate) fn try_visit<T>(
    produce: impl FnOnce(&mut dyn FnMut(T)),
    mut consume: impl FnMut(T) -> Result<(), FrameWorkspaceError>,
) -> Result<(), FrameWorkspaceError> {
    let mut error = None;
    produce(&mut |value| {
        if error.is_none() {
            error = consume(value).err();
        }
    });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(crate) trait RectangleStorage: std::ops::DerefMut<Target = [Rectangle<i32, Physical>]> {
    fn swap_remove(&mut self, index: usize) -> Rectangle<i32, Physical>;
    fn append(&mut self, rectangle: Rectangle<i32, Physical>) -> Result<(), FrameWorkspaceError>;
    fn declared_capacity(&self) -> Option<usize>;
}
impl RectangleStorage for Vec<Rectangle<i32, Physical>> {
    fn swap_remove(&mut self, index: usize) -> Rectangle<i32, Physical> {
        Vec::swap_remove(self, index)
    }
    fn append(&mut self, rectangle: Rectangle<i32, Physical>) -> Result<(), FrameWorkspaceError> {
        self.push(rectangle);
        Ok(())
    }
    fn declared_capacity(&self) -> Option<usize> {
        None
    }
}
impl RectangleStorage for WorkspaceVec<Rectangle<i32, Physical>> {
    fn swap_remove(&mut self, index: usize) -> Rectangle<i32, Physical> {
        self.swap_remove(index)
    }
    fn append(&mut self, rectangle: Rectangle<i32, Physical>) -> Result<(), FrameWorkspaceError> {
        self.push(rectangle)
    }
    fn declared_capacity(&self) -> Option<usize> {
        self.admitted_capacity()
    }
}

pub(crate) fn subtract<S: RectangleStorage>(
    rects: &mut S,
    others: impl IntoIterator<Item = Rectangle<i32, Physical>>,
    limit: Option<usize>,
) -> Result<(), FrameWorkspaceError> {
    let limit = match (limit, rects.declared_capacity()) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    for other in others {
        // Appended split pieces are already outside this cutter. Keep every
        // not-yet-tested original in a shrinking prefix; swap_remove alone
        // could otherwise pull a new split into that prefix and skip an
        // original rectangle when several originals intersect the cutter.
        let mut original_end = rects.len();
        let mut index = 0;
        while index < original_end {
            let item = rects[index];
            let Some(intersection) = item.intersection(other) else {
                index += 1;
                continue;
            };
            let pieces = [
                Rectangle::new(
                    item.loc,
                    (item.size.w, intersection.loc.y.saturating_sub(item.loc.y)).into(),
                ),
                Rectangle::new(
                    (item.loc.x, intersection.loc.y).into(),
                    (intersection.loc.x.saturating_sub(item.loc.x), intersection.size.h).into(),
                ),
                Rectangle::new(
                    (
                        intersection.loc.x.saturating_add(intersection.size.w),
                        intersection.loc.y,
                    )
                        .into(),
                    (
                        item.loc
                            .x
                            .saturating_add(item.size.w)
                            .saturating_sub(intersection.loc.x.saturating_add(intersection.size.w)),
                        intersection.size.h,
                    )
                        .into(),
                ),
                Rectangle::new(
                    (item.loc.x, intersection.loc.y.saturating_add(intersection.size.h)).into(),
                    (
                        item.size.w,
                        item.loc
                            .y
                            .saturating_add(item.size.h)
                            .saturating_sub(intersection.loc.y.saturating_add(intersection.size.h)),
                    )
                        .into(),
                ),
            ];
            let required = rects.len() - 1 + pieces.iter().filter(|piece| !piece.is_empty()).count();
            if let Some(capacity) = limit {
                if required > capacity {
                    return Err(FrameWorkspaceError {
                        resource: "rectangle subtraction",
                        required,
                        capacity,
                    });
                }
            }
            original_end -= 1;
            rects.swap(index, original_end);
            rects.swap_remove(original_end);
            if other.contains_rect(item) {
                continue;
            }
            for piece in pieces.into_iter().filter(|piece| !piece.is_empty()) {
                rects.append(piece)?;
            }
        }
    }
    Ok(())
}

pub(super) fn reserve<T>(vec: &mut Vec<T>, capacity: usize) {
    vec.reserve(capacity.saturating_sub(vec.len()));
}

/// Exact uncovered integer-pixel area without constructing a rectangle
/// arrangement. Y bands change only at cutter edges; X intervals are unioned
/// by endpoint scans. The conservative drawing policy therefore does not
/// falsely report fully occluded contributors as visibly covering pixels.
pub(crate) fn visible_area(rect: Rectangle<i32, Physical>, opaque: &[Rectangle<i32, Physical>]) -> usize {
    let left = i64::from(rect.loc.x);
    let right = left + i64::from(rect.size.w.max(0));
    let bottom = i64::from(rect.loc.y) + i64::from(rect.size.h.max(0));
    let mut y = i64::from(rect.loc.y);
    let mut area = 0u128;
    while y < bottom {
        let next_y = opaque
            .iter()
            .flat_map(|cut| {
                [
                    i64::from(cut.loc.y),
                    i64::from(cut.loc.y) + i64::from(cut.size.h.max(0)),
                ]
            })
            .filter(|edge| *edge > y)
            .min()
            .unwrap_or(bottom)
            .min(bottom);
        let active = |cut: &&Rectangle<i32, Physical>| {
            i64::from(cut.loc.y) <= y && i64::from(cut.loc.y) + i64::from(cut.size.h.max(0)) > y
        };
        let mut x = left;
        let mut visible_width = 0i64;
        while x < right {
            let covered_end = opaque
                .iter()
                .filter(active)
                .filter(|cut| i64::from(cut.loc.x) <= x)
                .map(|cut| i64::from(cut.loc.x) + i64::from(cut.size.w.max(0)))
                .filter(|end| *end > x)
                .max();
            if let Some(end) = covered_end {
                x = end.min(right);
            } else {
                let next_x = opaque
                    .iter()
                    .filter(active)
                    .map(|cut| i64::from(cut.loc.x))
                    .filter(|start| *start > x)
                    .min()
                    .unwrap_or(right)
                    .min(right);
                visible_width += next_x - x;
                x = next_x;
            }
        }
        area += (visible_width as u128) * ((next_y - y) as u128);
        y = next_y;
    }
    area.min(usize::MAX as u128) as usize
}

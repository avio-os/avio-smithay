//! Utility module for helpers around drawing [`WlSurface`](wayland_server::protocol::wl_surface::WlSurface)s
//! and [`RenderElement`](super::element::RenderElement)s with [`Renderer`](super::Renderer)s.

use crate::utils::{
    Buffer as BufferCoord, Coordinate, Logical, Physical, Point, Rectangle, Scale, Size, Transform,
};
use std::{collections::VecDeque, fmt, sync::Arc};

#[cfg(feature = "wayland_frontend")]
mod wayland;
#[cfg(feature = "wayland_frontend")]
pub use self::wayland::*;

/// A simple wrapper for counting commits
///
/// The purpose of the counter is to keep track
/// on the number of times something has changed.
/// It provides an easy way to obtain the distance
/// between two instances of a [`CommitCounter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct CommitCounter(usize);

impl CommitCounter {
    /// Increment the commit counter
    pub fn increment(&mut self) {
        self.0 = self.0.wrapping_add(1)
    }

    /// Get the distance between two [`CommitCounter`]s
    ///
    /// If the [`CommitCounter`] is incremented on each recorded
    /// damage this returns the count of damage that happened
    /// between the [`CommitCounter`]s
    ///
    /// Returns `None` in case the distance could not be calculated.
    /// If uses as part of damage tracking the tracked element
    /// should be considered as fully damaged.
    pub fn distance(&self, previous_commit: Option<CommitCounter>) -> Option<usize> {
        // if commit > commit_count we have overflown, in that case the following map might result
        // in a false-positive, if commit is still very large. So we force false in those cases.
        // That will result in a potentially sub-optimal full damage every usize::MAX frames,
        // which is acceptable.
        previous_commit
            .filter(|commit| commit <= self)
            .map(|commit| self.0.wrapping_sub(commit.0))
    }
}

impl From<usize> for CommitCounter {
    #[inline]
    fn from(counter: usize) -> Self {
        CommitCounter(counter)
    }
}

/// A tracker for holding damage
///
/// It keeps track of the submitted damage
/// and automatically caps the damage
/// with the specified limit.
///
/// See [`DamageSnapshot`] for more
/// information.
pub struct DamageBag<N, Kind> {
    limit: usize,
    state: DamageSnapshot<N, Kind>,
}

/// A snapshot of the current state of a [`DamageBag`]
///
/// The snapshot can be used to get an immutable view
/// into the current state of a [`DamageBag`].
/// It provides an easy way to get the damage between two
/// [`CommitCounter`]s.
pub struct DamageSnapshot<N, Kind> {
    limit: usize,
    commit_counter: CommitCounter,
    damage: Arc<VecDeque<smallvec::SmallVec<[Rectangle<N, Kind>; MAX_DAMAGE_RECTS]>>>,
}

impl<N, Kind> Clone for DamageSnapshot<N, Kind> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            limit: self.limit,
            commit_counter: self.commit_counter,
            damage: self.damage.clone(),
        }
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageBag<N, BufferCoord> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageBag")
            .field("limit", &self.limit)
            .field("state", &self.state)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageBag<N, Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageBag")
            .field("limit", &self.limit)
            .field("state", &self.state)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageBag<N, Logical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageBag")
            .field("limit", &self.limit)
            .field("state", &self.state)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSnapshot<N, BufferCoord> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSnapshot")
            .field("commit_counter", &self.commit_counter)
            .field("damage", &self.damage)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSnapshot<N, Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSnapshot")
            .field("commit_counter", &self.commit_counter)
            .field("damage", &self.damage)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSnapshot<N, Logical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSnapshot")
            .field("commit_counter", &self.commit_counter)
            .field("damage", &self.damage)
            .finish()
    }
}

const MAX_DAMAGE_AGE: usize = 4;
const MAX_DAMAGE_RECTS: usize = 16;
const MAX_DAMAGE_SET: usize = MAX_DAMAGE_RECTS * 2;

impl<N: Clone, Kind> Default for DamageBag<N, Kind> {
    #[inline]
    fn default() -> Self {
        DamageBag::new(MAX_DAMAGE_AGE)
    }
}

impl<N: Clone, Kind> DamageSnapshot<N, Kind> {
    fn new(limit: usize) -> Self {
        DamageSnapshot {
            limit,
            commit_counter: CommitCounter::default(),
            damage: Arc::new(VecDeque::with_capacity(limit)),
        }
    }

    /// Create an empty damage snapshot
    pub fn empty() -> Self {
        DamageSnapshot {
            limit: 0,
            commit_counter: CommitCounter::default(),
            damage: Default::default(),
        }
    }

    /// Gets the current [`CommitCounter`] of this snapshot
    ///
    /// The returned [`CommitCounter`] should be stored after
    /// calling [`damage_since`](DamageSnapshot::damage_since)
    /// and provided to the next call of [`damage_since`](DamageSnapshot::damage_since)
    /// to query the damage between these two [`CommitCounter`]s.
    #[inline]
    pub fn current_commit(&self) -> CommitCounter {
        self.commit_counter
    }

    /// Provides raw access to the stored damage
    pub fn raw(&self) -> impl Iterator<Item = impl Iterator<Item = &Rectangle<N, Kind>>> {
        self.damage.iter().map(|d| d.iter())
    }

    fn reset(&mut self) {
        Arc::make_mut(&mut self.damage).clear();
        self.commit_counter.increment();
    }
}

impl<N: Coordinate, Kind> DamageSnapshot<N, Kind> {
    /// Get the damage since the last commit
    ///
    /// Returns `None` in case the [`CommitCounter`] is too old
    /// or the damage has been reset. In that case the whole
    /// element geometry should be considered as damaged
    ///
    /// If the commit is recent enough and no damage has occurred
    /// an empty `Vec` will be returned
    pub fn damage_since(&self, commit: Option<CommitCounter>) -> Option<DamageSet<N, Kind>> {
        let distance = self.commit_counter.distance(commit);

        if distance
            .map(|distance| distance <= self.damage.len())
            .unwrap_or(false)
        {
            let mut damage_set = DamageSet::default();
            for damage in self.damage.iter().take(distance.unwrap()) {
                damage_set.damage.extend_from_slice(damage);
            }
            Some(damage_set)
        } else {
            None
        }
    }

    /// Visit the exact retained damage range without collecting an owning set.
    /// False means the predecessor is unavailable and visits no rectangles;
    /// callers then use full damage, as with `damage_since`.
    pub fn visit_damage_since(
        &self,
        commit: Option<CommitCounter>,
        visit: &mut dyn FnMut(Rectangle<N, Kind>),
    ) -> bool {
        let Some(distance) = self.commit_counter.distance(commit) else {
            return false;
        };
        if distance > self.damage.len() {
            return false;
        }
        for damage in self.damage.iter().take(distance) {
            for rect in damage {
                visit(*rect);
            }
        }
        true
    }

    fn add(&mut self, damage: impl IntoIterator<Item = Rectangle<N, Kind>>) {
        // FIXME: Get rid of this allocation here
        let mut damage = damage.into_iter().filter(|d| !d.is_empty()).collect::<Vec<_>>();

        if damage.is_empty() {
            // do not track empty damage
            return;
        }

        damage.dedup();

        let inner_damage = Arc::make_mut(&mut self.damage);
        inner_damage.push_front(smallvec::SmallVec::from_vec(damage));
        inner_damage.truncate(self.limit);

        self.commit_counter.increment();
    }
}

impl<N: Clone, Kind> DamageBag<N, Kind> {
    /// Initialize a a new [`DamageBag`] with the specified limit
    pub fn new(limit: usize) -> Self {
        DamageBag {
            limit,
            state: DamageSnapshot::new(limit),
        }
    }

    /// Gets the current [`CommitCounter`] of this tracker
    #[inline]
    pub fn current_commit(&self) -> CommitCounter {
        self.state.current_commit()
    }

    /// Provides raw access to the stored damage
    pub fn raw(&self) -> impl Iterator<Item = impl Iterator<Item = &Rectangle<N, Kind>>> {
        self.state.raw()
    }

    /// Reset the damage
    ///
    /// This should be called when the
    /// tracked item has been resized
    pub fn reset(&mut self) {
        self.state.reset()
    }
}

impl<N, Kind> DamageBag<N, Kind> {
    /// Get a snapshot of the current damage
    pub fn snapshot(&self) -> DamageSnapshot<N, Kind> {
        self.state.clone()
    }
}

impl<N: Coordinate, Kind> DamageBag<N, Kind> {
    /// Add some damage to the tracker
    pub fn add(&mut self, damage: impl IntoIterator<Item = Rectangle<N, Kind>>) {
        self.state.add(damage)
    }

    /// Get the damage since the last commit
    ///
    /// Returns `None` in case the [`CommitCounter`] is too old
    /// or the damage has been reset. In that case the whole
    /// element geometry should be considered as damaged
    ///
    /// If the commit is recent enough and no damage has occurred
    /// an empty `Vec` will be returned
    pub fn damage_since(&self, commit: Option<CommitCounter>) -> Option<DamageSet<N, Kind>> {
        self.state.damage_since(commit)
    }
}

/// A set of damage returned from [`DamageBag::damage_since`] of [`DamageSnapshot::damage_since`]
pub struct DamageSet<N, Kind> {
    damage: smallvec::SmallVec<[Rectangle<N, Kind>; MAX_DAMAGE_SET]>,
}

impl<N, Kind> Default for DamageSet<N, Kind> {
    fn default() -> Self {
        Self {
            damage: Default::default(),
        }
    }
}

impl<N: Copy, Kind> DamageSet<N, Kind> {
    /// Copy the damage from a slice into a new `DamageSet`.
    #[inline]
    pub fn from_slice(slice: &[Rectangle<N, Kind>]) -> Self {
        Self {
            damage: smallvec::SmallVec::from_slice(slice),
        }
    }
}

impl<N, Kind> std::ops::Deref for DamageSet<N, Kind> {
    type Target = [Rectangle<N, Kind>];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.damage
    }
}

impl<N, Kind> IntoIterator for DamageSet<N, Kind> {
    type Item = Rectangle<N, Kind>;

    type IntoIter = DamageSetIter<N, Kind>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        DamageSetIter {
            inner: self.damage.into_iter(),
        }
    }
}

impl<N, Kind> FromIterator<Rectangle<N, Kind>> for DamageSet<N, Kind> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = Rectangle<N, Kind>>>(iter: T) -> Self {
        Self {
            damage: smallvec::SmallVec::from_iter(iter),
        }
    }
}

/// Iterator for [`DamageSet::into_iter`]
pub struct DamageSetIter<N, Kind> {
    inner: smallvec::IntoIter<[Rectangle<N, Kind>; MAX_DAMAGE_SET]>,
}

impl<N: fmt::Debug> fmt::Debug for DamageSetIter<N, BufferCoord> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSetIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSetIter<N, Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSetIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSetIter<N, Logical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSetIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<N, Kind> Iterator for DamageSetIter<N, Kind> {
    type Item = Rectangle<N, Kind>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSet<N, BufferCoord> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSet").field("damage", &self.damage).finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSet<N, Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSet").field("damage", &self.damage).finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for DamageSet<N, Logical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DamageSet").field("damage", &self.damage).finish()
    }
}

const MAX_OPAQUE_REGIONS: usize = 16;

/// Wrapper for a set of opaque regions
pub struct OpaqueRegions<N, Kind> {
    regions: smallvec::SmallVec<[Rectangle<N, Kind>; MAX_OPAQUE_REGIONS]>,
}

impl<N, Kind> Default for OpaqueRegions<N, Kind>
where
    N: Default,
{
    #[inline]
    fn default() -> Self {
        Self {
            regions: Default::default(),
        }
    }
}

impl<N: Copy, Kind> OpaqueRegions<N, Kind> {
    /// Copy the opaque regions from a slice into a new `OpaqueRegions`.
    #[inline]
    pub fn from_slice(slice: &[Rectangle<N, Kind>]) -> Self {
        Self {
            regions: smallvec::SmallVec::from_slice(slice),
        }
    }
}

impl<N, Kind> std::ops::Deref for OpaqueRegions<N, Kind> {
    type Target = [Rectangle<N, Kind>];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.regions
    }
}

impl<N, Kind> IntoIterator for OpaqueRegions<N, Kind> {
    type Item = Rectangle<N, Kind>;

    type IntoIter = OpaqueRegionsIter<N, Kind>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        OpaqueRegionsIter {
            inner: self.regions.into_iter(),
        }
    }
}

impl<N, Kind> FromIterator<Rectangle<N, Kind>> for OpaqueRegions<N, Kind> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = Rectangle<N, Kind>>>(iter: T) -> Self {
        Self {
            regions: smallvec::SmallVec::from_iter(iter),
        }
    }
}

/// Iterator for [`OpaqueRegions::into_iter`]
pub struct OpaqueRegionsIter<N, Kind> {
    inner: smallvec::IntoIter<[Rectangle<N, Kind>; MAX_OPAQUE_REGIONS]>,
}

impl<N: fmt::Debug> fmt::Debug for OpaqueRegionsIter<N, BufferCoord> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueRegionsIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for OpaqueRegionsIter<N, Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueRegionsIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for OpaqueRegionsIter<N, Logical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueRegionsIter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<N, Kind> Iterator for OpaqueRegionsIter<N, Kind> {
    type Item = Rectangle<N, Kind>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<N: fmt::Debug> fmt::Debug for OpaqueRegions<N, BufferCoord> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueRegions")
            .field("regions", &self.regions)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for OpaqueRegions<N, Physical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueRegions")
            .field("regions", &self.regions)
            .finish()
    }
}

impl<N: fmt::Debug> fmt::Debug for OpaqueRegions<N, Logical> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueRegions")
            .field("regions", &self.regions)
            .finish()
    }
}

/// Defines a view into the surface
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct SurfaceView {
    /// The logical source used for cropping
    pub src: Rectangle<f64, Logical>,
    /// The logical destination size used for scaling
    pub dst: Size<i32, Logical>,
    /// The logical offset for a sub-surface
    pub offset: Point<i32, Logical>,
}

impl SurfaceView {
    /// Scale from source-local logical coordinates to destination logical coordinates.
    pub fn scale(&self) -> Scale<f64> {
        Scale::from((
            self.dst.w as f64 / self.src.size.w,
            self.dst.h as f64 / self.src.size.h,
        ))
    }

    /// Convert a rectangle from source-local logical coordinates to destination-local logical coordinates.
    pub fn rect_to_global<N>(&self, rect: Rectangle<N, Logical>) -> Rectangle<f64, Logical>
    where
        N: Coordinate,
    {
        let scale = self.scale();
        let mut rect = rect.to_f64();
        rect.loc -= self.src.loc;
        rect.upscale(scale)
    }

    /// Convert a rectangle from destination-local logical coordinates to source-local logical coordinates.
    pub fn rect_to_local<N>(&self, rect: Rectangle<N, Logical>) -> Rectangle<f64, Logical>
    where
        N: Coordinate,
    {
        let scale = self.scale();
        let mut rect = rect.to_f64().downscale(scale);
        rect.loc += self.src.loc;
        rect
    }

    /// Convert `wl_surface.damage` surface-local damage into buffer coordinates.
    pub fn surface_damage_to_buffer(
        &self,
        rect: Rectangle<i32, Logical>,
        buffer_scale: i32,
        buffer_transform: Transform,
        surface_size: &Size<i32, Logical>,
    ) -> Rectangle<i32, BufferCoord> {
        if self.src.size.w <= 0.0 || self.src.size.h <= 0.0 || self.dst.w <= 0 || self.dst.h <= 0 {
            return Rectangle::from_size(*surface_size).to_buffer(
                buffer_scale,
                buffer_transform,
                surface_size,
            );
        }

        self.rect_to_local(rect)
            .to_i32_up()
            .to_buffer(buffer_scale, buffer_transform, surface_size)
    }

    /// Project buffer-coordinate damage into element-local physical damage.
    ///
    /// This is the shared primitive used by Wayland surface elements and
    /// retained snapshot elements: exact buffer damage is clipped through the
    /// current surface view, scaled to the destination, then rounded into the
    /// element's physical coordinate space. `None` means the damage does not
    /// intersect this view.
    pub fn buffer_damage_to_element(
        &self,
        rect: Rectangle<i32, BufferCoord>,
        buffer_dimensions: Size<i32, BufferCoord>,
        buffer_scale: i32,
        buffer_transform: Transform,
        element_physical_size: Size<i32, Physical>,
        output_scale: Scale<f64>,
    ) -> Option<Rectangle<i32, Physical>> {
        if self.src.size.w <= 0.0 || self.src.size.h <= 0.0 || self.dst.w <= 0 || self.dst.h <= 0 {
            return Some(Rectangle::from_size(element_physical_size));
        }

        let ideal_dst_size = self.dst.to_f64().to_physical(output_scale);
        if ideal_dst_size.w <= 0.0 || ideal_dst_size.h <= 0.0 {
            return Some(Rectangle::from_size(element_physical_size));
        }

        rect.to_f64()
            .to_logical(buffer_scale as f64, buffer_transform, &buffer_dimensions.to_f64())
            .intersection(self.src)
            .map(|rect| self.rect_to_global(rect).to_i32_up::<i32>())
            .map(|rect| {
                let rounding_scale = element_physical_size.to_f64() / ideal_dst_size;
                rect.to_physical_precise_up(rounding_scale * output_scale)
            })
            .filter(|rect| !rect.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Transform;

    #[test]
    fn borrowed_damage_range_matches_large_legacy_set_and_retains_its_generation() {
        let mut bag = DamageBag::<i32, Physical>::new(4);
        let previous = bag.current_commit();
        for row in 0..3 {
            bag.add((0..40).map(|column| Rectangle::new((column * 2, row * 3).into(), (1, 2).into())));
        }
        let retained = bag.snapshot();
        let expected: Vec<_> = retained
            .damage_since(Some(previous))
            .unwrap()
            .into_iter()
            .collect();
        let mut actual = Vec::new();
        assert!(retained.visit_damage_since(Some(previous), &mut |rect| actual.push(rect)));
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 120);
        for _ in 0..5 {
            bag.add([Rectangle::from_size((1, 1).into())]);
        }
        let mut called = false;
        assert!(!bag
            .snapshot()
            .visit_damage_since(Some(previous), &mut |_| called = true));
        assert!(!called);
        actual.clear();
        assert!(retained.visit_damage_since(Some(previous), &mut |rect| actual.push(rect)));
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "renderer_vulkan")]
    #[test]
    fn repeated_large_damage_visits_allocate_nothing() {
        let mut bag = DamageBag::<i32, Physical>::new(4);
        let previous = bag.current_commit();
        bag.add((0..96).map(|index| Rectangle::new((index, 0).into(), (1, 1).into())));
        let snapshot = bag.snapshot();
        let (count, calls) = crate::backend::renderer::vulkan::storage_heap_probe::measure(|| {
            let mut count = 0;
            for _ in 0..120 {
                assert!(snapshot.visit_damage_since(Some(previous), &mut |_| count += 1));
            }
            count
        });
        assert_eq!(count, 96 * 120);
        assert_eq!(calls, [0; 4]);
    }

    #[test]
    fn surface_view_projects_buffer_damage_to_element_identity() {
        let view = SurfaceView {
            src: Rectangle::<f64, Logical>::new((0.0, 0.0).into(), (640.0, 480.0).into()),
            dst: Size::<i32, Logical>::from((640, 480)),
            offset: Default::default(),
        };

        let damage = view
            .buffer_damage_to_element(
                Rectangle::<i32, BufferCoord>::new((0, 160).into(), (640, 80).into()),
                Size::<i32, BufferCoord>::from((640, 480)),
                1,
                Transform::Normal,
                Size::<i32, Physical>::from((640, 480)),
                Scale::from(1.0),
            )
            .expect("damage intersects view");

        assert_eq!(
            damage,
            Rectangle::<i32, Physical>::new((0, 160).into(), (640, 80).into())
        );
    }

    #[test]
    fn surface_view_projects_buffer_damage_through_viewport_scale() {
        let view = SurfaceView {
            src: Rectangle::<f64, Logical>::new((0.0, 100.0).into(), (640.0, 200.0).into()),
            dst: Size::<i32, Logical>::from((640, 100)),
            offset: Default::default(),
        };

        let damage = view
            .buffer_damage_to_element(
                Rectangle::<i32, BufferCoord>::new((0, 150).into(), (640, 50).into()),
                Size::<i32, BufferCoord>::from((640, 480)),
                1,
                Transform::Normal,
                Size::<i32, Physical>::from((640, 100)),
                Scale::from(1.0),
            )
            .expect("damage intersects view");

        assert_eq!(
            damage,
            Rectangle::<i32, Physical>::new((0, 25).into(), (640, 25).into())
        );
    }

    #[test]
    fn surface_view_converts_surface_damage_to_buffer_damage() {
        let view = SurfaceView {
            src: Rectangle::<f64, Logical>::new((0.0, 100.0).into(), (640.0, 200.0).into()),
            dst: Size::<i32, Logical>::from((640, 100)),
            offset: Default::default(),
        };

        let damage = view.surface_damage_to_buffer(
            Rectangle::<i32, Logical>::new((0, 25).into(), (640, 25).into()),
            1,
            Transform::Normal,
            &Size::<i32, Logical>::from((640, 480)),
        );

        assert_eq!(
            damage,
            Rectangle::<i32, BufferCoord>::new((0, 150).into(), (640, 50).into())
        );
    }

    #[test]
    fn surface_view_invalid_projection_falls_back_to_full_element() {
        let view = SurfaceView {
            src: Rectangle::<f64, Logical>::new((0.0, 0.0).into(), (0.0, 480.0).into()),
            dst: Size::<i32, Logical>::from((640, 480)),
            offset: Default::default(),
        };

        let damage = view
            .buffer_damage_to_element(
                Rectangle::<i32, BufferCoord>::new((10, 10).into(), (20, 20).into()),
                Size::<i32, BufferCoord>::from((640, 480)),
                1,
                Transform::Normal,
                Size::<i32, Physical>::from((640, 480)),
                Scale::from(1.0),
            )
            .expect("invalid projection is conservative full damage");

        assert_eq!(
            damage,
            Rectangle::<i32, Physical>::new((0, 0).into(), (640, 480).into())
        );
    }
}

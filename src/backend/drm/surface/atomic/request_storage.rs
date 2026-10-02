//! Cold-admitted native atomic ioctl arrays, reused without transferring backing storage.
use crate::backend::drm::error::Error;
use drm::control::plane;
use std::{io, os::unix::io::BorrowedFd};

#[derive(Debug)]
pub(super) struct ObjectRoster {
    object: u32,
    properties: Vec<u32>,
}
impl ObjectRoster {
    pub(super) fn cold(object: u32, properties: impl IntoIterator<Item = u32>) -> Self {
        let mut properties: Vec<_> = properties.into_iter().collect();
        properties.sort_unstable();
        properties.dedup();
        Self { object, properties }
    }
}

#[derive(Debug)]
pub(in crate::backend::drm::surface) struct AtomicRequestStorage {
    roster: Vec<ObjectRoster>,
    objects: Vec<u32>,
    counts: Vec<u32>,
    properties: Vec<u32>,
    values: Vec<u64>,
    plane_edits: Vec<(plane::Handle, bool)>,
    object_limit: usize,
    property_limit: usize,
    plane_limit: usize,
}
impl AtomicRequestStorage {
    pub(super) fn cold(mut roster: Vec<ObjectRoster>, planes: usize) -> Self {
        roster.sort_unstable_by_key(|object| object.object);
        roster.dedup_by_key(|object| object.object);
        let object_limit = roster.len();
        let property_limit = roster.iter().map(|object| object.properties.len()).sum();
        Self {
            roster,
            objects: Vec::with_capacity(object_limit),
            counts: Vec::with_capacity(object_limit),
            properties: Vec::with_capacity(property_limit),
            values: Vec::with_capacity(property_limit),
            plane_edits: Vec::with_capacity(planes),
            object_limit,
            property_limit,
            plane_limit: planes,
        }
    }
    pub(in crate::backend::drm::surface) fn plane_capacity(&self) -> usize {
        self.plane_limit
    }
    pub(in crate::backend::drm::surface) fn connector_capacity(&self) -> usize {
        self.object_limit
    }
    pub(super) fn clear(&mut self) {
        self.objects.clear();
        self.counts.clear();
        self.properties.clear();
        self.values.clear();
        self.plane_edits.clear();
    }
    /// Connector hotplug is a cold admission operation. Replace its declared
    /// properties here, retaining the surface's exact compatible plane roster.
    pub(super) fn admit_objects_cold(&mut self, objects: impl IntoIterator<Item = ObjectRoster>) {
        self.clear();
        for object in objects {
            match self.roster.binary_search_by_key(&object.object, |r| r.object) {
                Ok(index) => self.roster[index] = object,
                Err(index) => self.roster.insert(index, object),
            }
        }
        self.object_limit = self.roster.len();
        self.property_limit = self.roster.iter().map(|object| object.properties.len()).sum();
        self.objects.reserve(self.object_limit);
        self.counts.reserve(self.object_limit);
        self.properties.reserve(self.property_limit);
        self.values.reserve(self.property_limit);
    }
    /// Identical sorted object/property and last-write semantics to AtomicModeReq.
    /// Every limit check precedes mutation; failure cannot publish a partial request.
    pub(super) fn add(&mut self, object: u32, property: u32, value: u64) -> Result<(), Error> {
        let roster = self
            .roster
            .binary_search_by_key(&object, |r| r.object)
            .map_err(|_| {
                capacity(
                    "DRM atomic object roster",
                    self.objects.len().saturating_add(1),
                    self.object_limit,
                )
            })?;
        if self.roster[roster].properties.binary_search(&property).is_err() {
            return Err(capacity(
                "DRM atomic property roster",
                self.properties.len().saturating_add(1),
                self.property_limit,
            ));
        }
        let (index, count, new_object) = match self.objects.binary_search(&object) {
            Ok(index) => (index, self.counts[index] as usize, false),
            Err(index) => (index, 0, true),
        };
        let start = self.counts[..index]
            .iter()
            .map(|count| *count as usize)
            .sum::<usize>();
        let insertion = match self.properties[start..start + count].binary_search(&property) {
            Ok(index) => {
                self.values[start + index] = value;
                return Ok(());
            }
            Err(index) => start + index,
        };
        if new_object
            && (self.objects.len() == self.object_limit
                || self.objects.len() == self.objects.capacity()
                || self.counts.len() == self.counts.capacity())
        {
            return Err(capacity(
                "DRM atomic objects",
                self.objects.len().saturating_add(1),
                self.object_limit,
            ));
        }
        if self.properties.len() == self.property_limit
            || self.properties.len() == self.properties.capacity()
            || self.values.len() == self.values.capacity()
        {
            return Err(capacity(
                "DRM atomic properties",
                self.properties.len().saturating_add(1),
                self.property_limit,
            ));
        }
        if new_object {
            self.objects.insert(index, object);
            self.counts.insert(index, 0);
        }
        self.counts[index] += 1;
        self.properties.insert(insertion, property);
        self.values.insert(insertion, value);
        Ok(())
    }
    pub(super) fn remember_plane(&mut self, plane: plane::Handle, configured: bool) -> Result<(), Error> {
        if let Some(entry) = self.plane_edits.iter_mut().find(|entry| entry.0 == plane) {
            entry.1 = configured;
            return Ok(());
        }
        if self.plane_edits.len() == self.plane_limit || self.plane_edits.len() == self.plane_edits.capacity()
        {
            return Err(capacity(
                "DRM atomic plane updates",
                self.plane_edits.len().saturating_add(1),
                self.plane_limit,
            ));
        }
        self.plane_edits.push((plane, configured));
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn value(&self, object: u32, property: u32) -> Option<u64> {
        let index = self.objects.binary_search(&object).ok()?;
        let start = self.counts[..index]
            .iter()
            .map(|count| *count as usize)
            .sum::<usize>();
        let offset = self.properties[start..start + self.counts[index] as usize]
            .binary_search(&property)
            .ok()?;
        Some(self.values[start + offset])
    }
    pub(super) fn plane_edits(&self) -> &[(plane::Handle, bool)] {
        &self.plane_edits
    }
    /// DRM copies the request arrays during this syscall even with NONBLOCK.
    /// The original frame continues to own every FB, damage blob and borrowed input fence.
    pub(super) fn commit(&mut self, fd: BorrowedFd<'_>, flags: u32) -> io::Result<()> {
        debug_assert_eq!(self.objects.len(), self.counts.len());
        debug_assert_eq!(self.properties.len(), self.values.len());
        debug_assert_eq!(
            self.counts.iter().map(|count| *count as usize).sum::<usize>(),
            self.properties.len()
        );
        drm_ffi::mode::atomic_commit(
            fd,
            flags,
            &mut self.objects,
            &mut self.counts,
            &mut self.properties,
            &mut self.values,
        )
    }
}
fn capacity(resource: &'static str, required: usize, capacity: usize) -> Error {
    Error::AtomicRequestCapacity {
        resource,
        required,
        capacity,
    }
}

#[cfg(test)]
#[path = "request_storage_tests.rs"]
mod tests;

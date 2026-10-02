//! Borrowed element sequences synthesized from stable numeric selections.
use super::Element;

/// An element sequence whose values borrow the actual accepted caller source.
///
/// Numeric workspaces keep indices, never erased element references. A source
/// can synthesize a borrowed enum for real/fake DRM elements without a fresh
/// temporary vector or copying the accepted element resources.
pub(crate) trait ElementSource {
    type Element<'a>: Element
    where
        Self: 'a;
    fn len(&self) -> usize;
    fn element(&self, index: usize) -> Self::Element<'_>;

    fn iter(&self) -> impl DoubleEndedIterator<Item = Self::Element<'_>> + ExactSizeIterator {
        (0..self.len()).map(|index| self.element(index))
    }
}

impl<E: Element> ElementSource for [E] {
    type Element<'a>
        = &'a E
    where
        E: 'a;
    fn len(&self) -> usize {
        <[E]>::len(self)
    }
    fn element(&self, index: usize) -> Self::Element<'_> {
        &self[index]
    }
}

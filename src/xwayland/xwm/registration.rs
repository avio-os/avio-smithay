//! Event registrations belong to the exact WM episode, including clipboard IO.

use calloop::{EventSource, LoopHandle, Poll, PostAction, Readiness, RegistrationToken, Token, TokenFactory};
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Debug, Clone)]
pub(crate) struct WmLifetime(Arc<AtomicBool>);

impl WmLifetime {
    pub(crate) fn is_alive(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

struct State {
    lifetime: WmLifetime,
    tokens: RefCell<Vec<RegistrationToken>>,
    remove: Box<dyn Fn(RegistrationToken)>,
}

pub(super) struct RegistrationScope(Rc<State>);

impl std::fmt::Debug for RegistrationScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistrationScope")
            .field("tokens", &self.0.tokens.borrow())
            .finish()
    }
}

impl RegistrationScope {
    pub(super) fn new<D: 'static>(handle: &LoopHandle<'static, D>) -> Self {
        let handle = handle.clone();
        Self(Rc::new(State {
            lifetime: WmLifetime(Arc::new(AtomicBool::new(true))),
            tokens: RefCell::new(Vec::new()),
            remove: Box::new(move |token| {
                handle.remove(token);
            }),
        }))
    }

    pub(super) fn lifetime(&self) -> WmLifetime {
        self.0.lifetime.clone()
    }

    pub(super) fn invalidate(&self) {
        self.0.lifetime.0.store(false, Ordering::Release);
    }

    pub(super) fn cancel(&self) {
        self.invalidate();
        let tokens = std::mem::take(&mut *self.0.tokens.borrow_mut());
        for token in tokens {
            (self.0.remove)(token);
        }
    }

    pub(super) fn insert<'l, S, D, F>(
        &self,
        handle: &LoopHandle<'l, D>,
        source: S,
        mut callback: F,
    ) -> calloop::Result<RegistrationToken>
    where
        S: EventSource + 'l,
        S::Ret: StaleReturn,
        D: 'static,
        F: FnMut(S::Event, &mut S::Metadata, &mut D) -> S::Ret + 'l,
    {
        let token = Rc::new(Cell::new(None));
        let scope = Rc::downgrade(&self.0);
        let source = ScopedSource {
            source,
            scope: scope.clone(),
            token: token.clone(),
        };
        let registration = handle
            .insert_source(source, move |event, metadata, data| {
                if scope.upgrade().is_some_and(|s| s.lifetime.is_alive()) {
                    callback(event, metadata, data)
                } else {
                    S::Ret::stale()
                }
            })
            .map_err(|error| error.error)?;
        token.set(Some(registration));
        self.0.tokens.borrow_mut().push(registration);
        Ok(registration)
    }
}

impl Drop for RegistrationScope {
    fn drop(&mut self) {
        // Invalidate before cancellation: an already executing channel can have
        // more queued events, and calloop defers removal of its active source.
        self.cancel();
    }
}

pub(super) trait StaleReturn {
    fn stale() -> Self;
}
impl StaleReturn for () {
    fn stale() {}
}
impl<E> StaleReturn for Result<PostAction, E> {
    fn stale() -> Self {
        Ok(PostAction::Remove)
    }
}

struct ScopedSource<S> {
    source: S,
    scope: Weak<State>,
    token: Rc<Cell<Option<RegistrationToken>>>,
}

impl<S> Drop for ScopedSource<S> {
    fn drop(&mut self) {
        if let (Some(scope), Some(token)) = (self.scope.upgrade(), self.token.get()) {
            scope.tokens.borrow_mut().retain(|current| *current != token);
        }
    }
}

impl<S: EventSource> EventSource for ScopedSource<S> {
    type Event = S::Event;
    type Metadata = S::Metadata;
    type Ret = S::Ret;
    type Error = S::Error;
    fn process_events<F>(
        &mut self,
        readiness: Readiness,
        token: Token,
        callback: F,
    ) -> Result<PostAction, S::Error>
    where
        F: FnMut(S::Event, &mut S::Metadata) -> S::Ret,
    {
        if !self.scope.upgrade().is_some_and(|s| s.lifetime.is_alive()) {
            return Ok(PostAction::Remove);
        }
        self.source.process_events(readiness, token, callback)
    }
    fn register(&mut self, poll: &mut Poll, factory: &mut TokenFactory) -> calloop::Result<()> {
        self.source.register(poll, factory)
    }
    fn reregister(&mut self, poll: &mut Poll, factory: &mut TokenFactory) -> calloop::Result<()> {
        self.source.reregister(poll, factory)
    }
    fn unregister(&mut self, poll: &mut Poll) -> calloop::Result<()> {
        self.source.unregister(poll)
    }
    const NEEDS_EXTRA_LIFECYCLE_EVENTS: bool = S::NEEDS_EXTRA_LIFECYCLE_EVENTS;
    fn before_sleep(&mut self) -> calloop::Result<Option<(Readiness, Token)>> {
        self.source.before_sleep()
    }
    fn before_handle_events(&mut self, events: calloop::EventIterator<'_>) {
        self.source.before_handle_events(events);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calloop::{channel, generic::Generic, EventLoop, Interest, Mode};
    use std::{os::unix::net::UnixStream, time::Duration};
    use x11rb::connection::Connection as _;

    #[test]
    fn retired_episode_cancels_queued_events_and_does_not_touch_successor() {
        let mut event_loop = EventLoop::<usize>::try_new().unwrap();
        let scope = RegistrationScope::new(&event_loop.handle());
        let lifetime = scope.lifetime();
        let (sender, source) = channel::channel::<()>();
        scope
            .insert(&event_loop.handle(), source, |_, _, _| {
                panic!("retired WM lookup")
            })
            .unwrap();
        sender.send(()).unwrap();
        drop(scope);
        assert!(!lifetime.is_alive());
        let successor = RegistrationScope::new(&event_loop.handle());
        let (sender, source) = channel::channel::<()>();
        successor
            .insert(&event_loop.handle(), source, |_, _, count| *count += 1)
            .unwrap();
        sender.send(()).unwrap();
        let mut count = 0;
        event_loop.dispatch(Duration::ZERO, &mut count).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn callback_teardown_suppresses_remaining_events_in_same_dispatch() {
        let mut event_loop = EventLoop::<Option<RegistrationScope>>::try_new().unwrap();
        let scope = RegistrationScope::new(&event_loop.handle());
        let (sender, source) = channel::channel::<()>();
        scope
            .insert(&event_loop.handle(), source, |_, _, scope| {
                assert!(scope.take().is_some(), "late WM lookup");
            })
            .unwrap();
        sender.send(()).unwrap();
        sender.send(()).unwrap();
        let mut scope = Some(scope);
        event_loop.dispatch(Duration::ZERO, &mut scope).unwrap();
        assert!(scope.is_none());
    }

    #[test]
    fn clipboard_source_owns_no_fd_or_callback_after_episode_drop() {
        let mut event_loop = EventLoop::<()>::try_new().unwrap();
        let scope = RegistrationScope::new(&event_loop.handle());
        let (owned, peer) = UnixStream::pair().unwrap();
        scope
            .insert(
                &event_loop.handle(),
                Generic::new(owned, Interest::WRITE, Mode::Level),
                |_, _, _| -> std::io::Result<PostAction> { panic!("retired clipboard WM lookup") },
            )
            .unwrap();
        drop(scope);
        let mut byte = [0];
        assert_eq!(std::io::Read::read(&mut &peer, &mut byte).unwrap(), 0);
        event_loop.dispatch(Duration::ZERO, &mut ()).unwrap();
    }

    #[test]
    fn idle_episode_cancellation_shuts_down_its_actual_x11_reader() {
        let mut event_loop = EventLoop::<()>::try_new().unwrap();
        let scope = RegistrationScope::new(&event_loop.handle());
        let lifetime = scope.lifetime();
        let (connection, mut peer) = crate::utils::x11rb::tests::connection_fixture();
        scope
            .insert(
                &event_loop.handle(),
                crate::utils::x11rb::X11Source::new_owned_connection(connection.clone()),
                |_, _, _| panic!("retired WM lookup"),
            )
            .unwrap();
        drop(scope);
        assert!(!lifetime.is_alive());
        let mut byte = [0];
        assert_eq!(std::io::Read::read(&mut peer, &mut byte).unwrap(), 0);
        event_loop.dispatch(Duration::ZERO, &mut ()).unwrap();
        assert!(connection.wait_for_event().is_err());
    }

    #[test]
    fn completed_source_removes_registration_and_loop_can_drop_first() {
        let mut event_loop = EventLoop::<()>::try_new().unwrap();
        let scope = RegistrationScope::new(&event_loop.handle());
        let (owned, _peer) = UnixStream::pair().unwrap();
        scope
            .insert(
                &event_loop.handle(),
                Generic::new(owned, Interest::WRITE, Mode::Level),
                |_, _, _| -> std::io::Result<PostAction> { Ok(PostAction::Remove) },
            )
            .unwrap();
        event_loop.dispatch(Duration::ZERO, &mut ()).unwrap();
        assert!(scope.0.tokens.borrow().is_empty());
        drop(event_loop);
        drop(scope);
    }
}

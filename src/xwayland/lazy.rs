//! Socket-triggered Xwayland lifetime. Listening sockets outlive server episodes.

use atomic_float::AtomicF64;
use std::{
    ffi::OsString,
    io,
    os::unix::net::UnixStream,
    process::Stdio,
    sync::{atomic::Ordering, Arc},
};

use calloop::{
    channel::{self, Channel, Sender},
    generic::Generic,
    transient::TransientSource,
    EventSource, Interest, Mode, PostAction,
};
use wayland_server::{Client, DisplayHandle};

use super::{
    x11_sockets::{prepare_x11_sockets, X11Lock},
    XWayland, XWaylandEvent,
};

/// An X11 display reservation that survives each Xwayland process episode.
#[derive(Debug)]
pub struct XWaylandDisplay {
    pub(crate) lock: Arc<X11Lock>,
    pub(crate) listen_sockets: Vec<UnixStream>,
}

impl XWaylandDisplay {
    /// Reserves a display number and binds its filesystem/optional abstract sockets.
    pub fn prepare(display: impl Into<Option<u32>>, open_abstract_socket: bool) -> io::Result<Self> {
        let (lock, listen_sockets) = prepare_x11_sockets(display.into(), open_abstract_socket)?;
        Ok(Self {
            lock: Arc::new(lock),
            listen_sockets,
        })
    }

    /// The stable display number, available before a server has started.
    pub fn display_number(&self) -> u32 {
        self.lock.display_number()
    }
}

#[derive(Clone)]
pub(crate) struct XWaylandDisconnectNotify(Arc<dyn Fn() + Send + Sync>);

impl XWaylandDisconnectNotify {
    pub(crate) fn disconnected(&self) {
        (self.0)();
    }
}

/// Shared topology-derived scale for current and future server episodes.
///
/// Each episode has its own transaction queue, but shares this one scale with
/// its compositor client state before that client is inserted in the display.
#[derive(Clone, Debug)]
pub struct LazyXWaylandClientScale(Arc<AtomicF64>);

impl LazyXWaylandClientScale {
    /// Sets a finite positive client scale, including before the first startup.
    pub fn set(&self, scale: f64) -> io::Result<()> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid Xwayland client scale",
            ));
        }
        self.0.store(scale, Ordering::Release);
        Ok(())
    }
}

/// Events from a socket-triggered Xwayland source.
#[derive(Debug)]
pub enum LazyXWaylandEvent {
    /// A server has completed Wayland startup; install the XWM with this client.
    Ready {
        /// Privileged XWM connection.
        x11_socket: UnixStream,
        /// Exact Wayland client for this server episode.
        client: Client,
        /// Reserved display number.
        display_number: u32,
    },
    /// The server exited. Existing XWM/input state must be discarded.
    Exited,
    /// Startup failed. The source stops accepting starts to avoid a crash loop.
    Error(io::Error),
}

enum Completion {
    Spawned(u64, io::Result<(XWayland, Client)>),
    Exited(u64),
}

fn install_source<S: EventSource>(slot: &mut TransientSource<S>, source: S) {
    // Calloop's replace() only replaces an existing source. On None it drops
    // the incoming value. Assignment is safe only for an already empty slot;
    // otherwise replacement must retain the old FD until deregistration.
    if slot.is_none() {
        *slot = source.into();
    } else {
        slot.replace(source);
    }
}

/// Owns a stable display reservation and launches Xwayland on its first client.
///
/// Process creation runs on a helper thread. `-terminate` retires the server
/// after its last X client; the Wayland disconnect event then rearms the same
/// sockets. No mapped-window count, timer, or compositor frame drives lifetime.
/// Return `true` from a `Ready` callback only after the XWM was installed.
pub struct LazyXWayland {
    display: Arc<XWaylandDisplay>,
    client_scale: LazyXWaylandClientScale,
    dh: DisplayHandle,
    envs: Vec<(OsString, OsString)>,
    listeners: Vec<TransientSource<Generic<UnixStream>>>,
    server: TransientSource<OptionalXWayland>,
    sender: Sender<Completion>,
    channel: Channel<Completion>,
    generation: u64,
    starting: bool,
    ready: bool,
    exited_during_start: bool,
    failed: bool,
    client: Option<Client>,
}

impl std::fmt::Debug for LazyXWayland {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyXWayland")
            .field("display", &self.display.display_number())
            .field("generation", &self.generation)
            .field("starting", &self.starting)
            .field("ready", &self.ready)
            .field("failed", &self.failed)
            .finish_non_exhaustive()
    }
}

impl LazyXWayland {
    /// Prepares listening sockets without launching a process.
    pub fn new(
        dh: &DisplayHandle,
        display: impl Into<Option<u32>>,
        open_abstract_socket: bool,
        envs: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> io::Result<Self> {
        let display = Arc::new(XWaylandDisplay::prepare(display, open_abstract_socket)?);
        let (sender, channel) = channel::channel();
        let mut instance = Self {
            display,
            client_scale: LazyXWaylandClientScale(Arc::new(AtomicF64::new(1.0))),
            dh: dh.clone(),
            envs: envs.into_iter().collect(),
            listeners: Vec::new(),
            server: TransientSource::default(),
            sender,
            channel,
            generation: 0,
            starting: false,
            ready: false,
            exited_during_start: false,
            failed: false,
            client: None,
        };
        instance.arm()?;
        Ok(instance)
    }

    /// Display number to publish to X11 clients before their first connection.
    pub fn display_number(&self) -> u32 {
        self.display.display_number()
    }

    /// Handle for updating the exact scale seen by this and future episodes.
    pub fn client_scale(&self) -> LazyXWaylandClientScale {
        self.client_scale.clone()
    }

    fn arm(&mut self) -> io::Result<()> {
        for (index, socket) in self.display.listen_sockets.iter().enumerate() {
            let source = Generic::new(socket.try_clone()?, Interest::READ, Mode::Level);
            if let Some(listener) = self.listeners.get_mut(index) {
                install_source(listener, source);
            } else {
                self.listeners.push(source.into());
            }
        }
        Ok(())
    }

    fn start(&mut self) -> io::Result<()> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| io::Error::other("Xwayland generation exhausted"))?;
        self.starting = true;
        self.exited_during_start = false;
        for listener in &mut self.listeners {
            listener.remove();
        }
        let (dh, display, envs, sender, generation) = (
            self.dh.clone(),
            self.display.clone(),
            self.envs.clone(),
            self.sender.clone(),
            self.generation,
        );
        let client_scale = self.client_scale.0.clone();
        std::thread::Builder::new()
            .name("xwayland-start".into())
            .spawn(move || {
                let disconnect_sender = sender.clone();
                let result = XWayland::spawn_prepared_with_scale(
                    &dh,
                    &display,
                    envs,
                    Stdio::null(),
                    Stdio::null(),
                    move |data| {
                        data.insert_if_missing(|| {
                            XWaylandDisconnectNotify(Arc::new(move || {
                                let _ = disconnect_sender.send(Completion::Exited(generation));
                            }))
                        });
                    },
                    client_scale,
                );
                let _ = sender.send(Completion::Spawned(generation, result));
            })?;
        tracing::trace!(generation = self.generation, "Xwayland socket demand admitted");
        Ok(())
    }

    fn fail(&mut self, error: io::Error, callback: &mut impl FnMut(LazyXWaylandEvent, &mut ()) -> bool) {
        self.failed = true;
        self.starting = false;
        self.ready = false;
        self.server.remove();
        self.client = None;
        for listener in &mut self.listeners {
            listener.remove();
        }
        callback(LazyXWaylandEvent::Error(error), &mut ());
    }
}

impl EventSource for LazyXWayland {
    type Event = LazyXWaylandEvent;
    type Metadata = ();
    type Ret = bool;
    type Error = io::Error;

    fn process_events<F>(
        &mut self,
        readiness: calloop::Readiness,
        token: calloop::Token,
        mut callback: F,
    ) -> io::Result<PostAction>
    where
        F: FnMut(Self::Event, &mut ()) -> bool,
    {
        let mut reregister = false;
        let mut demanded = false;
        for listener in &mut self.listeners {
            let action = listener.process_events(readiness, token, |_, _| {
                demanded = true;
                Ok(PostAction::Continue)
            })?;
            reregister |= action == PostAction::Reregister;
        }
        if demanded && !self.failed && !self.starting && self.client.is_none() {
            if let Err(error) = self.start() {
                self.fail(error, &mut callback);
            }
            reregister = true;
        }
        let mut completions = Vec::new();
        self.channel
            .process_events(readiness, token, |event, _| {
                if let channel::Event::Msg(completion) = event {
                    completions.push(completion);
                }
            })
            .map_err(io::Error::other)?;
        for completion in completions {
            match completion {
                Completion::Spawned(generation, result) if generation == self.generation => {
                    self.starting = false;
                    match result {
                        Ok((server, client)) if !self.exited_during_start && !self.failed => {
                            self.client = Some(client);
                            install_source(
                                &mut self.server,
                                OptionalXWayland {
                                    source: Some(server),
                                    registered: false,
                                    startup_finished: false,
                                },
                            );
                        }
                        Ok(_) => self.fail(io::Error::other("Xwayland exited during startup"), &mut callback),
                        Err(error) => self.fail(error, &mut callback),
                    }
                    reregister = true;
                }
                Completion::Exited(generation) if generation == self.generation => {
                    if self.starting {
                        self.exited_during_start = true;
                        continue;
                    }
                    self.server.remove();
                    self.client = None;
                    if self.ready && !self.failed {
                        self.ready = false;
                        callback(LazyXWaylandEvent::Exited, &mut ());
                        if let Err(error) = self.arm() {
                            self.fail(error, &mut callback);
                        }
                    } else if !self.failed {
                        self.fail(
                            io::Error::other("Xwayland exited before readiness"),
                            &mut callback,
                        );
                    }
                    reregister = true;
                }
                _ => {}
            }
        }
        let mut ready_event = None;
        let action = self.server.process_events(readiness, token, |event, _| {
            ready_event = Some(event);
        })?;
        reregister |= action == PostAction::Reregister;
        if let Some(event) = ready_event {
            match event {
                XWaylandEvent::Ready {
                    x11_socket,
                    display_number,
                } => {
                    let client = self
                        .client
                        .as_ref()
                        .expect("registered server has client")
                        .clone();
                    if callback(
                        LazyXWaylandEvent::Ready {
                            x11_socket,
                            client,
                            display_number,
                        },
                        &mut (),
                    ) {
                        self.ready = true;
                    } else {
                        self.fail(
                            io::Error::other("Xwayland XWM installation rejected"),
                            &mut callback,
                        );
                    }
                }
                XWaylandEvent::Error => self.fail(io::Error::other("Xwayland startup failed"), &mut callback),
            }
            reregister = true;
        }
        Ok(if reregister {
            PostAction::Reregister
        } else {
            PostAction::Continue
        })
    }

    fn register(
        &mut self,
        poll: &mut calloop::Poll,
        factory: &mut calloop::TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.register(poll, factory)?;
        for listener in &mut self.listeners {
            listener.register(poll, factory)?;
        }
        self.server.register(poll, factory)
    }
    fn reregister(
        &mut self,
        poll: &mut calloop::Poll,
        factory: &mut calloop::TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.reregister(poll, factory)?;
        for listener in &mut self.listeners {
            listener.reregister(poll, factory)?;
        }
        self.server.reregister(poll, factory)
    }
    fn unregister(&mut self, poll: &mut calloop::Poll) -> calloop::Result<()> {
        self.channel.unregister(poll)?;
        for listener in &mut self.listeners {
            listener.unregister(poll)?;
        }
        self.server.unregister(poll)
    }
}

// Calloop 0.14's derived TransientSource default unnecessarily requires the
// wrapped source to implement Default. This adapter provides an inert source
// until the first helper result, while TransientSource still owns deregistration.
#[derive(Default)]
struct OptionalXWayland {
    source: Option<XWayland>,
    registered: bool,
    startup_finished: bool,
}

impl EventSource for OptionalXWayland {
    type Event = XWaylandEvent;
    type Metadata = ();
    type Ret = ();
    type Error = io::Error;
    fn process_events<F>(
        &mut self,
        readiness: calloop::Readiness,
        token: calloop::Token,
        mut callback: F,
    ) -> io::Result<PostAction>
    where
        F: FnMut(Self::Event, &mut ()),
    {
        if self.startup_finished {
            return Ok(PostAction::Continue);
        }
        let finished = &mut self.startup_finished;
        match self.source.as_mut() {
            Some(source) => source.process_events(readiness, token, |event, metadata| {
                *finished = true;
                callback(event, metadata);
            }),
            None => Ok(PostAction::Continue),
        }
    }
    fn register(
        &mut self,
        poll: &mut calloop::Poll,
        factory: &mut calloop::TokenFactory,
    ) -> calloop::Result<()> {
        if let Some(source) = self.source.as_mut() {
            if !self.registered && !self.startup_finished {
                source.register(poll, factory)?;
                self.registered = true;
            }
        }
        Ok(())
    }
    fn reregister(
        &mut self,
        poll: &mut calloop::Poll,
        factory: &mut calloop::TokenFactory,
    ) -> calloop::Result<()> {
        if self.registered {
            if let Some(source) = self.source.as_mut() {
                source.reregister(poll, factory)?;
            }
        }
        Ok(())
    }
    fn unregister(&mut self, poll: &mut calloop::Poll) -> calloop::Result<()> {
        // TransientSource retains a disabled server, then unregisters it
        // again when removed. The displayfd has already been deregistered at
        // readiness; retain process custody without deleting the poll entry twice.
        if self.registered {
            if let Some(source) = self.source.as_mut() {
                source.unregister(poll)?;
            }
            self.registered = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calloop::{Dispatcher, EventLoop};

    fn fixture() -> (
        EventLoop<'static, Vec<&'static str>>,
        Dispatcher<'static, LazyXWayland, Vec<&'static str>>,
    ) {
        std::fs::create_dir_all("/tmp/.X11-unix").unwrap();
        let display = wayland_server::Display::<()>::new().unwrap();
        let source = LazyXWayland::new(&display.handle(), None, false, []).unwrap();
        let dispatcher = Dispatcher::new(source, |event, _, events: &mut Vec<&'static str>| {
            events.push(match event {
                LazyXWaylandEvent::Exited => "exited",
                LazyXWaylandEvent::Error(_) => "error",
                LazyXWaylandEvent::Ready { .. } => panic!("fixture never launches a process"),
            });
            true
        });
        let event_loop = EventLoop::try_new().unwrap();
        event_loop
            .handle()
            .register_dispatcher(dispatcher.clone())
            .unwrap();
        (event_loop, dispatcher)
    }

    #[test]
    fn preparing_display_does_not_launch_xwayland() {
        let (_, dispatcher) = fixture();
        let source = dispatcher.as_source_ref();
        assert_eq!(source.generation, 0);
        assert!(!source.starting && !source.ready && !source.failed);
        assert!(source.server.is_none());
    }

    #[test]
    fn completed_episode_rearms_the_same_socket_and_ignores_stale_exit() {
        let (mut event_loop, dispatcher) = fixture();
        let mut source = dispatcher.as_source_mut();
        source.generation = 2;
        source.ready = true;
        let inode = rustix::fs::fstat(&source.display.listen_sockets[0])
            .unwrap()
            .st_ino;
        let number = source.display_number();
        source.sender.send(Completion::Exited(1)).unwrap();
        drop(source);
        let mut events = Vec::new();
        event_loop
            .dispatch(std::time::Duration::from_millis(100), &mut events)
            .unwrap();
        assert!(events.is_empty());
        assert!(dispatcher.as_source_ref().ready);
        dispatcher
            .as_source_ref()
            .sender
            .send(Completion::Exited(2))
            .unwrap();
        event_loop
            .dispatch(std::time::Duration::from_millis(100), &mut events)
            .unwrap();
        assert_eq!(events, ["exited"]);
        let source = dispatcher.as_source_ref();
        assert!(!source.ready && !source.starting && !source.failed);
        assert_eq!(source.display_number(), number);
        assert_eq!(
            rustix::fs::fstat(&source.display.listen_sockets[0])
                .unwrap()
                .st_ino,
            inode
        );
        assert_eq!(source.generation, 2);
        assert!(source.listeners.iter().all(|listener| !listener.is_none()));
    }

    #[test]
    fn exit_before_spawn_result_fails_once_without_restarting() {
        let (mut event_loop, dispatcher) = fixture();
        let mut source = dispatcher.as_source_mut();
        source.generation = 1;
        source.starting = true;
        source.sender.send(Completion::Exited(1)).unwrap();
        source
            .sender
            .send(Completion::Spawned(1, Err(io::Error::other("failed executable"))))
            .unwrap();
        drop(source);
        let mut events = Vec::new();
        event_loop
            .dispatch(std::time::Duration::from_millis(100), &mut events)
            .unwrap();
        assert_eq!(events, ["error"]);
        let source = dispatcher.as_source_ref();
        assert!(source.failed && !source.starting && !source.ready);
        assert!(source.listeners.iter().all(TransientSource::is_none));
    }

    #[test]
    fn actual_ready_source_keeps_client_custody_until_exact_exit_and_rearms() {
        let mut display = wayland_server::Display::<()>::new().unwrap();
        std::fs::create_dir_all("/tmp/.X11-unix").unwrap();
        let mut lazy = LazyXWayland::new(&display.handle(), None, false, []).unwrap();
        lazy.generation = 1;
        lazy.starting = true;
        for listener in &mut lazy.listeners {
            listener.remove();
        }
        let number = lazy.display_number();
        let inode = rustix::fs::fstat(&lazy.display.listen_sockets[0]).unwrap().st_ino;
        let (server, client, _peer) =
            XWayland::readiness_fixture(&display.handle(), &lazy.display, lazy.client_scale.0.clone());
        lazy.sender
            .send(Completion::Spawned(1, Ok((server, client.clone()))))
            .unwrap();
        let mut event_loop = EventLoop::<Vec<&'static str>>::try_new().unwrap();
        let dispatcher = Dispatcher::new(lazy, |event, _, events: &mut Vec<&'static str>| {
            events.push(match event {
                LazyXWaylandEvent::Ready { .. } => "ready",
                LazyXWaylandEvent::Exited => "exited",
                LazyXWaylandEvent::Error(error) => panic!("unexpected readiness error: {error}"),
            });
            true
        });
        event_loop
            .handle()
            .register_dispatcher(dispatcher.clone())
            .unwrap();
        let mut events = Vec::new();
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        // The Rust Wayland backend retires killed client entries during
        // dispatch/cleanup; a fixture has no child peer EOF to wake that pass.
        let _ = display.backend().dispatch_single_client(&mut (), client.id());
        assert_eq!(events, ["ready"]);
        {
            let mut lazy = dispatcher.as_source_mut();
            assert!(lazy.ready);
            assert!(lazy.server.map(|optional| optional.source.is_some()).unwrap());
            assert!(
                display
                    .handle()
                    .backend_handle()
                    .get_client_data(client.id())
                    .is_ok(),
                "Ready must not run XWayland::drop"
            );
            assert!(lazy.listeners.iter().all(TransientSource::is_none));
            lazy.sender.send(Completion::Exited(0)).unwrap();
        }
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        assert!(dispatcher.as_source_ref().ready);
        dispatcher
            .as_source_ref()
            .sender
            .send(Completion::Exited(1))
            .unwrap();
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        let _ = display.backend().dispatch_single_client(&mut (), client.id());
        assert_eq!(events, ["ready", "exited"]);
        assert!(
            display
                .handle()
                .backend_handle()
                .get_client_data(client.id())
                .is_err(),
            "Exact exit retires actual Wayland client custody"
        );
        let lazy = dispatcher.as_source_ref();
        assert_eq!(lazy.display_number(), number);
        assert_eq!(
            rustix::fs::fstat(&lazy.display.listen_sockets[0]).unwrap().st_ino,
            inode
        );
        assert!(lazy.server.is_none());
        assert!(lazy.listeners.iter().all(|source| !source.is_none()));
        drop(lazy);

        // A second episode installs into the now empty server/listener slots,
        // not merely into the original initial registration.
        let mut lazy = dispatcher.as_source_mut();
        lazy.generation = 2;
        lazy.starting = true;
        for listener in &mut lazy.listeners {
            listener.remove();
        }
        let (server, successor, _successor_peer) =
            XWayland::readiness_fixture(&display.handle(), &lazy.display, lazy.client_scale.0.clone());
        lazy.sender
            .send(Completion::Spawned(2, Ok((server, successor.clone()))))
            .unwrap();
        drop(lazy);
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        let _ = display.backend().dispatch_single_client(&mut (), successor.id());
        assert_eq!(events, ["ready", "exited", "ready"]);
        dispatcher
            .as_source_ref()
            .sender
            .send(Completion::Exited(1))
            .unwrap();
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        assert!(display
            .handle()
            .backend_handle()
            .get_client_data(successor.id())
            .is_ok());
        dispatcher
            .as_source_ref()
            .sender
            .send(Completion::Exited(2))
            .unwrap();
        event_loop
            .dispatch(std::time::Duration::ZERO, &mut events)
            .unwrap();
        assert_eq!(events, ["ready", "exited", "ready", "exited"]);
        assert!(dispatcher
            .as_source_ref()
            .listeners
            .iter()
            .all(|source| !source.is_none()));
    }

    #[test]
    fn initial_scale_and_topology_changes_reach_each_exact_client_episode() {
        let display = wayland_server::Display::<()>::new().unwrap();
        std::fs::create_dir_all("/tmp/.X11-unix").unwrap();
        let lazy = LazyXWayland::new(&display.handle(), None, false, []).unwrap();
        let scale = lazy.client_scale();
        scale.set(1.75).unwrap();
        let (first, client, _peer) =
            XWayland::readiness_fixture(&display.handle(), &lazy.display, scale.0.clone());
        let state = &client
            .get_data::<super::super::XWaylandClientData>()
            .unwrap()
            .compositor_state;
        assert_eq!(state.client_scale(), 1.75);
        scale.set(2.5).unwrap();
        assert_eq!(state.client_scale(), 2.5);
        for invalid in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert!(scale.set(invalid).is_err());
        }
        assert_eq!(state.client_scale(), 2.5);
        drop(first);
        let (_second, client, _peer) =
            XWayland::readiness_fixture(&display.handle(), &lazy.display, scale.0.clone());
        assert_eq!(
            client
                .get_data::<super::super::XWaylandClientData>()
                .unwrap()
                .compositor_state
                .client_scale(),
            2.5
        );
    }
}

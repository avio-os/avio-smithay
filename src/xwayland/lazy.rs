//! Socket-triggered Xwayland lifetime. Listening sockets outlive server episodes.

use std::{ffi::OsString, io, os::unix::net::UnixStream, process::Stdio, sync::Arc};

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

/// Owns a stable display reservation and launches Xwayland on its first client.
///
/// Process creation runs on a helper thread. `-terminate` retires the server
/// after its last X client; the Wayland disconnect event then rearms the same
/// sockets. No mapped-window count, timer, or compositor frame drives lifetime.
/// Return `true` from a `Ready` callback only after the XWM was installed.
pub struct LazyXWayland {
    display: Arc<XWaylandDisplay>,
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

    fn arm(&mut self) -> io::Result<()> {
        for (index, socket) in self.display.listen_sockets.iter().enumerate() {
            let source = Generic::new(socket.try_clone()?, Interest::READ, Mode::Level);
            if let Some(listener) = self.listeners.get_mut(index) {
                listener.replace(source);
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
        std::thread::Builder::new()
            .name("xwayland-start".into())
            .spawn(move || {
                let disconnect_sender = sender.clone();
                let result = XWayland::spawn_prepared(
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
                            self.server.replace(OptionalXWayland(Some(server)));
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
struct OptionalXWayland(Option<XWayland>);

impl EventSource for OptionalXWayland {
    type Event = XWaylandEvent;
    type Metadata = ();
    type Ret = ();
    type Error = io::Error;
    fn process_events<F>(
        &mut self,
        readiness: calloop::Readiness,
        token: calloop::Token,
        callback: F,
    ) -> io::Result<PostAction>
    where
        F: FnMut(Self::Event, &mut ()),
    {
        match self.0.as_mut() {
            Some(source) => source.process_events(readiness, token, callback),
            None => Ok(PostAction::Continue),
        }
    }
    fn register(
        &mut self,
        poll: &mut calloop::Poll,
        factory: &mut calloop::TokenFactory,
    ) -> calloop::Result<()> {
        if let Some(source) = self.0.as_mut() {
            source.register(poll, factory)?;
        }
        Ok(())
    }
    fn reregister(
        &mut self,
        poll: &mut calloop::Poll,
        factory: &mut calloop::TokenFactory,
    ) -> calloop::Result<()> {
        if let Some(source) = self.0.as_mut() {
            source.reregister(poll, factory)?;
        }
        Ok(())
    }
    fn unregister(&mut self, poll: &mut calloop::Poll) -> calloop::Result<()> {
        if let Some(source) = self.0.as_mut() {
            source.unregister(poll)?;
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
}

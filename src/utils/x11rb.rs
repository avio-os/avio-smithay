//! Helper utilities for using x11rb as an event source in calloop.
//!
//! The primary use for this module is XWayland integration but is also widely useful for an X11
//! backend in a compositor.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{spawn, JoinHandle},
};

use tracing::{error, warn};
use x11rb::{
    connection::Connection as _,
    protocol::{
        xproto::{Atom, ClientMessageEvent, ConnectionExt as _, EventMask, Window, CLIENT_MESSAGE_EVENT},
        Event,
    },
    rust_connection::RustConnection,
};

use calloop::{
    channel::{channel, Channel, ChannelError, Event as ChannelEvent, Sender},
    EventSource, Poll, PostAction, Readiness, Token, TokenFactory,
};

/// Integration of an x11rb X11 connection with calloop.
///
/// This is a thin wrapper around `Channel`. It works by spawning an extra thread reads events from
/// the X11 connection and then sends them across the channel.
///
/// See [1] for why this extra thread is necessary. The single-thread solution proposed on that
/// page does not work with calloop, since it requires checking something on every main loop
/// iteration. Calloop only allows "when an FD becomes readable".
///
/// [1]: https://docs.rs/x11rb/0.8.1/x11rb/event_loop_integration/index.html#threads-and-races
#[derive(Debug)]
pub struct X11Source {
    connection: Arc<RustConnection>,
    channel: Option<Channel<Event>>,
    event_thread: Option<JoinHandle<()>>,
    close_window: Window,
    close_type: Atom,
    owned_connection_cancelled: Option<Arc<AtomicBool>>,
}

impl X11Source {
    /// Create a new X11 source.
    ///
    /// The returned instance will use `SendRequest` to cause a `ClientMessageEvent` to be sent to
    /// the given window with the given type. The expectation is that this is a window that was
    /// created by us. Thus, the event reading thread will wake up and check an internal exit flag,
    /// then exit.
    pub fn new(connection: Arc<RustConnection>, close_window: Window, close_type: Atom) -> Self {
        Self::create(connection, close_window, close_type, None)
    }

    /// Create an event source that owns the connection's event-reader lifetime.
    ///
    /// Dropping this source shuts down the socket without a server round trip or
    /// a thread join. All aliases become unusable. This is intended for the
    /// private XWM connection, not a connection shared by independent owners.
    pub fn new_owned_connection(connection: Arc<RustConnection>) -> Self {
        Self::create(connection, 0, 0, Some(Arc::new(AtomicBool::new(false))))
    }

    fn create(
        connection: Arc<RustConnection>,
        close_window: Window,
        close_type: Atom,
        owned_connection_cancelled: Option<Arc<AtomicBool>>,
    ) -> Self {
        let (sender, channel) = channel();
        let conn = Arc::clone(&connection);
        let cancelled = owned_connection_cancelled.clone();
        let event_thread = Some(spawn(move || {
            run_event_thread(conn, sender, cancelled);
        }));

        Self {
            connection,
            channel: Some(channel),
            event_thread,
            close_window,
            close_type,
            owned_connection_cancelled,
        }
    }
}

impl Drop for X11Source {
    fn drop(&mut self) {
        if let Some(cancelled) = &self.owned_connection_cancelled {
            cancelled.store(true, Ordering::Release);
            self.channel.take();
            // Shutdown wakes the existing reader even when the server never
            // replies. Its Arc keeps the exact FD alive until wait_for_event
            // returns EOF; no FD duplication, protocol request, or join occurs.
            if let Err(error) = rustix::net::shutdown(self.connection.stream(), rustix::net::Shutdown::Both) {
                warn!(?error, "Failed to shut down the owned X11 connection");
            }
            self.event_thread.take();
            return;
        }

        // General shared connections retain the original client-message API.
        self.channel.take();

        // Send an event to wake up the worker so that it actually exits
        let event = ClientMessageEvent {
            response_type: CLIENT_MESSAGE_EVENT,
            format: 8,
            sequence: 0,
            window: self.close_window,
            type_: self.close_type,
            data: [0; 20].into(),
        };

        let _ = self
            .connection
            .send_event(false, self.close_window, EventMask::NO_EVENT, event);
        let _ = self.connection.flush();

        // Wait for the worker thread to exit
        self.event_thread.take().map(|handle| handle.join());
    }
}

impl EventSource for X11Source {
    type Event = ChannelEvent<Event>;
    type Metadata = ();
    type Ret = ();
    type Error = ChannelError;

    #[profiling::function]
    fn process_events<C>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut callback: C,
    ) -> Result<PostAction, ChannelError>
    where
        C: FnMut(Self::Event, &mut Self::Metadata) -> Self::Ret,
    {
        if let Some(channel) = &mut self.channel {
            channel.process_events(readiness, token, move |event, meta| {
                if matches!(event, ChannelEvent::Closed) {
                    warn!("Event thread exited");
                }
                callback(event, meta)
            })
        } else {
            Ok(PostAction::Remove)
        }
    }

    fn register(&mut self, poll: &mut Poll, factory: &mut TokenFactory) -> calloop::Result<()> {
        if let Some(channel) = &mut self.channel {
            channel.register(poll, factory)?;
        }

        Ok(())
    }

    fn reregister(&mut self, poll: &mut Poll, factory: &mut TokenFactory) -> calloop::Result<()> {
        if let Some(channel) = &mut self.channel {
            channel.reregister(poll, factory)?;
        }

        Ok(())
    }

    fn unregister(&mut self, poll: &mut Poll) -> calloop::Result<()> {
        if let Some(channel) = &mut self.channel {
            channel.unregister(poll)?;
        }

        Ok(())
    }
}

/// This thread reads X11 events from the connection and sends them on the channel.
///
/// This is run in an extra thread since sending an X11 request or waiting for the reply to an X11
/// request can both read X11 events from the underlying socket which are then saved in the
/// RustConnection. Thus, readability of the underlying socket is not enough to guarantee we do not
/// miss wakeups.
///
/// This thread will call wait_for_event(). RustConnection then ensures internally to wake us up
/// when an event arrives. So far, this seems to be the only safe way to integrate x11rb with
/// calloop.
fn run_event_thread(
    connection: Arc<RustConnection>,
    sender: Sender<Event>,
    cancelled: Option<Arc<AtomicBool>>,
) {
    loop {
        let event = match connection.wait_for_event() {
            Ok(event) => event,
            Err(err) => {
                // Connection errors are most likely permanent. Thus, exit the thread.
                if !cancelled
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::Acquire))
                {
                    error!("Event thread exiting due to connection error {}", err);
                }
                break;
            }
        };
        match sender.send(event) {
            Ok(()) => {}
            Err(_) => {
                // The only possible error is that the other end of the channel was dropped.
                // This happens in X11Source's Drop impl.
                break;
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        os::unix::net::UnixStream,
        sync::mpsc,
        time::{Duration, Instant},
    };
    use x11rb::{
        protocol::xproto::{Screen, Setup},
        rust_connection::DefaultStream,
        x11_utils::Serialize,
    };

    /// A real RustConnection handshake and reader, with a controlled silent
    /// protocol peer. This does not launch or certify an X server process.
    pub(crate) fn connection_fixture() -> (Arc<RustConnection>, UnixStream) {
        let (socket, mut peer) = UnixStream::pair().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let handshake = spawn(move || {
            let mut request = [0; 12];
            peer.read_exact(&mut request).unwrap();
            assert_eq!(&request[6..10], &[0; 4]); // No authentication payload.
            let mut setup = Setup {
                status: 1,
                protocol_major_version: 11,
                resource_id_base: 0x200000,
                resource_id_mask: 0x1fffff,
                maximum_request_length: u16::MAX,
                roots: vec![Screen {
                    root: 1,
                    width_in_pixels: 1920,
                    height_in_pixels: 1080,
                    root_depth: 24,
                    ..Screen::default()
                }],
                ..Setup::default()
            };
            setup.length = ((setup.serialize().len() - 8) / 4).try_into().unwrap();
            peer.write_all(&setup.serialize()).unwrap();
            peer
        });
        let connection =
            RustConnection::connect_to_stream(DefaultStream::from_unix_stream(socket).unwrap().0, 0).unwrap();
        (Arc::new(connection), handshake.join().unwrap())
    }

    #[test]
    fn owned_reader_shutdown_needs_no_peer_reply_or_close_request() {
        let (connection, mut peer) = connection_fixture();
        let source = X11Source::new_owned_connection(connection.clone());
        let (done, finished) = mpsc::channel();
        let dropper = spawn(move || {
            drop(source);
            done.send(()).unwrap();
        });
        finished.recv_timeout(Duration::from_secs(2)).unwrap();
        // The peer sends no event/reply. Drop has performed native shutdown,
        // not a ClientMessage request, and the real blocked reader exits.
        let mut byte = [0];
        assert_eq!(peer.read(&mut byte).unwrap(), 0);
        dropper.join().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Arc::strong_count(&connection) != 1 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(
            Arc::strong_count(&connection),
            1,
            "reader retained its connection after socket EOF"
        );
        assert!(
            connection.wait_for_event().is_err(),
            "retained alias survived the owned epoch withdrawal"
        );
    }
}

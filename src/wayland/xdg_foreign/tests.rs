//! Protocol tests use real Wayland sockets without rendering or a session.
use super::{XdgForeignHandler, XdgForeignState};
use crate::{
    delegate_compositor, delegate_xdg_foreign, delegate_xdg_shell,
    utils::Serial,
    wayland::{
        compositor::{CompositorClientState, CompositorHandler, CompositorState},
        shell::xdg::{PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState},
    },
};
use std::{collections::HashMap, os::unix::net::UnixStream, sync::Arc};
use wayland_client::protocol::{wl_callback, wl_compositor, wl_registry, wl_surface};
use wayland_client::{delegate_noop, Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols::xdg::{
    foreign::zv2::client::{zxdg_exported_v2, zxdg_exporter_v2, zxdg_imported_v2, zxdg_importer_v2},
    shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base},
};
use wayland_server::{Client, Display, Resource};

#[derive(Debug, Default)]
struct ClientData {
    compositor: CompositorClientState,
}
impl wayland_server::backend::ClientData for ClientData {}
struct Server {
    compositor: CompositorState,
    shell: XdgShellState,
    foreign: XdgForeignState,
    changes: usize,
}
impl CompositorHandler for Server {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor
    }
    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientData>().unwrap().compositor
    }
    fn commit(&mut self, _: &wayland_server::protocol::wl_surface::WlSurface) {}
}
impl XdgShellHandler for Server {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.shell
    }
    fn new_toplevel(&mut self, _: ToplevelSurface) {}
    fn new_popup(&mut self, _: PopupSurface, _: PositionerState) {}
    fn grab(&mut self, _: PopupSurface, _: wayland_server::protocol::wl_seat::WlSeat, _: Serial) {}
    fn reposition_request(&mut self, _: PopupSurface, _: PositionerState, _: u32) {}
    fn parent_changed(&mut self, _: ToplevelSurface) {
        self.changes += 1;
    }
}
impl XdgForeignHandler for Server {
    fn xdg_foreign_state(&mut self) -> &mut XdgForeignState {
        &mut self.foreign
    }
}
delegate_compositor!(Server);
delegate_xdg_shell!(Server);
delegate_xdg_foreign!(Server);
#[derive(Default)]
struct ClientEvents {
    globals: HashMap<String, u32>,
    handles: Vec<String>,
    revoked: Vec<u32>,
}
impl Dispatch<wl_registry::WlRegistry, ()> for ClientEvents {
    fn event(
        s: &mut Self,
        _: &wl_registry::WlRegistry,
        e: wl_registry::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, .. } = e {
            s.globals.insert(interface, name);
        }
    }
}
impl Dispatch<zxdg_exported_v2::ZxdgExportedV2, ()> for ClientEvents {
    fn event(
        s: &mut Self,
        _: &zxdg_exported_v2::ZxdgExportedV2,
        e: zxdg_exported_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zxdg_exported_v2::Event::Handle { handle } = e {
            s.handles.push(handle);
        }
    }
}
impl Dispatch<zxdg_imported_v2::ZxdgImportedV2, ()> for ClientEvents {
    fn event(
        s: &mut Self,
        o: &zxdg_imported_v2::ZxdgImportedV2,
        e: zxdg_imported_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zxdg_imported_v2::Event::Destroyed = e {
            s.revoked.push(o.id().protocol_id());
        }
    }
}
delegate_noop!(ClientEvents: ignore wl_callback::WlCallback);
delegate_noop!(ClientEvents: ignore wl_compositor::WlCompositor);
delegate_noop!(ClientEvents: ignore wl_surface::WlSurface);
delegate_noop!(ClientEvents: ignore xdg_wm_base::XdgWmBase);
delegate_noop!(ClientEvents: ignore xdg_surface::XdgSurface);
delegate_noop!(ClientEvents: ignore xdg_toplevel::XdgToplevel);
delegate_noop!(ClientEvents: ignore zxdg_exporter_v2::ZxdgExporterV2);
delegate_noop!(ClientEvents: ignore zxdg_importer_v2::ZxdgImporterV2);
struct Harness {
    display: Display<Server>,
    server: Server,
    client: Connection,
    queue: EventQueue<ClientEvents>,
    events: ClientEvents,
    compositor: wl_compositor::WlCompositor,
    shell: xdg_wm_base::XdgWmBase,
    exporter: zxdg_exporter_v2::ZxdgExporterV2,
    importer: zxdg_importer_v2::ZxdgImporterV2,
}
fn pump(
    display: &mut Display<Server>,
    server: &mut Server,
    client: &Connection,
    queue: &mut EventQueue<ClientEvents>,
    events: &mut ClientEvents,
) {
    client.display().sync(&queue.handle(), ());
    client.flush().unwrap();
    display.dispatch_clients(server).unwrap();
    display.flush_clients().unwrap();
    if let Some(read) = queue.prepare_read() {
        read.read().unwrap();
    }
    queue.dispatch_pending(events).unwrap();
}
impl Harness {
    fn new() -> Self {
        let mut display = Display::<Server>::new().unwrap();
        let mut dh = display.handle();
        let mut server = Server {
            compositor: CompositorState::new::<Server>(&dh),
            shell: XdgShellState::new::<Server>(&dh),
            foreign: XdgForeignState::new::<Server>(&dh),
            changes: 0,
        };
        let (a, b) = UnixStream::pair().unwrap();
        dh.insert_client(a, Arc::new(ClientData::default())).unwrap();
        let client = Connection::from_socket(b).unwrap();
        let mut queue = client.new_event_queue();
        let q = queue.handle();
        let mut events = ClientEvents::default();
        let registry = client.display().get_registry(&q, ());
        pump(&mut display, &mut server, &client, &mut queue, &mut events);
        let compositor = registry.bind(events.globals["wl_compositor"], 1, &q, ());
        let shell = registry.bind(events.globals["xdg_wm_base"], 1, &q, ());
        let exporter = registry.bind(events.globals["zxdg_exporter_v2"], 1, &q, ());
        let importer = registry.bind(events.globals["zxdg_importer_v2"], 1, &q, ());
        Self {
            display,
            server,
            client,
            queue,
            events,
            compositor,
            shell,
            exporter,
            importer,
        }
    }
    fn pump(&mut self) {
        pump(
            &mut self.display,
            &mut self.server,
            &self.client,
            &mut self.queue,
            &mut self.events,
        );
    }
    fn window(
        &mut self,
    ) -> (
        wl_surface::WlSurface,
        xdg_surface::XdgSurface,
        xdg_toplevel::XdgToplevel,
    ) {
        let q = self.queue.handle();
        let surface = self.compositor.create_surface(&q, ());
        let xdg = self.shell.get_xdg_surface(&surface, &q, ());
        let top = xdg.get_toplevel(&q, ());
        self.pump();
        (surface, xdg, top)
    }
    fn export(&mut self, surface: &wl_surface::WlSurface) -> zxdg_exported_v2::ZxdgExportedV2 {
        let export = self.exporter.export_toplevel(surface, &self.queue.handle(), ());
        self.pump();
        export
    }
    fn import(&mut self) -> zxdg_imported_v2::ZxdgImportedV2 {
        self.importer.import_toplevel(
            self.events.handles.last().unwrap().clone(),
            &self.queue.handle(),
            (),
        )
    }
    fn parent(&self, child: &wl_surface::WlSurface) -> Option<u32> {
        self.server
            .shell
            .toplevel_surfaces()
            .iter()
            .find(|t| t.wl_surface().id().protocol_id() == child.id().protocol_id())
            .unwrap()
            .parent()
            .map(|p| p.id().protocol_id())
    }
}
#[test]
fn multiple_imports_and_children_revoke_independently() {
    let mut h = Harness::new();
    let (p, _, _) = h.window();
    let (a, _, _) = h.window();
    let (b, _, _) = h.window();
    let (c, _, _) = h.window();
    let export = h.export(&p);
    let first = h.import();
    let second = h.import();
    first.set_parent_of(&a);
    first.set_parent_of(&b);
    second.set_parent_of(&c);
    h.pump();
    for child in [&a, &b, &c] {
        assert_eq!(h.parent(child), Some(p.id().protocol_id()));
    }
    first.destroy();
    h.pump();
    assert_eq!(h.parent(&a), None);
    assert_eq!(h.parent(&b), None);
    assert_eq!(h.parent(&c), Some(p.id().protocol_id()));
    export.destroy();
    h.pump();
    assert_eq!(h.parent(&c), None);
    assert_eq!(h.events.revoked, vec![second.id().protocol_id()]);
    assert_eq!(h.server.changes, 6);
}
#[test]
fn parent_surface_destruction_revokes_every_import_once() {
    let mut h = Harness::new();
    let (p, px, pt) = h.window();
    let (a, _, _) = h.window();
    let (b, _, _) = h.window();
    let export = h.export(&p);
    let first = h.import();
    let second = h.import();
    first.set_parent_of(&a);
    second.set_parent_of(&b);
    h.pump();
    pt.destroy();
    px.destroy();
    p.destroy();
    h.pump();
    assert_eq!(h.parent(&a), None);
    assert_eq!(h.parent(&b), None);
    assert_eq!(h.events.revoked.len(), 2);
    export.destroy();
    h.pump();
    assert_eq!(h.events.revoked.len(), 2);
}
#[test]
fn superseded_import_cannot_clear_current_parent() {
    let mut h = Harness::new();
    let (p, _, pt) = h.window();
    let (a, _, at) = h.window();
    let export = h.export(&p);
    let first = h.import();
    let second = h.import();
    first.set_parent_of(&a);
    second.set_parent_of(&a);
    h.pump();
    first.destroy();
    h.pump();
    assert_eq!(h.parent(&a), Some(p.id().protocol_id()));
    at.set_parent(Some(&pt));
    h.pump();
    export.destroy();
    h.pump();
    assert_eq!(h.parent(&a), Some(p.id().protocol_id()));
}
#[test]
fn destroyed_child_leaves_no_foreign_relationship() {
    let mut h = Harness::new();
    let (p, _, _) = h.window();
    let (a, ax, at) = h.window();
    let export = h.export(&p);
    let first = h.import();
    first.set_parent_of(&a);
    h.pump();
    at.destroy();
    ax.destroy();
    a.destroy();
    h.pump();
    assert!(h
        .server
        .foreign
        .exported
        .values()
        .all(|e| e.imported_by.values().all(|c| c.is_empty())));
    export.destroy();
    h.pump();
    assert_eq!(h.events.revoked.len(), 1);
}

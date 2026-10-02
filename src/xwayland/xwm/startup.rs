//! Initial policy succeeds before the one native X11 client-admission request.

pub(super) fn initialize_before_admission<T, E>(
    wm: &mut T,
    initialize: impl FnOnce(&mut T) -> Result<(), E>,
    admit: impl FnOnce(&mut T) -> Result<(), E>,
) -> Result<(), E> {
    initialize(wm)?;
    admit(wm)
}

use super::*;

impl X11Wm {
    /// Starts the WM while holding Xwayland's initial client-admission gate.
    ///
    /// `initialize` must publish initial toolkit settings, resources and any
    /// required cursor before returning. An initialization error never claims
    /// `WM_S0`, so waiting clients cannot observe partially initialized policy.
    pub fn start_wm_with_initialization<D, F>(
        handle: LoopHandle<'static, D>,
        dh: &DisplayHandle,
        connection: UnixStream,
        client: wayland_server::Client,
        initialize: F,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        D: XwmHandler + xwayland_shell::XWaylandShellHandler + SeatHandler + DndGrabHandler + 'static,
        <D as SeatHandler>::PointerFocus: DndFocus<D>,
        <D as SeatHandler>::TouchFocus: DndFocus<D>,
        F: FnOnce(&mut Self) -> Result<(), Box<dyn std::error::Error>>,
    {
        let id = XwmId(xwm_id::next());
        let span = debug_span!("xwayland_wm", id = id.0);
        let _guard = span.enter();

        // Create an X11 connection. XWayland only uses screen 0.
        let screen = 0;
        let stream = DefaultStream::from_unix_stream(connection)?.0;
        let conn = RustConnection::connect_to_stream(stream, screen)?;
        let atoms = Atoms::new(&conn)?.reply()?;
        let screen = conn.setup().roots[0].clone();
        let randr_primary = conn.randr_get_output_primary(screen.root)?.reply()?.output;

        {
            let font = FontWrapper::open_font(&conn, "cursor".as_bytes())?;
            let cursor = CursorWrapper::create_glyph_cursor(
                &conn,
                font.font(),
                font.font(),
                68,
                69,
                0,
                0,
                0,
                u16::MAX,
                u16::MAX,
                u16::MAX,
            )?;

            // Actually become the WM by redirecting some operations
            conn.change_window_attributes(
                screen.root,
                &ChangeWindowAttributesAux::default()
                    .event_mask(
                        EventMask::SUBSTRUCTURE_REDIRECT
                            | EventMask::SUBSTRUCTURE_NOTIFY
                            | EventMask::PROPERTY_CHANGE
                            | EventMask::FOCUS_CHANGE,
                    )
                    // and also set a default root cursor in case downstream doesn't
                    .cursor(cursor.cursor()),
            )?;
            // Watch for primary output changes
            conn.randr_select_input(screen.root, NotifyMask::OUTPUT_CHANGE | NotifyMask::SCREEN_CHANGE)?;
        }

        // Build the WM window without claiming WM_S0: queued X11 clients
        // stay blocked until all initial policy has been published.
        let win = conn.generate_id()?;
        conn.create_window(
            screen.root_depth,
            win,
            screen.root,
            // x, y, width, height, border width
            0,
            0,
            1,
            1,
            0,
            WindowClass::INPUT_OUTPUT,
            x11rb::COPY_FROM_PARENT,
            &Default::default(),
        )?;
        conn.set_selection_owner(win, atoms._NET_WM_CM_S0, x11rb::CURRENT_TIME)?;
        conn.composite_redirect_subwindows(screen.root, Redirect::MANUAL)?;

        // Set some EWMH properties
        conn.change_property32(
            PropMode::REPLACE,
            screen.root,
            atoms._NET_SUPPORTED,
            AtomEnum::ATOM,
            &[
                atoms._NET_WM_STATE,
                atoms._NET_WM_STATE_MAXIMIZED_HORZ,
                atoms._NET_WM_STATE_MAXIMIZED_VERT,
                atoms._NET_WM_STATE_HIDDEN,
                atoms._NET_WM_STATE_FULLSCREEN,
                atoms._NET_WM_STATE_MODAL,
                atoms._NET_WM_STATE_FOCUSED,
                atoms._NET_ACTIVE_WINDOW,
                atoms._NET_WM_MOVERESIZE,
                atoms._NET_CLIENT_LIST,
                atoms._NET_CLIENT_LIST_STACKING,
            ],
        )?;
        conn.change_property32(
            PropMode::REPLACE,
            screen.root,
            atoms._NET_CLIENT_LIST,
            AtomEnum::WINDOW,
            &[],
        )?;
        conn.change_property32(
            PropMode::REPLACE,
            screen.root,
            atoms._NET_CLIENT_LIST_STACKING,
            AtomEnum::WINDOW,
            &[],
        )?;
        conn.change_property32(
            PropMode::REPLACE,
            screen.root,
            atoms._NET_ACTIVE_WINDOW,
            AtomEnum::WINDOW,
            &[0],
        )?;
        conn.change_property32(
            PropMode::REPLACE,
            screen.root,
            atoms._NET_SUPPORTING_WM_CHECK,
            AtomEnum::WINDOW,
            &[win],
        )?;
        conn.change_property32(
            PropMode::REPLACE,
            win,
            atoms._NET_SUPPORTING_WM_CHECK,
            AtomEnum::WINDOW,
            &[win],
        )?;
        conn.change_property8(
            PropMode::REPLACE,
            win,
            atoms._NET_WM_NAME,
            atoms.UTF8_STRING,
            "Smithay X WM".as_bytes(),
        )?;
        debug!(window = win, "Created WM Window");

        let conn = Arc::new(conn);
        let xsettings = XSettings::new(&conn, screen.root_depth, screen.root, &atoms)?;
        conn.flush()?;

        let source = X11Source::new_owned_connection(Arc::clone(&conn));

        let client_data = client.get_data::<XWaylandClientData>().unwrap();
        let client_scale = client_data.compositor_state.clone_client_scale();

        // We need this for the commit hook.
        client_data.user_data().insert_if_missing(|| id);

        let _xfixes_data = conn
            .query_extension(x11rb::protocol::xfixes::X11_EXTENSION_NAME.as_bytes())?
            .reply_unchecked()?
            .ok_or(ConnectionError::UnsupportedExtension)?;
        if !_xfixes_data.present {
            return Err(ConnectionError::UnsupportedExtension.into());
        }
        conn.xfixes_query_version(1, 0)?.reply_unchecked()?; // we just need version 1 for clipboard monitoring

        let clipboard = XWmSelection::new(&conn, &screen, &atoms, atoms.CLIPBOARD)?;
        let primary = XWmSelection::new(&conn, &screen, &atoms, atoms.PRIMARY)?;
        let dnd = XWmDnd::new(&conn, &screen, &atoms)?;
        let wm_window = OwnedX11Window::new(win, &conn);

        drop(_guard);
        let mut wm = Self {
            id,
            registrations: RegistrationScope::new(&handle),
            conn,
            client_scale,
            screen,
            atoms,
            xsettings,
            randr_primary,
            wm_window,
            _xfixes_data,
            clipboard,
            primary,
            dnd,
            unpaired_surfaces: Default::default(),
            sequences_to_ignore: Default::default(),
            windows: Vec::new(),
            client_list: Vec::new(),
            client_list_stacking: Vec::new(),
            span,
        };

        let event_handle = handle.clone();
        let dh = dh.clone();
        // The lifetime is also checked by late Wayland commit hooks. Old
        // client user data cannot resolve a newly minted WM episode.
        client_data
            .user_data()
            .insert_if_missing(|| wm.registrations.lifetime());
        wm.registrations
            .insert(&handle, source, move |event, _, data| match event {
                calloop::channel::Event::Msg(event) => {
                    if let Err(err) = handle_event(&event_handle, &dh, data, id, event) {
                        warn!(id = id.0, err = ?err, "Failed to handle X11 event");
                    }
                }
                calloop::channel::Event::Closed => {
                    data.disconnected(id);
                }
            })?;
        // This is the only initial X11 client admission operation. It follows
        // successful initialization and registration, on the same connection.
        initialize_before_admission(&mut wm, initialize, |wm| {
            wm.conn
                .set_selection_owner(win, wm.atoms.WM_S0, x11rb::CURRENT_TIME)?
                .check()?;
            wm.conn.flush()?;
            Ok(())
        })?;
        Ok(wm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_clients_can_observe_only_the_complete_initial_policy() {
        let mut observed = Vec::new();
        initialize_before_admission(
            &mut observed,
            |operations| {
                operations.extend(["client-scale", "XSETTINGS", "RESOURCE_MANAGER"]);
                Ok::<_, ()>(())
            },
            |operations| {
                assert_eq!(operations, &["client-scale", "XSETTINGS", "RESOURCE_MANAGER"]);
                operations.push("WM_S0");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(observed.last(), Some(&"WM_S0"));
    }

    #[test]
    fn failed_required_metadata_never_releases_first_client_admission() {
        for failure in ["XSETTINGS", "RESOURCE_MANAGER"] {
            let mut observed = Vec::new();
            let result = initialize_before_admission(
                &mut observed,
                |operations| {
                    operations.push("client-scale");
                    operations.push(failure);
                    Err(failure)
                },
                |_| -> Result<(), &str> { panic!("failed initialization released clients") },
            );
            assert_eq!(result, Err(failure));
            assert!(!observed.contains(&"WM_S0"));
        }
    }
}

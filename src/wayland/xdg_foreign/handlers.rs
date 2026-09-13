use std::collections::HashMap;

use wayland_protocols::xdg::foreign::zv2::server::{
    zxdg_exported_v2::{self, ZxdgExportedV2},
    zxdg_exporter_v2::{self, ZxdgExporterV2},
    zxdg_imported_v2::{self, ZxdgImportedV2},
    zxdg_importer_v2::{self, ZxdgImporterV2},
};
use wayland_server::{
    backend::ClientId, Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};

use crate::wayland::{
    compositor,
    shell::{
        is_valid_parent,
        xdg::{XdgShellHandler, XdgToplevelSurfaceData, XDG_TOPLEVEL_ROLE},
    },
};

use super::{
    ExportedState, XdgExportedUserData, XdgForeignHandle, XdgForeignHandler, XdgForeignState,
    XdgImportedUserData,
};

//
// Export
//

impl<D> GlobalDispatch<ZxdgExporterV2, (), D> for XdgForeignState
where
    D: Dispatch<ZxdgExporterV2, ()>,
{
    fn bind(
        _state: &mut D,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<ZxdgExporterV2>,
        _global_data: &(),
        data_init: &mut DataInit<'_, D>,
    ) {
        data_init.init(resource, ());
    }
}

impl<D> Dispatch<ZxdgExporterV2, (), D> for XdgForeignState
where
    D: Dispatch<ZxdgExportedV2, XdgExportedUserData>,
    D: XdgForeignHandler + XdgShellHandler,
{
    fn request(
        state: &mut D,
        _client: &Client,
        resource: &ZxdgExporterV2,
        request: zxdg_exporter_v2::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            zxdg_exporter_v2::Request::ExportToplevel { id, surface } => {
                if compositor::get_role(&surface) != Some(XDG_TOPLEVEL_ROLE) {
                    resource.post_error(
                        zxdg_exporter_v2::Error::InvalidSurface,
                        "exported surface had an invalid role",
                    );
                    return;
                }

                let handle = XdgForeignHandle::new();
                let exported = data_init.init(
                    id,
                    XdgExportedUserData {
                        handle: handle.clone(),
                    },
                );
                exported.handle(handle.as_str().to_owned());

                let revoked = handle.clone();
                let destruction_hook = compositor::add_destruction_hook::<D, _>(&surface, move |state, _| {
                    revoke_export(state, &revoked, false)
                });
                state.xdg_foreign_state().exported.insert(
                    handle,
                    ExportedState {
                        exported_surface: surface,
                        destruction_hook,
                        imported_by: HashMap::new(),
                    },
                );
            }
            zxdg_exporter_v2::Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch<ZxdgExportedV2, XdgExportedUserData, D> for XdgForeignState
where
    D: XdgForeignHandler + XdgShellHandler,
{
    fn request(
        _state: &mut D,
        _client: &Client,
        _resource: &ZxdgExportedV2,
        _request: zxdg_exported_v2::Request,
        _data: &XdgExportedUserData,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
    }

    fn destroyed(state: &mut D, _client: ClientId, _resource: &ZxdgExportedV2, data: &XdgExportedUserData) {
        // Revoke the previously exported surface.
        // This invalidates any relationship the importer may have set up using the xdg_imported created given the handle sent via xdg_exported.handle.
        revoke_export(state, &data.handle, true);
    }
}

//
// Import
//

impl<D> GlobalDispatch<ZxdgImporterV2, (), D> for XdgForeignState
where
    D: Dispatch<ZxdgImporterV2, ()>,
{
    fn bind(
        _state: &mut D,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: New<ZxdgImporterV2>,
        _global_data: &(),
        data_init: &mut DataInit<'_, D>,
    ) {
        data_init.init(resource, ());
    }
}

impl<D: XdgForeignHandler> Dispatch<ZxdgImporterV2, (), D> for XdgForeignState
where
    D: Dispatch<ZxdgImportedV2, XdgImportedUserData>,
{
    fn request(
        state: &mut D,
        _client: &Client,
        _resource: &ZxdgImporterV2,
        request: zxdg_importer_v2::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            zxdg_importer_v2::Request::ImportToplevel { id, handle } => {
                let exported = state
                    .xdg_foreign_state()
                    .exported
                    .iter_mut()
                    .find(|(key, _)| key.as_str() == handle.as_str());

                let imported = data_init.init(
                    id,
                    XdgImportedUserData {
                        handle: XdgForeignHandle(handle),
                    },
                );

                match exported {
                    Some((_, state)) => {
                        state.imported_by.insert(imported, Default::default());
                    }
                    None => {
                        imported.destroyed();
                    }
                }
            }
            zxdg_importer_v2::Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch<ZxdgImportedV2, XdgImportedUserData, D> for XdgForeignState
where
    D: XdgForeignHandler + XdgShellHandler,
{
    fn request(
        state: &mut D,
        _client: &Client,
        resource: &ZxdgImportedV2,
        request: zxdg_imported_v2::Request,
        data: &XdgImportedUserData,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            zxdg_imported_v2::Request::SetParentOf { surface: child } => {
                let Some(parent) = state
                    .xdg_foreign_state()
                    .exported
                    .get(&data.handle)
                    .map(|entry| entry.exported_surface.clone())
                else {
                    return;
                };
                // Every import can parent multiple children. A child has only
                // one current parent and one exact import owning that edge.
                if compositor::get_role(&child) != Some(XDG_TOPLEVEL_ROLE)
                    || !is_valid_parent(&child, &parent)
                {
                    resource.post_error(
                        zxdg_imported_v2::Error::InvalidSurface,
                        "invalid parent relationship",
                    );
                    return;
                }
                forget_child(state, &child);
                let (changed, needs_hook) = compositor::with_states(&child, |states| {
                    let mut role = states
                        .data_map
                        .get::<XdgToplevelSurfaceData>()
                        .expect("validated xdg toplevel")
                        .lock()
                        .unwrap();
                    let changed = role.parent.as_ref() != Some(&parent);
                    role.parent = Some(parent);
                    role.foreign_parent = Some(resource.id());
                    (
                        changed,
                        states.data_map.insert_if_missing_threadsafe(|| ForeignChildHook),
                    )
                });
                if needs_hook {
                    compositor::add_destruction_hook::<D, _>(&child, forget_child::<D>);
                }
                state
                    .xdg_foreign_state()
                    .exported
                    .get_mut(&data.handle)
                    .expect("export validated above")
                    .imported_by
                    .get_mut(resource)
                    .expect("live import")
                    .insert(child.clone());
                if changed {
                    notify_parent_changed(state, &child);
                }
            }
            zxdg_imported_v2::Request::Destroy => {}
            _ => {}
        }
    }

    fn destroyed(state: &mut D, _client: ClientId, resource: &ZxdgImportedV2, data: &XdgImportedUserData) {
        let relation = state
            .xdg_foreign_state()
            .exported
            .get_mut(&data.handle)
            .and_then(|entry| {
                entry
                    .imported_by
                    .remove(resource)
                    .map(|children| (entry.exported_surface.clone(), children))
            });
        if let Some((parent, children)) = relation {
            for child in children {
                clear_parent(state, &child, &parent, resource);
            }
        }
    }
}

#[derive(Debug)]
struct ForeignChildHook;

fn forget_child<D: XdgForeignHandler>(
    state: &mut D,
    child: &wayland_server::protocol::wl_surface::WlSurface,
) {
    for entry in state.xdg_foreign_state().exported.values_mut() {
        for children in entry.imported_by.values_mut() {
            children.remove(child);
        }
    }
}

fn revoke_export<D: XdgForeignHandler + XdgShellHandler>(
    state: &mut D,
    handle: &XdgForeignHandle,
    remove_hook: bool,
) {
    let Some(entry) = state.xdg_foreign_state().exported.remove(handle) else {
        return;
    };
    if remove_hook {
        compositor::remove_destruction_hook(&entry.exported_surface, entry.destruction_hook);
    }
    for (imported, children) in entry.imported_by {
        for child in children {
            clear_parent(state, &child, &entry.exported_surface, &imported);
        }
        if imported.is_alive() {
            imported.destroyed();
        }
    }
}

fn clear_parent<D: XdgShellHandler>(
    state: &mut D,
    child: &wayland_server::protocol::wl_surface::WlSurface,
    parent: &wayland_server::protocol::wl_surface::WlSurface,
    imported: &ZxdgImportedV2,
) {
    if !child.is_alive() {
        return;
    }
    let changed = compositor::with_states(child, |states| {
        let Some(role) = states.data_map.get::<XdgToplevelSurfaceData>() else {
            return false;
        };
        let mut role = role.lock().unwrap();
        if role.foreign_parent.as_ref() != Some(&imported.id()) || role.parent.as_ref() != Some(parent) {
            return false;
        }
        role.foreign_parent = None;
        role.parent = None;
        true
    });
    if changed {
        notify_parent_changed(state, child);
    }
}

fn notify_parent_changed<D: XdgShellHandler>(
    state: &mut D,
    child: &wayland_server::protocol::wl_surface::WlSurface,
) {
    let toplevel = state
        .xdg_shell_state()
        .toplevel_surfaces()
        .iter()
        .find(|toplevel| toplevel.wl_surface() == child)
        .cloned();
    if let Some(toplevel) = toplevel {
        XdgShellHandler::parent_changed(state, toplevel);
    }
}

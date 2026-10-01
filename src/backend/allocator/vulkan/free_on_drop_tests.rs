// GPU-gated proofs that an exported image's memory leaves the allocator's
// device when the image is dropped, while the dma-buf keeps the payload.
//
// The allocator's own VkDevice is a DRM client of this process. Its kernel
// accounting (`drm-total-*` in /proc/self/fdinfo) is the measured fact: a
// dropped image must leave that client without any further allocation.

mod free_on_drop {
    use std::collections::{BTreeMap, BTreeSet};

    use crate::backend::{
        allocator::{
            dmabuf::AsDmabuf,
            vulkan::{ImageUsageFlags, VulkanAllocator},
            Allocator, Fourcc, Modifier,
        },
        renderer::{vulkan::VulkanRenderer, ExportMem, Frame, Renderer},
        vulkan::{version::Version, Instance, PhysicalDevice},
    };
    use crate::utils::{Rectangle, Transform};

    const SIDE: u32 = 1024;
    const IMAGE_BYTES: u64 = SIDE as u64 * SIDE as u64 * 4;

    fn parse_bytes(value: &str) -> u64 {
        let mut parts = value.split_whitespace();
        let amount = parts.next().and_then(|n| n.parse::<u64>().ok()).unwrap_or(0);
        match parts.next() {
            Some("KiB") => amount * 1024,
            Some("MiB") => amount * 1024 * 1024,
            Some("GiB") => amount * 1024 * 1024 * 1024,
            _ => amount,
        }
    }

    /// `drm-client-id` -> sum of every `drm-total-<region>` for each DRM
    /// client this process holds (dup'ed fds name one client once).
    fn drm_clients() -> BTreeMap<u64, u64> {
        let mut clients = BTreeMap::new();
        for entry in std::fs::read_dir("/proc/self/fdinfo").expect("fdinfo") {
            let Ok(text) = std::fs::read_to_string(entry.expect("fdinfo entry").path()) else {
                continue;
            };
            let mut id = None;
            let mut total = 0;
            for line in text.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                if key == "drm-client-id" {
                    id = value.trim().parse::<u64>().ok();
                } else if key.starts_with("drm-total-") && !key.starts_with("drm-total-cycles") {
                    total += parse_bytes(value.trim());
                }
            }
            if let Some(id) = id {
                clients.insert(id, total);
            }
        }
        clients
    }

    fn total_of(clients: &BTreeSet<u64>) -> u64 {
        let now = drm_clients();
        clients.iter().filter_map(|id| now.get(id)).sum()
    }

    fn hardware() -> (Instance, PhysicalDevice) {
        let instance = Instance::new(Version::VERSION_1_3, None).expect("Vulkan instance");
        let physical = PhysicalDevice::enumerate(&instance)
            .expect("devices")
            .find(|device| device.render_node().ok().flatten().is_some())
            .expect("hardware Vulkan render node required");
        (instance, physical)
    }

    /// Modifiers the renderer can both render to and import, the contract
    /// every exported image in this fork's users satisfies.
    fn shared_modifiers(renderer: &VulkanRenderer) -> Vec<Modifier> {
        renderer
            .dmabuf_import_formats()
            .iter()
            .filter(|format| {
                format.code == Fourcc::Argb8888
                    && format.modifier != Modifier::Invalid
                    && renderer.has_dmabuf_render_format(**format)
            })
            .map(|format| format.modifier)
            .collect()
    }

    /// Opens the allocator after `before` and names the DRM client(s) its
    /// device created.
    fn allocator_with_clients(physical: &PhysicalDevice) -> (VulkanAllocator, BTreeSet<u64>) {
        let before = drm_clients().into_keys().collect::<BTreeSet<_>>();
        let allocator = VulkanAllocator::new(
            physical,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
        )
        .expect("allocator");
        let clients = drm_clients()
            .into_keys()
            .filter(|id| !before.contains(id))
            .collect::<BTreeSet<_>>();
        assert!(
            !clients.is_empty(),
            "the allocator's VkDevice must appear as a DRM client of this process"
        );
        (allocator, clients)
    }

    #[test]
    #[ignore = "requires a hardware Vulkan render node; run explicitly"]
    fn a_dropped_exported_image_frees_its_memory_and_the_dmabuf_still_imports() {
        let (_instance, physical) = hardware();
        let mut writer = VulkanRenderer::new(&physical).expect("renderer");
        let modifiers = shared_modifiers(&writer);
        assert!(!modifiers.is_empty());
        let (mut allocator, clients) = allocator_with_clients(&physical);
        let idle = total_of(&clients);

        let image = allocator
            .create_buffer(SIDE, SIDE, Fourcc::Argb8888, &modifiers)
            .expect("image");
        let dmabuf = image.export().expect("export");
        let allocated = total_of(&clients);
        assert!(
            allocated >= idle + IMAGE_BYTES,
            "the image is charged to the allocator's client: {idle} -> {allocated}"
        );

        // Give the payload known pixels through another device before the
        // exporter lets go of it.
        let size = (SIDE as i32, SIDE as i32).into();
        {
            let mut target = writer.bind_dmabuf_target(&dmabuf).expect("bind");
            let mut frame = writer
                .render(&mut target, size, Transform::Normal)
                .expect("frame");
            frame
                .clear([0.0, 0.0, 1.0, 1.0].into(), &[Rectangle::from_size(size)])
                .expect("clear");
            let ready = frame.finish().expect("submit");
            writer.wait(&ready).expect("written");
        }
        drop(writer);

        drop(image);
        let after_drop = total_of(&clients);
        assert!(
            after_drop + IMAGE_BYTES <= allocated,
            "dropping the image must free its memory without another allocation: \
             idle {idle}, allocated {allocated}, after drop {after_drop}"
        );

        // The dma-buf holds the kernel object: a fresh device imports it and
        // reads the pixels written before the exporter freed its memory.
        let mut reader = VulkanRenderer::new(&physical).expect("reader");
        let texture = reader.import_dmabuf_texture(&dmabuf).expect("import after free");
        let mapping = reader
            .copy_texture(&texture, Rectangle::from_size((1, 1).into()), Fourcc::Argb8888)
            .expect("readback");
        assert_eq!(reader.map_texture(&mapping).expect("map"), &[255, 0, 0, 255]);
        drop(allocator);
    }

    #[test]
    #[ignore = "requires a hardware Vulkan render node; run explicitly"]
    fn an_image_keeps_its_device_after_the_allocator_and_releases_it_last() {
        let before = drm_clients().into_keys().collect::<BTreeSet<_>>();
        let (instance, physical) = hardware();
        // The driver's physical-device fds: they close only when the
        // instance is destroyed.
        let instance_clients = drm_clients()
            .into_keys()
            .filter(|id| !before.contains(id))
            .collect::<BTreeSet<_>>();
        assert!(
            !instance_clients.is_empty(),
            "the instance's physical device must appear as a DRM client of this process"
        );
        let modifiers = {
            let renderer = VulkanRenderer::new(&physical).expect("renderer");
            shared_modifiers(&renderer)
        };
        let (mut allocator, clients) = allocator_with_clients(&physical);
        let image = allocator
            .create_buffer(64, 64, Fourcc::Argb8888, &modifiers)
            .expect("image");

        // Every other holder of the instance goes first, so the allocator is
        // its last caller-side owner.
        drop(physical);
        drop(instance);
        drop(allocator);
        let instance_lives = instance_clients.iter().all(|id| drm_clients().contains_key(id));
        if !instance_lives {
            // Using or dropping the image now would reach its device after
            // its instance was destroyed; leak it so this panic is the report.
            std::mem::forget(image);
            panic!("the instance lives while an image holds its device");
        }
        let dmabuf = image.export().expect("an image outlives its allocator");
        assert!(
            clients.iter().all(|id| drm_clients().contains_key(id)),
            "the device lives while an image holds it"
        );

        drop(image);
        assert!(
            clients.iter().all(|id| !drm_clients().contains_key(id)),
            "the last image destroys the device"
        );
        assert!(
            instance_clients.iter().all(|id| !drm_clients().contains_key(id)),
            "the instance is destroyed after the last image's device"
        );
        drop(dmabuf);
    }
}

//! Probe: can a Venus(Vulkan)-allocated dmabuf be scanned out by virtio-gpu KMS?
//!
//! Allocates a linear image via `VulkanAllocator` on the device matching the
//! given DRM node, exports it as a dmabuf, PRIME-imports it into the KMS fd
//! (raw kernel path — deliberately no GBM/Mesa involvement), creates a legacy
//! (modifier-less) AddFB2 framebuffer, and performs a legacy modeset on the
//! first connected connector. Optionally fills the buffer orange via dmabuf
//! mmap when the exporter allows CPU mapping.
//!
//! Run as root on a VT with no other DRM master (e.g. `systemctl stop sddm`).

use std::fs::{File, OpenOptions};
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::fs::OpenOptionsExt;

use drm::buffer::PlanarBuffer;
use drm::control::{connector, Device as ControlDevice, FbCmd2Flags};
use drm::Device as DrmBaseDevice;
use smithay::backend::allocator::{
    dmabuf::{AsDmabuf, Dmabuf},
    vulkan::{ImageUsageFlags, VulkanAllocator},
    Allocator, Buffer, Fourcc, Modifier,
};
use smithay::backend::drm::DrmNode;
use smithay::backend::vulkan::{version::Version, Instance, PhysicalDevice};

struct Card(File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl DrmBaseDevice for Card {}
impl ControlDevice for Card {}

struct ProbeFb {
    size: (u32, u32),
    format: Fourcc,
    pitch: u32,
    offset: u32,
    handle: drm::buffer::Handle,
}

impl PlanarBuffer for ProbeFb {
    fn size(&self) -> (u32, u32) {
        self.size
    }
    fn format(&self) -> Fourcc {
        self.format
    }
    fn modifier(&self) -> Option<drm::buffer::DrmModifier> {
        None
    }
    fn pitches(&self) -> [u32; 4] {
        [self.pitch, 0, 0, 0]
    }
    fn handles(&self) -> [Option<drm::buffer::Handle>; 4] {
        [Some(self.handle), None, None, None]
    }
    fn offsets(&self) -> [u32; 4] {
        [self.offset, 0, 0, 0]
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/dev/dri/card0".into());
    let fourcc = Fourcc::Xrgb8888;

    let card = Card(
        OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(&path)
            .expect("open drm device"),
    );
    println!("[1] opened {path}");

    // Pick first connected connector + preferred mode + a CRTC.
    let res = card.resource_handles().expect("resource handles");
    let (conn, mode) = res
        .connectors()
        .iter()
        .filter_map(|c| card.get_connector(*c, false).ok())
        .find(|c| c.state() == connector::State::Connected && !c.modes().is_empty())
        .map(|c| {
            let mode = c
                .modes()
                .iter()
                .find(|m| m.mode_type().contains(drm::control::ModeTypeFlags::PREFERRED))
                .copied()
                .unwrap_or(c.modes()[0]);
            (c, mode)
        })
        .expect("no connected connector");
    let crtc = res.crtcs()[0];
    let (w, h) = (mode.size().0 as u32, mode.size().1 as u32);
    println!(
        "[2] connector {:?} mode {}x{} crtc {:?}",
        conn.interface(),
        w,
        h,
        crtc
    );

    // Vulkan allocation on the device matching this DRM node.
    let node = DrmNode::from_file(card.as_fd()).expect("drm node");
    let instance = Instance::new(Version::VERSION_1_2, None).expect("vulkan instance");
    let phd = PhysicalDevice::enumerate(&instance)
        .expect("enumerate")
        .filter(|phd| phd.has_device_extension(c"VK_EXT_physical_device_drm"))
        .find(|phd| {
            phd.primary_node().unwrap() == Some(node) || phd.render_node().unwrap() == Some(node)
        })
        .expect("no vulkan device for node");
    println!("[3] vulkan device: {:?}", phd.name());

    let mut allocator = VulkanAllocator::new(
        &phd,
        ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
    )
    .expect("vulkan allocator");
    let image = allocator
        .create_buffer(w, h, fourcc, &[Modifier::Linear])
        .expect("allocate image");
    let dmabuf: Dmabuf = image.export().expect("export dmabuf");
    println!(
        "[4] allocated {}x{} {:?} modifier {:?} planes {}",
        w,
        h,
        dmabuf.format().code,
        dmabuf.format().modifier,
        dmabuf.num_planes()
    );

    let plane_fd = dmabuf.handles().next().expect("plane fd");
    let stride = dmabuf.strides().next().unwrap();
    let offset = dmabuf.offsets().next().unwrap();

    // Optional CPU fill (orange) via dmabuf mmap.
    unsafe {
        let len = (stride as usize) * (h as usize);
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_WRITE,
            libc::MAP_SHARED,
            plane_fd.as_fd().as_raw_fd(),
            offset as i64,
        );
        if ptr == libc::MAP_FAILED {
            println!(
                "[5] dmabuf mmap not supported ({}); continuing without fill",
                std::io::Error::last_os_error()
            );
        } else {
            let px = ptr as *mut u32;
            for i in 0..(len / 4) {
                px.add(i).write(0x00FF7F00); // orange XRGB
            }
            libc::munmap(ptr, len);
            println!("[5] filled buffer orange via mmap");
        }
    }

    // PRIME import into KMS fd (raw kernel, no GBM).
    let handle = card
        .prime_fd_to_buffer(plane_fd.as_fd())
        .expect("prime fd -> gem handle");
    println!("[6] PRIME imported: handle {handle:?}");

    let fb = card
        .add_planar_framebuffer(
            &ProbeFb {
                size: (w, h),
                format: fourcc,
                pitch: stride,
                offset,
                handle,
            },
            FbCmd2Flags::empty(),
        )
        .expect("AddFB2");
    println!("[7] AddFB2 ok: {fb:?}");

    card.set_crtc(crtc, Some(fb), (0, 0), &[conn.handle()], Some(mode))
        .expect("set_crtc modeset");
    println!("[8] modeset ok — holding for 8s; the display should be ORANGE");

    std::thread::sleep(std::time::Duration::from_secs(8));
    println!("PROBE PASS");
}

use std::os::fd::AsRawFd;

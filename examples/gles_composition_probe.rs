//! Native-driver validation for the strict GLES compositor. Run in the VM only.
use smithay::{
    backend::{
        allocator::{
            dmabuf::AsDmabuf,
            gbm::{ExplicitGbmAllocator, GbmBufferFlags},
            Allocator, Fourcc, Modifier,
        },
        egl::{ffi::egl, EGLDisplay},
        renderer::{
            gles::{
                ffi as gl, GlesDeviceOrigin, GlesKawaseOutput, GlesKawasePass, GlesRenderer, GlesTexture,
            },
            sync::{Fence, Interrupted, SyncPoint},
            Bind, Blit, Color32F, ExportMem, Frame, ImportMem, Renderer, RoundedClip, TextureFilter,
            TextureRenderEffect, TextureRenderEffectKind,
        },
    },
    utils::{Buffer, Physical, Rectangle, Transform},
};
use std::{error::Error, fs::OpenOptions, io::Write, os::fd::OwnedFd};

#[derive(Debug)]
struct NotNative;
impl Fence for NotNative {
    fn is_signaled(&self) -> bool {
        false
    }
    fn wait(&self) -> Result<(), Interrupted> {
        panic!("strict path performed a CPU wait")
    }
    fn is_exportable(&self) -> bool {
        false
    }
    fn export(&self) -> Option<OwnedFd> {
        None
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut render_node = None;
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--render-node" => {
                index += 1;
                render_node = args.get(index).cloned();
            }
            "--modifier" => {
                index += 1;
                assert_eq!(args.get(index).map(String::as_str), Some("0x0"));
            }
            "--allocation-api" => {
                index += 1;
                assert_eq!(args.get(index).map(String::as_str), Some("linear-usage"));
            }
            "--software-diagnostic" => {}
            arg => panic!("unsupported argument: {arg}"),
        }
        index += 1;
    }
    let node = render_node.ok_or("missing render node")?;
    let file = std::sync::Arc::new(OpenOptions::new().read(true).write(true).open(&node)?);
    let gbm = gbm::Device::new(file)?;
    let mut allocator =
        ExplicitGbmAllocator::new(gbm.clone(), GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT);
    let display = unsafe { EGLDisplay::new(gbm)? };
    let origin = GlesDeviceOrigin::new(&display)?;
    let helper_origin = origin.clone();
    let bank = std::thread::spawn(move || {
        let mut helper = helper_origin.create_renderer().map_err(|e| e.to_string())?;
        helper.prepare_composition_programs().map_err(|e| e.to_string())
    })
    .join()
    .expect("helper panicked")
    .map_err(std::io::Error::other)?;
    let mut renderer = origin.create_renderer()?;
    renderer.adopt_composition_programs(bank.clone())?;
    renderer.prepare_damage_scratch(4, 16)?;
    renderer.reserve_composition_framebuffers(8)?;
    println!("SMITHAY_GLES_PROGRAMS=PASS");
    let output = renderer.create_composition_buffer(Fourcc::Abgr8888, (32, 32).into())?;
    let prefix = renderer.create_composition_buffer(Fourcc::Abgr8888, (32, 32).into())?;
    let half = renderer.create_composition_buffer(Fourcc::Abgr8888, (16, 16).into())?;
    let atlas = renderer.create_composition_buffer(Fourcc::Abgr8888, (64, 64).into())?;
    renderer.prepare_composition_textures(&[&output, &prefix, &half, &atlas])?;
    let white = renderer.import_memory(&[128; 4 * 32 * 32], Fourcc::Abgr8888, (32, 32).into(), false)?;
    let opaque = renderer.import_memory(&[255; 4 * 32 * 32], Fourcc::Abgr8888, (32, 32).into(), false)?;
    let display_handle = **display.get_display_handle();
    let context_handle = renderer.egl_context().get_context_handle();
    let external_name = renderer.with_context(|gl| unsafe {
        let image = egl::CreateImageKHR(
            display_handle,
            context_handle,
            egl::GL_TEXTURE_2D,
            opaque.tex_id() as usize as *mut _,
            [egl::GL_TEXTURE_LEVEL as i32, 0, egl::NONE as i32].as_ptr(),
        );
        assert_ne!(image, egl::NO_IMAGE_KHR, "external source EGL image creation");
        let mut name = 0;
        gl.GenTextures(1, &mut name);
        gl.BindTexture(gl::TEXTURE_EXTERNAL_OES, name);
        gl.EGLImageTargetTexture2DOES(gl::TEXTURE_EXTERNAL_OES, image);
        gl.BindTexture(gl::TEXTURE_EXTERNAL_OES, 0);
        assert_eq!(gl.GetError(), gl::NO_ERROR, "external source image attachment");
        assert_ne!(egl::DestroyImageKHR(display_handle, image), egl::FALSE);
        name
    })?;
    // SAFETY: The current native context created this external texture above;
    // opaque retains the image's allocation through the probe's final fence.
    let external =
        unsafe { GlesTexture::from_raw_external(&renderer, true, external_name, (32, 32).into(), false) };
    let passes = [GlesKawasePass::new(&prefix, &half, false, 1.0).with_encoded_srgb()];
    renderer.prepare_kawase_passes(&passes)?;
    let rect = Rectangle::<i32, Physical>::from_size((32, 32).into());
    let clip = RoundedClip {
        rect: rect.to_f64(),
        radius: 10.0,
        exponent: 2.0,
        aa_width: 0.5,
        corner_mask: RoundedClip::ALL_CORNERS,
    };
    assert!(
        output.is_externally_idle(),
        "weak native inventory retained a strong texture reader"
    );
    let terminal = renderer.import_memory(
        &[0u8, 255, 0, 255].repeat(32 * 32),
        Fourcc::Abgr8888,
        (32, 32).into(),
        false,
    )?;
    renderer.prepare_framebuffer_textures(&[&terminal])?;
    let mut terminal_target = renderer.bind_prepared_texture(&terminal)?;
    let mut target = renderer.bind_prepared_texture(&output)?;
    let snapshot = renderer.submission_snapshot();
    let mut frame = renderer.render(&mut target, (32, 32).into(), Transform::Normal)?;
    frame.clear(Color32F::BLACK, &[rect])?;
    Frame::render_texture_from_to(
        &mut frame,
        &white,
        Rectangle::from_size((32.0, 32.0).into()),
        rect,
        &[rect],
        &[],
        Transform::Normal,
        1.0,
    )?;
    let patch = Rectangle::new((8, 8).into(), (8, 8).into());
    let completion = frame.finish_into(&mut terminal_target, &[patch])?;
    assert!(completion.is_exportable());
    assert!(renderer.completion_since(snapshot).is_some());
    // This standalone cold validation program may wait/read back evidence.
    completion.wait()?;
    let mapping =
        renderer.copy_framebuffer(&target, Rectangle::from_size((32, 32).into()), Fourcc::Abgr8888)?;
    let pixels = renderer.map_texture(&mapping)?;
    assert!(
        (186..=190).contains(&pixels[0]),
        "linear-light blend channel={}",
        pixels[0]
    );
    println!("SMITHAY_GLES_LINEAR_BLEND=PASS channel={}", pixels[0]);
    let mapping = renderer.copy_framebuffer(
        &terminal_target,
        Rectangle::from_size((32, 32).into()),
        Fourcc::Abgr8888,
    )?;
    let terminal_pixels = renderer.map_texture(&mapping)?;
    assert_eq!(
        &terminal_pixels[..4],
        &[0, 255, 0, 255],
        "terminal resolve overwrote pixels outside damage"
    );
    assert!(
        (186..=190).contains(&terminal_pixels[(8 * 32 + 8) * 4]),
        "terminal resolve did not encode linear composition"
    );
    println!("SMITHAY_GLES_TERMINAL_ENCODING_DAMAGE=PASS");
    let mut frame = renderer.render(&mut target, (32, 32).into(), Transform::Normal)?;
    frame.clear(Color32F::BLACK, &[rect])?;
    frame.render_texture_from_to_with_rounded_clip(
        &external,
        Rectangle::from_size((32.0, 32.0).into()),
        rect,
        &[rect],
        &[],
        Transform::Normal,
        1.0,
        clip,
    )?;
    frame.capture_and_filter_framebuffer(rect, &prefix, &passes)?;
    frame.render_kawase_with_rounded_clip(
        &half,
        Rectangle::from_size((16.0, 16.0).into()),
        rect,
        &[rect],
        Transform::Normal,
        1.0,
        clip,
        GlesKawaseOutput::new((16, 16).into(), 1.0),
    )?;
    frame.render_owner_clipped_group(&atlas, rect, rect, clip, &[rect], |frame| {
        assert!(frame.canonical_coverage_lane().is_some());
        frame.render_texture_from_to_with_rounded_clip_and_effect(
            &opaque,
            Rectangle::from_size((32.0, 32.0).into()),
            rect,
            &[rect],
            &[],
            Transform::Normal,
            0.25,
            clip,
            TextureRenderEffect {
                kind: TextureRenderEffectKind::Fullscreen,
                progress: 0.5,
                anchor: [0.5, 0.5],
                params: [0.0, 0.1, 0.0, 0.0],
            },
        )
    })?;
    frame.finish()?.wait()?;
    let mapping =
        renderer.copy_framebuffer(&target, Rectangle::from_size((32, 32).into()), Fourcc::Abgr8888)?;
    let pixels = renderer.map_texture(&mapping)?;
    assert!(pixels[0] < 8, "corner escaped clip: {}", pixels[0]);
    assert!(pixels[(16 * 32 + 16) * 4] > 240, "center lost content");
    println!("SMITHAY_GLES_EFFECTS_OWNER_KAWASE_EXTERNAL=PASS");
    drop(target);
    drop(terminal_target);
    verify_encoded_dma_transfer_and_transforms(&mut renderer, &mut allocator)?;
    verify_inverted_source_crops(&mut renderer)?;
    let source_target = renderer.bind_prepared_texture(&output)?;
    let mut prefix_target = renderer.bind_prepared_texture(&prefix)?;
    renderer
        .blit(
            &source_target,
            &mut prefix_target,
            rect,
            rect,
            TextureFilter::Nearest,
        )?
        .wait()?;
    let mapping = renderer.copy_framebuffer(
        &prefix_target,
        Rectangle::from_size((32, 32).into()),
        Fourcc::Abgr8888,
    )?;
    let pixels = renderer.map_texture(&mapping)?;
    assert!(
        pixels[0] < 8 && pixels[(16 * 32 + 16) * 4] > 240,
        "blit lost prior readback's source attachment"
    );
    drop(source_target);
    drop(prefix_target);
    println!("SMITHAY_GLES_BLIT_AFTER_READBACK=PASS");
    let excessive = (0..5)
        .map(|_| renderer.create_composition_buffer(Fourcc::Abgr8888, (4, 4).into()))
        .collect::<Result<Vec<_>, _>>()?;
    let references: Vec<_> = excessive.iter().collect();
    assert!(renderer.can_prepare_framebuffer_textures(&references).is_err());
    assert!(renderer.check_completion_health().is_ok());
    println!("SMITHAY_GLES_BOUNDED_INVENTORY=PASS");
    let mut second = origin.create_renderer()?;
    second.adopt_composition_programs(bank)?;
    second.reserve_composition_framebuffers(3)?;
    let mut second_texture = second.create_composition_buffer(Fourcc::Abgr8888, (32, 32).into())?;
    second.prepare_composition_textures(&[&second_texture])?;
    let mut second_target = second.bind(&mut second_texture)?;
    second.cleanup_texture_cache()?;
    assert!(renderer
        .render(&mut second_target, (32, 32).into(), Transform::Normal)
        .is_err());
    let mut frame = second.render(&mut second_target, (32, 32).into(), Transform::Normal)?;
    frame.clear(Color32F::BLACK, &[rect])?;
    frame.finish()?.wait()?;
    drop(second_target);
    println!("SMITHAY_GLES_SHARED_PROGRAMS_CONTEXT_FBO=PASS");
    let fp16 = second.create_composition_buffer(Fourcc::Abgr16161616f, (32, 32).into())?;
    second.prepare_composition_textures(&[&fp16])?;
    let mut fp16_target = second.bind_prepared_texture(&fp16)?;
    let mut frame = second.render(&mut fp16_target, (32, 32).into(), Transform::Normal)?;
    frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &[rect])?;
    Frame::render_texture_from_to(
        &mut frame,
        &white,
        Rectangle::from_size((32.0, 32.0).into()),
        rect,
        &[rect],
        &[],
        Transform::Normal,
        1.0,
    )?;
    frame.finish()?.wait()?;
    // ES requires RGBA/FLOAT readback for floating color attachments. Keep
    // this cold probe independent of optional HALF_FLOAT readback extensions.
    let mut pixels = vec![0.0f32; 32 * 32 * 4];
    second.with_context(|gl| unsafe {
        gl.ReadBuffer(gl::COLOR_ATTACHMENT0);
        gl.ReadPixels(0, 0, 32, 32, gl::RGBA, gl::FLOAT, pixels.as_mut_ptr().cast());
        assert_eq!(gl.GetError(), gl::NO_ERROR, "floating attachment readback");
    })?;
    let channel = pixels[0];
    assert!(
        (channel - 128.0 / 255.0).abs() < 0.002,
        "float encoded-blend policy {channel}",
    );
    assert!((pixels[3] - 128.0 / 255.0).abs() < 0.002);
    let fp16_terminal =
        second.import_memory(&[0u8; 32 * 32 * 4], Fourcc::Abgr8888, (32, 32).into(), false)?;
    second.prepare_framebuffer_textures(&[&fp16_terminal])?;
    let mut fp16_destination = second.bind_prepared_texture(&fp16_terminal)?;
    let frame = second.render(&mut fp16_target, (32, 32).into(), Transform::Normal)?;
    frame.finish_into(&mut fp16_destination, &[rect])?.wait()?;
    let mapping = second.copy_framebuffer(
        &fp16_destination,
        Rectangle::from_size((32, 32).into()),
        Fourcc::Abgr8888,
    )?;
    let pixels = second.map_texture(&mapping)?;
    assert_pixel(&pixels[..4], [128; 4], "floating electrical-alpha terminal");
    drop(mapping);
    let mut linear_target = second.bind_prepared_texture(&second_texture)?;
    let mut frame = second.render(&mut linear_target, (32, 32).into(), Transform::Normal)?;
    frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &[rect])?;
    Frame::render_texture_from_to(
        &mut frame,
        &white,
        Rectangle::from_size((32.0, 32.0).into()),
        rect,
        &[rect],
        &[],
        Transform::Normal,
        1.0,
    )?;
    frame.finish()?.wait()?;
    second
        .blit(
            &linear_target,
            &mut fp16_target,
            rect,
            rect,
            TextureFilter::Nearest,
        )?
        .wait()?;
    let frame = second.render(&mut fp16_target, (32, 32).into(), Transform::Normal)?;
    frame.finish_into(&mut fp16_destination, &[rect])?.wait()?;
    let mapping = second.copy_framebuffer(
        &fp16_destination,
        Rectangle::from_size((32, 32).into()),
        Fourcc::Abgr8888,
    )?;
    let pixels = second.map_texture(&mapping)?;
    assert_pixel(&pixels[..4], [128; 4], "linear 8888 to electrical F16 transfer");
    drop(mapping);
    drop(linear_target);
    drop(fp16_destination);
    drop(fp16_target);
    println!("SMITHAY_GLES_FLOAT_COMPOSITION=PASS");
    let mut exhausted = origin.create_renderer()?;
    exhausted.reserve_native_completions(1)?;
    let before = exhausted.submission_snapshot();
    let held = exhausted.publish_shared_work()?;
    assert!(held.is_exportable());
    let after = exhausted.submission_snapshot();
    assert!(exhausted.publish_shared_work().is_err());
    assert!(exhausted.check_completion_health().is_err());
    assert!(exhausted.completion_since(before).is_some());
    assert!(exhausted.completion_since(after).is_none());
    // Releasing a reader never rehabilitates an epoch that already emitted
    // commands without an observable native completion.
    held.wait()?;
    drop(held);
    assert!(exhausted.publish_shared_work().is_err());
    println!("SMITHAY_GLES_COMPLETION_EXHAUSTION=PASS");
    let wait = renderer.wait(&SyncPoint::from(NotNative));
    assert!(wait.is_err());
    assert!(renderer.check_completion_health().is_err());
    println!("SMITHAY_GLES_NATIVE_REFUSAL=PASS");
    println!("SMITHAY_GLES_PROBE=PASS");
    Ok(())
}

fn verify_encoded_dma_transfer_and_transforms(
    renderer: &mut GlesRenderer,
    allocator: &mut ExplicitGbmAllocator<std::sync::Arc<std::fs::File>>,
) -> Result<(), Box<dyn Error>> {
    const WIDTH: i32 = 44;
    const HEIGHT: i32 = 28;
    let size = (WIDTH, HEIGHT).into();
    let full = Rectangle::<i32, Physical>::from_size(size);
    let bo = allocator.create_buffer(WIDTH as u32, HEIGHT as u32, Fourcc::Argb8888, &[Modifier::Linear])?;
    let dmabuf = bo.export()?;
    let encoded = renderer.import_dmabuf_2d(&dmabuf)?;
    let scratch = renderer.create_composition_buffer(Fourcc::Abgr8888, (WIDTH, HEIGHT).into())?;
    let terminal = renderer.import_memory(
        &[0u8, 255, 0, 255].repeat((WIDTH * HEIGHT) as usize),
        Fourcc::Abgr8888,
        (WIDTH, HEIGHT).into(),
        false,
    )?;
    renderer.prepare_framebuffer_textures(&[&encoded, &scratch, &terminal])?;
    let mut source = renderer.bind_prepared_texture(&encoded)?;
    let mut frame = renderer.render(&mut source, size, Transform::Normal)?;
    frame.with_context(|gl| unsafe {
        gl.Disable(gl::BLEND);
        gl.Enable(gl::SCISSOR_TEST);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let pixel = pattern(x, y);
                gl.ClearColor(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                    pixel[3] as f32 / 255.0,
                );
                gl.Scissor(x, y, 1, 1);
                gl.Clear(gl::COLOR_BUFFER_BIT);
            }
        }
        gl.Disable(gl::SCISSOR_TEST);
        assert_eq!(gl.GetError(), gl::NO_ERROR, "encoded DMA pattern initialization");
    })?;
    frame.finish()?.wait()?;
    let mut scratch_target = renderer.bind_prepared_texture(&scratch)?;
    let mut destination = renderer.bind_prepared_texture(&terminal)?;
    let mut frame = renderer.render(&mut scratch_target, size, Transform::Normal)?;
    frame.clear(Color32F::new(0.0, 1.0, 0.0, 1.0), &[full])?;
    frame.finish()?.wait()?;
    let copy_src = Rectangle::new((12, 5).into(), (10, 9).into());
    let copy_dst = Rectangle::new((4, 7).into(), (10, 9).into());
    renderer
        .blit(
            &source,
            &mut scratch_target,
            copy_src,
            copy_dst,
            TextureFilter::Nearest,
        )?
        .wait()?;
    let frame = renderer.render(&mut scratch_target, size, Transform::Normal)?;
    frame.finish_into(&mut destination, &[copy_dst])?.wait()?;
    let mapping = renderer.copy_framebuffer(
        &destination,
        Rectangle::from_size((WIDTH, HEIGHT).into()),
        Fourcc::Abgr8888,
    )?;
    let pixels = renderer.map_texture(&mapping)?;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let expected = if copy_dst.contains((x, y)) {
                pattern(
                    x - copy_dst.loc.x + copy_src.loc.x,
                    y - copy_dst.loc.y + copy_src.loc.y,
                )
            } else {
                [0, 255, 0, 255]
            };
            assert_pixel(
                &pixels[((y * WIDTH + x) * 4) as usize..][..4],
                expected,
                "partial encoded DMA transfer",
            );
        }
    }
    drop(mapping);
    renderer
        .blit(&source, &mut scratch_target, full, full, TextureFilter::Nearest)?
        .wait()?;
    let frame = renderer.render(&mut scratch_target, size, Transform::Normal)?;
    frame.finish_into(&mut destination, &[full])?.wait()?;
    let mapping = renderer.copy_framebuffer(
        &destination,
        Rectangle::from_size((WIDTH, HEIGHT).into()),
        Fourcc::Abgr8888,
    )?;
    let pixels = renderer.map_texture(&mapping)?;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            assert_pixel(
                &pixels[((y * WIDTH + x) * 4) as usize..][..4],
                pattern(x, y),
                "encoded DMA transfer",
            );
        }
    }
    drop(mapping);
    let overlay = Rectangle::new((9, 5).into(), (7, 11).into());
    let mut frame = renderer.render(&mut scratch_target, size, Transform::Normal)?;
    let left = Rectangle::new(overlay.loc, (3, overlay.size.h).into());
    let right = Rectangle::new(
        (overlay.loc.x + 3, overlay.loc.y).into(),
        (4, overlay.size.h).into(),
    );
    Frame::draw_solid(
        &mut frame,
        left,
        &[Rectangle::from_size(left.size)],
        Color32F::new(1.0, 0.0, 0.0, 0.5),
    )?;
    frame.draw_solid(
        right,
        &[Rectangle::from_size(right.size)],
        Color32F::new(1.0, 0.0, 0.0, 0.5),
    )?;
    frame.finish_into(&mut destination, &[overlay])?.wait()?;
    let mapping = renderer.copy_framebuffer(
        &destination,
        Rectangle::from_size((WIDTH, HEIGHT).into()),
        Fourcc::Abgr8888,
    )?;
    let pixels = renderer.map_texture(&mapping)?;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let mut expected = pattern(x, y);
            if overlay.contains((x, y)) {
                let alpha = expected[3] as f64 / 255.0;
                let output_alpha = 0.5 + alpha * 0.5;
                for channel in 0..3 {
                    let encoded = if alpha > 0.0 {
                        expected[channel] as f64 / 255.0 / alpha
                    } else {
                        0.0
                    };
                    let linear = if encoded <= 0.04045 {
                        encoded / 12.92
                    } else {
                        ((encoded + 0.055) / 1.055).powf(2.4)
                    };
                    let mixed = (linear * alpha * 0.5 + if channel == 0 { 0.5 } else { 0.0 }) / output_alpha;
                    let encoded = if mixed <= 0.0031308 {
                        mixed * 12.92
                    } else {
                        1.055 * mixed.powf(1.0 / 2.4) - 0.055
                    };
                    expected[channel] = (encoded * output_alpha * 255.0).round() as u8;
                }
                expected[3] = (output_alpha * 255.0).round() as u8;
            }
            assert_pixel(
                &pixels[((y * WIDTH + x) * 4) as usize..][..4],
                expected,
                "overlay history",
            );
        }
    }
    drop(mapping);
    println!("SMITHAY_GLES_ENCODED_DMA_TRANSFER_OVERLAY=PASS");
    for (transform, finish_frame) in [
        Transform::Normal,
        Transform::_90,
        Transform::_180,
        Transform::_270,
        Transform::Flipped,
        Transform::Flipped90,
        Transform::Flipped180,
        Transform::Flipped270,
    ]
    .into_iter()
    .flat_map(|transform| [(transform, false), (transform, true)])
    {
        renderer
            .blit(&source, &mut scratch_target, full, full, TextureFilter::Nearest)?
            .wait()?;
        // Temporarily release the destination's write guard for the memory update.
        drop(destination);
        renderer.update_memory(
            &terminal,
            &[0u8, 255, 0, 255].repeat((WIDTH * HEIGHT) as usize),
            Rectangle::from_size((WIDTH, HEIGHT).into()),
        )?;
        destination = renderer.bind_prepared_texture(&terminal)?;
        let damage = Rectangle::new((3, 7).into(), (5, 9).into());
        let logical_size = transform.transform_size(size);
        let native = transform.transform_rect_in(damage, &logical_size);
        let completion = if finish_frame {
            let mut frame = renderer.render(&mut scratch_target, size, transform)?;
            frame.clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &[damage])?;
            frame.finish_into_transformed(&mut destination, &[damage], transform, logical_size)?
        } else {
            // The separate resolve cannot sample a texture with its write target bound.
            drop(scratch_target);
            let receipt = renderer.resolve_composition_transformed(
                &scratch,
                &mut destination,
                &[damage],
                transform,
                logical_size,
            )?;
            scratch_target = renderer.bind_prepared_texture(&scratch)?;
            receipt
        };
        completion.wait()?;
        let mapping = renderer.copy_framebuffer(
            &destination,
            Rectangle::from_size((WIDTH, HEIGHT).into()),
            Fourcc::Abgr8888,
        )?;
        let pixels = renderer.map_texture(&mapping)?;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let expected = if !native.contains((x, y)) {
                    [0, 255, 0, 255]
                } else if finish_frame {
                    [255, 0, 0, 255]
                } else {
                    pattern(x, y)
                };
                assert_pixel(
                    &pixels[((y * WIDTH + x) * 4) as usize..][..4],
                    expected,
                    "nonsquare transformed damage",
                );
            }
        }
    }
    println!("SMITHAY_GLES_TRANSFORMED_DAMAGE=PASS");
    Ok(())
}

// Exercise the actual shared fragment shader, not a CPU copy of its formula.
// Both inputs describe the same logical image; only physical row order and its
// declared storage flag differ. A nonzero asymmetric crop catches a local-UV
// flip that full-image comparisons cannot distinguish from whole-image origin.
fn verify_inverted_source_crops(renderer: &mut GlesRenderer) -> Result<(), Box<dyn Error>> {
    const WIDTH: i32 = 44;
    const HEIGHT: i32 = 28;
    const OUTPUT_WIDTH: i32 = 48;
    const OUTPUT_HEIGHT: i32 = 40;
    let color = |x: i32, y: i32| [16 + x as u8 * 5, 20 + y as u8 * 7, 96, 255];
    let pixels: Vec<u8> = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).flat_map(move |x| color(x, y)))
        .collect();
    let inverted_pixels: Vec<u8> = pixels
        .chunks_exact(WIDTH as usize * 4)
        .rev()
        .flatten()
        .copied()
        .collect();
    let normal = renderer.import_memory(&pixels, Fourcc::Abgr8888, (WIDTH, HEIGHT).into(), false)?;
    let inverted =
        renderer.import_memory(&inverted_pixels, Fourcc::Abgr8888, (WIDTH, HEIGHT).into(), true)?;
    assert!(!normal.is_y_inverted());
    assert!(inverted.is_y_inverted());
    let size = (OUTPUT_WIDTH, OUTPUT_HEIGHT).into();
    let full = Rectangle::<i32, Physical>::from_size(size);
    let scratch =
        renderer.create_composition_buffer(Fourcc::Abgr8888, (OUTPUT_WIDTH, OUTPUT_HEIGHT).into())?;
    let terminal = renderer.import_memory(
        &[0u8, 255, 0, 255].repeat((OUTPUT_WIDTH * OUTPUT_HEIGHT) as usize),
        Fourcc::Abgr8888,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT).into(),
        false,
    )?;
    renderer.prepare_framebuffer_textures(&[&scratch, &terminal])?;
    let mut scratch_target = renderer.bind_prepared_texture(&scratch)?;
    let mut terminal_target = renderer.bind_prepared_texture(&terminal)?;
    let cropped = Rectangle::<i32, Buffer>::new((5, 3).into(), (11, 7).into());
    let cases = std::iter::once((
        Rectangle::from_size((WIDTH, HEIGHT).into()),
        Transform::Normal,
        false,
    ))
    .chain(
        [
            Transform::Normal,
            Transform::_90,
            Transform::_180,
            Transform::_270,
            Transform::Flipped,
            Transform::Flipped90,
            Transform::Flipped180,
            Transform::Flipped270,
        ]
        .into_iter()
        .flat_map(|transform| [(cropped, transform, false), (cropped, transform, true)]),
    );
    for (src, transform, clipped) in cases {
        let extent = transform.transform_size(src.size);
        let dst = Rectangle::<i32, Physical>::new((2, 3).into(), (extent.w, extent.h).into());
        let damage = [Rectangle::from_size(dst.size)];
        let mut reference: Option<Vec<u8>> = None;
        for source in [&normal, &inverted] {
            let mut frame = renderer.render(&mut scratch_target, size, Transform::Normal)?;
            frame.clear(Color32F::new(0.0, 1.0, 0.0, 1.0), &[full])?;
            if clipped {
                frame.render_texture_from_to_with_rounded_clip(
                    source,
                    src.to_f64(),
                    dst,
                    &damage,
                    &[],
                    transform,
                    1.0,
                    RoundedClip {
                        rect: dst.to_f64(),
                        radius: 2.0,
                        exponent: 2.0,
                        aa_width: 0.5,
                        corner_mask: RoundedClip::ALL_CORNERS,
                    },
                )?;
            } else {
                Frame::render_texture_from_to(
                    &mut frame,
                    source,
                    src.to_f64(),
                    dst,
                    &damage,
                    &[],
                    transform,
                    1.0,
                )?;
            }
            let completion = frame.finish_into(&mut terminal_target, &[full])?;
            assert!(completion.is_exportable(), "crop render lacks native completion");
            completion.wait()?;
            let mapping = renderer.copy_framebuffer(
                &terminal_target,
                Rectangle::from_size((OUTPUT_WIDTH, OUTPUT_HEIGHT).into()),
                Fourcc::Abgr8888,
            )?;
            let actual = renderer.map_texture(&mapping)?.to_vec();
            drop(mapping);
            for y in 0..OUTPUT_HEIGHT {
                for x in 0..OUTPUT_WIDTH {
                    let pixel = &actual[((y * OUTPUT_WIDTH + x) * 4) as usize..][..4];
                    if !dst.contains((x, y)) {
                        assert_pixel(pixel, [0, 255, 0, 255], "crop escaped destination");
                    } else if transform == Transform::Normal && !clipped {
                        assert_pixel(
                            pixel,
                            color(x - dst.loc.x + src.loc.x, y - dst.loc.y + src.loc.y),
                            "logical source crop",
                        );
                    }
                }
            }
            if let Some(reference) = &reference {
                assert!(
                    actual.iter().zip(reference).all(|(a, b)| a.abs_diff(*b) <= 2),
                    "storage orientation changed logical crop: {src:?} {transform:?} clipped={clipped}"
                );
            } else {
                reference = Some(actual);
            }
        }
    }
    writeln!(
        std::io::stdout(),
        "SMITHAY_GLES_Y_INVERTED_SOURCE_CROP=PASS cases=17"
    )?;
    Ok(())
}

fn pattern(x: i32, y: i32) -> [u8; 4] {
    let alpha = [0, 64, 128, 255][(x / 11) as usize];
    [
        ((32 + x * 3) * alpha / 255) as u8,
        ((24 + y * 5) * alpha / 255) as u8,
        ((96 + x + y) * alpha / 255) as u8,
        alpha as u8,
    ]
}

fn assert_pixel(actual: &[u8], expected: [u8; 4], operation: &str) {
    assert!(
        actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 2),
        "{operation}: {actual:?} != {expected:?}"
    );
}

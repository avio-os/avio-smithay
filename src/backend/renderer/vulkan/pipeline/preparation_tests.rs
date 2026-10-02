//! Real adoption/lifetime code with mocked Vulkan destruction ABI; no driver.
use super::*;
use ash::vk::Handle;
use std::{
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
};
static PIPELINE_DROPS: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());
unsafe extern "system" fn destroy_pipeline(
    _: vk::Device,
    _: vk::Pipeline,
    _: *const vk::AllocationCallbacks<'_>,
) {
    PIPELINE_DROPS.fetch_add(1, Ordering::Relaxed);
}
macro_rules! no_destroy {
    ($name:ident,$kind:ty) => {
        unsafe extern "system" fn $name(_: vk::Device, _: $kind, _: *const vk::AllocationCallbacks<'_>) {}
    };
}
no_destroy!(destroy_render_pass, vk::RenderPass);
no_destroy!(destroy_layout, vk::PipelineLayout);
no_destroy!(destroy_shader, vk::ShaderModule);
no_destroy!(destroy_cache, vk::PipelineCache);
unsafe extern "system" fn destroy_device(_: vk::Device, _: *const vk::AllocationCallbacks<'_>) {}
fn authority() -> Arc<PipelineCreationAuthority> {
    // Only destructor entrypoints below are invoked. These handles never reach
    // a loader, GPU or driver; this is native ownership/adoption validation.
    let raw = unsafe {
        ash::Device::load_with(
            |name| match name.to_bytes() {
                b"vkDestroyPipeline" => destroy_pipeline as *const c_void,
                b"vkDestroyRenderPass" => destroy_render_pass as *const c_void,
                b"vkDestroyPipelineLayout" => destroy_layout as *const c_void,
                b"vkDestroyShaderModule" => destroy_shader as *const c_void,
                b"vkDestroyPipelineCache" => destroy_cache as *const c_void,
                b"vkDestroyDevice" => destroy_device as *const c_void,
                _ => std::ptr::null(),
            },
            vk::Device::from_raw(1),
        )
    };
    Arc::new(PipelineCreationAuthority {
        device: Arc::new(DeviceHandle::for_retirement_test(raw, ())),
        pipeline_cache: vk::PipelineCache::from_raw(1),
        cache_access: Mutex::new(()),
        solid_layout: vk::PipelineLayout::from_raw(2),
        textured_layout: vk::PipelineLayout::from_raw(3),
        solid_vertex_module: vk::ShaderModule::from_raw(4),
        solid_fragment_module: vk::ShaderModule::from_raw(5),
        texture_vertex_module: vk::ShaderModule::from_raw(6),
        texture_fragment_module: vk::ShaderModule::from_raw(7),
        kawase_fragment_module: vk::ShaderModule::from_raw(8),
        kawase_layout: vk::PipelineLayout::from_raw(9),
    })
}
fn context() -> crate::backend::renderer::ErasedContextId {
    crate::backend::renderer::ContextId::<super::super::super::VulkanTexture>::new().erased()
}
fn set(id: u64) -> FormatPipelineSet {
    FormatPipelineSet {
        render_pass: vk::RenderPass::from_raw(id),
        solid_pipeline: vk::Pipeline::from_raw(id + 1),
        solid_opaque_pipeline: vk::Pipeline::from_raw(id + 2),
        textured_pipeline: vk::Pipeline::from_raw(id + 3),
        textured_opaque_pipeline: vk::Pipeline::from_raw(id + 4),
        prefix_mix_pipeline: vk::Pipeline::from_raw(id + 5),
        kawase_pipeline: vk::Pipeline::from_raw(id + 6),
    }
}
fn prepared(
    authority: &Arc<PipelineCreationAuthority>,
    context: &crate::backend::renderer::ErasedContextId,
    format: vk::Format,
) -> PreparedPipelineBank {
    PreparedPipelineBank {
        authority: authority.clone(),
        context: context.clone(),
        generation: 7,
        format,
        set: Some(set(100)),
    }
}
fn state(authority: &Arc<PipelineCreationAuthority>, capacity: usize) -> PipelineState {
    PipelineState {
        authority: authority.clone(),
        per_format: IndexMap::with_capacity(capacity),
    }
}
const FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
#[test]
fn prepared_pipeline_adoption_uses_exact_native_handles_without_heap_or_driver_calls() {
    let _serial = SERIAL.lock().unwrap();
    PIPELINE_DROPS.store(0, Ordering::Relaxed);
    let authority = authority();
    let context = context();
    let mut owner = state(&authority, 2);
    let mut bank = prepared(&authority, &context, FORMAT);
    let (result, heap) = super::super::super::storage_heap_probe::measure(|| {
        owner.adopt_prepared_for_context(&mut bank, &context, 7, FORMAT)
    });
    assert_eq!(result.unwrap(), PreparedResourceAdoption::Adopted);
    assert_eq!(heap, [0; 4]);
    assert_eq!(PIPELINE_DROPS.load(Ordering::Relaxed), 0);
    assert_eq!(
        owner
            .prepared_pipelines_for_format(FORMAT)
            .unwrap()
            .textured_pipeline
            .as_raw(),
        103
    );
    drop(bank);
    assert_eq!(PIPELINE_DROPS.load(Ordering::Relaxed), 0);
    drop(owner);
    assert_eq!(PIPELINE_DROPS.load(Ordering::Relaxed), 6);
}
#[test]
fn prepared_pipeline_wrong_identity_keeps_both_owner_and_bank_unchanged() {
    let _serial = SERIAL.lock().unwrap();
    PIPELINE_DROPS.store(0, Ordering::Relaxed);
    let authority = authority();
    let context = context();
    let other_context =
        crate::backend::renderer::ContextId::<super::super::super::VulkanTexture>::new().erased();
    let mut owner = state(&authority, 2);
    let mut bank = prepared(&authority, &context, FORMAT);
    for (expected, generation, format) in [
        (&other_context, 7, FORMAT),
        (&context, 8, FORMAT),
        (&context, 7, vk::Format::B8G8R8A8_SRGB),
    ] {
        let (result, heap) = super::super::super::storage_heap_probe::measure(|| {
            owner.adopt_prepared_for_context(&mut bank, expected, generation, format)
        });
        assert!(result.is_err());
        assert_eq!(heap, [0; 4]);
        assert!(bank.set.is_some());
        assert!(owner.per_format.is_empty());
    }
    let other = self::authority();
    let mut foreign = state(&other, 2);
    assert!(foreign
        .adopt_prepared_for_context(&mut bank, &context, 7, FORMAT)
        .is_err());
    assert_eq!(PIPELINE_DROPS.load(Ordering::Relaxed), 0);
}
#[test]
fn full_prepared_cache_refuses_without_evicting_or_destroying_native_readers() {
    let _serial = SERIAL.lock().unwrap();
    PIPELINE_DROPS.store(0, Ordering::Relaxed);
    let authority = authority();
    let context = context();
    let mut owner = state(&authority, 1);
    owner.per_format.insert(FORMAT, set(200));
    let other = vk::Format::B8G8R8A8_SRGB;
    let mut bank = prepared(&authority, &context, other);
    let (result, heap) = super::super::super::storage_heap_probe::measure(|| {
        owner.adopt_prepared_for_context(&mut bank, &context, 7, other)
    });
    assert_eq!(result.unwrap(), PreparedResourceAdoption::CapacityDeferred);
    assert_eq!(heap, [0; 4]);
    assert!(bank.set.is_some());
    assert_eq!(PIPELINE_DROPS.load(Ordering::Relaxed), 0);
    assert_eq!(
        owner
            .prepared_pipelines_for_format(FORMAT)
            .unwrap()
            .textured_pipeline
            .as_raw(),
        203
    );
}
#[test]
fn duplicate_prepared_bank_returns_unused_handles_to_its_cold_result_owner() {
    let _serial = SERIAL.lock().unwrap();
    PIPELINE_DROPS.store(0, Ordering::Relaxed);
    let authority = authority();
    let context = context();
    let mut owner = state(&authority, 1);
    owner.per_format.insert(FORMAT, set(200));
    let mut bank = prepared(&authority, &context, FORMAT);
    let (result, heap) = super::super::super::storage_heap_probe::measure(|| {
        owner.adopt_prepared_for_context(&mut bank, &context, 7, FORMAT)
    });
    assert_eq!(result.unwrap(), PreparedResourceAdoption::AlreadyPrepared);
    assert_eq!(heap, [0; 4]);
    assert!(bank.set.is_some());
    assert_eq!(PIPELINE_DROPS.load(Ordering::Relaxed), 0);
    drop(bank);
    assert_eq!(PIPELINE_DROPS.load(Ordering::Relaxed), 6);
    assert_eq!(
        owner
            .prepared_pipelines_for_format(FORMAT)
            .unwrap()
            .textured_pipeline
            .as_raw(),
        203
    );
}

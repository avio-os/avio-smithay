# Smithay VulkanRenderer Full Completion Plan (ash-based)

Status: Draft (Phases 0-7 complete on 2026-02-16)  
Scope: Full completion, production readiness, and long-term maintainability  
Location: `src/backend/renderer/vulkan/`  
API backend: `ash` only

## 1. Goal

Build a first-class `VulkanRenderer` for Smithay with the same engineering quality bar as existing renderers, designed for zero-copy dmabuf import/render paths and direct compatibility with `DrmCompositor`.

This plan is for full completion, not MVP.

## 2. Definition Of Full Completion

Full completion means all items below are true:

- `VulkanRenderer` is production-usable for compositor composition paths and not limited to demos.
- Core renderer traits are fully implemented and tested:
  - `RendererSuper`
  - `Renderer`
  - `Frame`
  - `Bind<Dmabuf>`
  - `ImportDma`
- Extended renderer functionality is implemented where applicable:
  - `ImportDmaWl`
  - `ImportMem`
  - `ImportMemWl`
  - `ExportMem`
  - `Offscreen<VulkanImage>` (or equivalent Vulkan target type)
  - `Blit`
- `DrmCompositor` integration is robust for real workloads:
  - direct scanout decision paths do not regress
  - composition path supports damage, transforms, alpha, opaque regions
  - explicit sync paths interoperate with Smithay `SyncPoint`
- Format and modifier handling is complete for common Linux compositor deployments:
  - robust `FormatSet` reporting for render and import paths
  - modifier-aware import and render support
  - strict validation with clear fallback behavior
- Safety and maintainability match existing Smithay standards:
  - all `unsafe` blocks documented with invariants
  - no resource leaks under error or drop paths
  - clear internal module boundaries
- Validation and CI coverage include unit, integration, and stress scenarios.

## 3. Explicit Non-Goals

- Replacing or removing `GlesRenderer`.
- Implementing EGL-specific import traits directly in Vulkan renderer (`ImportEgl`).
- Supporting every Vulkan extension on day one; only validated extension sets are enabled.

## 4. High-Level Architecture

## 4.1 Module Layout

Create `src/backend/renderer/vulkan/` with focused modules:

- `mod.rs`: public API, renderer struct, trait impl wiring.
- `error.rs`: renderer error model and mapping to `SwapBuffersError`.
- `device.rs`: logical device, queues, pools, extension dispatch.
- `texture.rs`: Vulkan texture/image handles and metadata.
- `target.rs`: framebuffer/render-target binding types.
- `frame.rs`: `Frame` state machine and command recording.
- `pipeline.rs`: graphics pipelines (solid + textured, shader lifecycle).
- `descriptor.rs`: descriptor set layouts, pools, bindings.
- `format.rs`: drm fourcc and modifier negotiation helpers.
- `dmabuf.rs`: import/bind helpers and dmabuf caches.
- `sync.rs`: fence/semaphore, sync_file import/export, `SyncPoint` bridge.
- `upload.rs`: shm/memory upload staging for `ImportMem`.
- `readback.rs`: `ExportMem` implementation.
- `blit.rs`: `Blit` implementation.

## 4.2 Resource Lifetime Rules

- Renderer owns Vulkan device-side long-lived objects:
  - `VkDevice`, queue(s), command pool(s), descriptor pool(s), pipeline cache.
- Per-frame objects are short-lived and strictly bounded by `Frame` lifetime.
- Imported dmabufs are cached by `WeakDmabuf` with liveness checks.
- Drop paths must be deterministic and safe even for partially initialized objects.

## 4.3 Frame State Machine

- `render()` starts command buffer recording and render pass scope.
- Drawing methods record into the in-flight command buffer.
- `finish()` ends pass, submits, and returns a `SyncPoint`.
- Dropping without `finish()` must not submit partially recorded work.
- Re-entrant misuse is prevented by type and runtime guards.

## 5. Feature And Parity Matrix

## 5.1 Core parity (required)

- [ ] `RendererSuper` and `Renderer` contract compliance.
- [ ] `Frame` methods:
  - [ ] `clear`
  - [ ] `draw_solid`
  - [ ] `render_texture_from_to`
  - [ ] `wait`
  - [ ] `finish`
- [ ] `Bind<Dmabuf>` for render targets.
- [ ] `ImportDma` for client dmabufs.
- [ ] Stable `context_id` semantics.
- [ ] Filter controls (`downscale_filter`, `upscale_filter`).
- [ ] `DebugFlags` support.

## 5.2 Extended parity (required for full completion)

- [ ] `ImportDmaWl`.
- [ ] `ImportMem` and `ImportMemWl`.
- [ ] `ExportMem` and `TextureMapping`.
- [ ] `Offscreen` buffers.
- [ ] `Blit`.
- [ ] Multi-format read/write support with clear unsupported errors.

## 5.3 Integration parity

- [ ] `DrmCompositor` stable operation under:
  - [ ] partial damage
  - [ ] full damage
  - [ ] overlay/cursor assignment fallback cases
  - [ ] explicit sync flow
- [ ] `renderer::multigpu` compatibility where trait bounds require it.
- [ ] Wayland surface import path for dmabuf and shm clients.

## 6. Phased Work Plan

## Phase 0: Design freeze and scaffolding

- [x] Add `renderer_vulkan` feature in `Cargo.toml` (depends on `backend_vulkan`).
- [x] Export module from `src/backend/renderer/mod.rs`.
- [x] Add top-level docs in `src/backend/renderer/mod.rs` and `src/backend/mod.rs` to include Vulkan renderer support.
- [x] Establish module skeleton and compile-only stubs.
- [x] Define error taxonomy and `SwapBuffersError` mappings.

Exit criteria:
- `cargo check` passes for minimal Vulkan renderer feature set.
- Verified on 2026-02-16 with `cargo check --no-default-features --features renderer_vulkan`.

## Phase 1: Device and command infrastructure

- [x] Build renderer-owned device state from existing Smithay Vulkan abstractions.
- [x] Validate required extensions and features at startup with explicit errors.
- [x] Select queue family strategy and command pool model.
- [x] Implement command buffer allocation/recycling strategy.
- [x] Add synchronization primitives for submission tracking.

Exit criteria:
- Renderer initializes and tears down cleanly without leaks in repeated create/drop loops.
- Verified on 2026-02-16 with `cargo test --no-default-features --features renderer_vulkan renderer_create_drop_loop -- --nocapture`.

## Phase 2: Format and modifier negotiation

- [x] Implement format conversion helpers using allocator Vulkan format mappings.
- [x] Build `FormatSet` for import and render targets separately.
- [x] Implement `VK_EXT_image_drm_format_modifier` query path and cache.
- [x] Handle implicit modifier (`Invalid`) logic explicitly and safely.
- [x] Add capability query APIs used by bind/import paths.

Exit criteria:
- Deterministic format tables and modifier intersection behavior with unit tests.
- Verified on 2026-02-16 with `cargo test --no-default-features --features renderer_vulkan vulkan::format::tests -- --nocapture`.

## Phase 3: dmabuf import and bind

- [x] Implement dmabuf to `VkImage` import for sampled textures.
- [x] Implement dmabuf to render-target bind path (image view + framebuffer attachments).
- [x] Add cache keys using `WeakDmabuf` plus metadata validation.
- [x] Handle re-import, stale cache, and lifetimes correctly.
- [x] Include strict validation for plane count, strides, offsets, modifiers, and usage flags.

Exit criteria:
- Import/bind passes stress tests for rapid create/destroy and repeated frame usage.
- Verified on 2026-02-16 with `cargo test --no-default-features --features renderer_vulkan vulkan::dmabuf::tests -- --nocapture`.

## Phase 4: Pipeline and shader system

- [x] Add solid-color pipeline.
- [x] Add texture sampling pipeline with transform and alpha controls.
- [x] Add descriptor set layouts/pools and update model.
- [x] Precompile/ship SPIR-V assets or build-time shader compilation path.
- [x] Add pipeline cache persistence strategy (optional but recommended).

Exit criteria:
- `clear`, `draw_solid`, and textured draw produce expected output in offscreen tests.
- Verified on 2026-02-16 with `cargo test --no-default-features --features renderer_vulkan vulkan::pipeline::tests -- --nocapture`.

## Phase 5: Frame implementation

- [x] Implement `Renderer::render`.
- [x] Implement `Frame` methods and shared draw internals.
- [x] Correctly handle output transform and projection mapping.
- [x] Implement damage-scissoring and opaque-region optimizations.
- [x] Implement robust `finish` semantics with exactly-once submission behavior.

Exit criteria:
- Rendering operations behave correctly for all transforms and damage scenarios in tests.
- Verified on 2026-02-16 with `cargo test --no-default-features --features renderer_vulkan vulkan::frame::tests -- --nocapture`.

## Phase 6: SyncPoint bridge and explicit sync

- [x] Add Vulkan fence wrapper implementing Smithay `Fence`.
- [x] Export sync fd from submitted work where supported.
- [x] Import `SyncPoint` fd into Vulkan wait path for `Renderer::wait` and `Frame::wait`.
- [x] Fallback to blocking wait when import/export is unavailable.
- [x] Document sync guarantees and limitations by driver capability.

Exit criteria:
- Explicit sync tests pass and no deadlock or missed-wait regressions appear under stress.
- Verified on 2026-02-16 with:
  - `cargo check --no-default-features --features renderer_vulkan`
  - `cargo test --no-default-features --features renderer_vulkan vulkan::frame::tests -- --nocapture`
  - `cargo test --no-default-features --features renderer_vulkan renderer_create_drop_loop -- --nocapture`

## Phase 7: Memory import path (shm / CPU upload)

- [x] Implement `ImportMem` with staging upload and format conversion gates.
- [x] Implement partial update path (`update_memory`) with bounds checks.
- [x] Implement `ImportMemWl`.
- [x] Add format support table for memory uploads.

Exit criteria:
- Shm clients render correctly and partial updates are validated.
- Verified on 2026-02-16 with:
  - `cargo check --no-default-features --features renderer_vulkan`
  - `cargo check --no-default-features --features renderer_vulkan,wayland_frontend`
  - `cargo test --no-default-features --features renderer_vulkan vulkan::upload::tests -- --nocapture`
  - `cargo test --no-default-features --features renderer_vulkan vulkan::frame::tests -- --nocapture`

## Phase 8: Export and offscreen support

- [ ] Implement `Offscreen` buffer creation for Vulkan-backed render targets.
- [ ] Implement `ExportMem` readback path with format conversion handling.
- [ ] Implement `TextureMapping` lifetime and map/unmap safety.

Exit criteria:
- Offscreen rendering and readback pass deterministic image validation tests.

## Phase 9: Blit and multigpu requirements

- [ ] Implement `Blit` trait with Vulkan transfer/blit operations.
- [ ] Ensure compatibility with `renderer::multigpu` trait constraints.
- [ ] Validate import/export interactions across GPUs where supported.

Exit criteria:
- `renderer::multigpu` compiles and runtime smoke tests pass for Vulkan renderer paths.

## Phase 10: DrmCompositor integration hardening

- [ ] Validate all direct scanout fallback paths with Vulkan renderer bound dmabufs.
- [ ] Validate compositor behavior with implicit and explicit modifiers.
- [ ] Verify no regressions in queue/commit timing and state tracking.
- [ ] Add integration tests for typical compositor scene composition.

Exit criteria:
- Stable multi-frame rendering with `DrmCompositor` in real and synthetic tests.

## Phase 11: Diagnostics, profiling, and debugging

- [ ] Add tracing spans analogous to existing renderer instrumentation.
- [ ] Add optional Vulkan debug markers and validation hooks in debug builds.
- [ ] Add runtime stats for cache hit/miss and command submission timing.

Exit criteria:
- Debug output is actionable and profiling data can isolate performance bottlenecks.

## Phase 12: Documentation and examples

- [ ] Public module docs for renderer lifecycle and trait behavior.
- [ ] Safety docs for all `unsafe` blocks.
- [ ] Developer docs for format/modifier and sync expectations.
- [ ] Add example integration snippet for Vulkan renderer with DRM composition.

Exit criteria:
- API docs and implementation docs are complete enough for external users to adopt safely.

## Phase 13: Stabilization and release gating

- [ ] Run full test matrix (unit + integration + stress).
- [ ] Run sanitizer/valgrind-like leak checks where possible.
- [ ] Run clippy and formatting checks.
- [ ] Benchmark against existing renderer paths for baseline scenarios.
- [ ] Resolve all P0 and P1 defects before merge.

Exit criteria:
- All release gates pass and maintainers approve merge quality.

## 7. Test Strategy

## 7.1 Unit Tests

- format conversion and modifier selection
- command/frame state transitions
- cache behavior and eviction
- sync bridge behavior with mocked/real fence handles

## 7.2 Integration Tests

- offscreen clear/solid/texture correctness
- dmabuf import to render target composition
- `DrmCompositor` render/queue/frame_submitted cycle
- wayland dmabuf and shm client rendering paths

## 7.3 Stress Tests

- rapid dmabuf churn with repeated import/bind
- many short frames with random damage regions
- repeated renderer create/destroy cycles
- sync wait/export/import contention scenarios

## 7.4 Hardware/Driver Matrix

Minimum matrix before production recommendation:

- Mesa RADV (AMD)
- Mesa ANV (Intel)
- NVIDIA proprietary driver path
- Lavapipe software fallback for CI smoke testing

## 8. Safety, Robustness, And Performance Gates

## 8.1 Safety Gates

- every `unsafe` block has invariant comments
- no use-after-free under cache invalidation
- no double-destroy or stale-handle use
- no panics in normal runtime paths

## 8.2 Robustness Gates

- renderer recovers from temporary failures when possible
- clear separation of context-lost vs temporary errors
- deterministic behavior when extensions are missing

## 8.3 Performance Gates

- no unnecessary CPU copies on dmabuf import path
- no redundant pipeline recreation per frame
- bounded descriptor and command pool growth
- measurable frame-time overhead within acceptable range of existing renderer baselines

## 9. Risk Register And Mitigations

- Risk: Driver-specific modifier quirks break import.
  - Mitigation: strict probe logic, per-device cache, explicit fallback and telemetry.
- Risk: Sync fd import/export differs across drivers.
  - Mitigation: layered fallback (`SyncPoint` wait), capability detection at runtime.
- Risk: Descriptor pool exhaustion in pathological scenes.
  - Mitigation: pool sizing heuristics, recycling, and pressure metrics.
- Risk: Hidden lifetime bugs in dmabuf cache.
  - Mitigation: weak-key caches, explicit liveness checks, stress tests.
- Risk: Shader/pipeline combinatoric growth.
  - Mitigation: fixed baseline pipelines first, controlled permutation strategy.

## 10. Milestone Summary

- Milestone A: Core renderer and dmabuf import/bind operational.
- Milestone B: Full frame rendering and sync integration stable.
- Milestone C: Memory import/export, offscreen, and blit complete.
- Milestone D: `DrmCompositor` and multigpu hardening complete.
- Milestone E: Documentation, testing, and release gates complete.

## 11. Completion Checklist (Single View)

- [x] Feature gating and module scaffolding complete.
- [ ] Core traits implemented and validated.
- [ ] Extended traits implemented and validated.
- [x] Format/modifier negotiation complete.
- [x] dmabuf import/bind robust under stress.
- [x] Frame lifecycle and submission semantics complete.
- [x] Explicit sync integration complete.
- [x] Memory upload path complete.
- [ ] Export/offscreen/readback complete.
- [ ] Blit and multigpu compatibility complete.
- [ ] `DrmCompositor` hardening complete.
- [ ] Diagnostics and profiling complete.
- [ ] Documentation complete.
- [ ] Test matrix and release gates complete.

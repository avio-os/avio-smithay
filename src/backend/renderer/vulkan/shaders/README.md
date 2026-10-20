# Vulkan renderer shaders

The `.spv` files next to each source are checked in and pulled into the binary by
`include_bytes!` in `pipeline.rs`. There is no build script, so editing a `.frag`
or `.vert` has no effect until its `.spv` is regenerated and committed with it.

## Regenerating

Compile with **shaderc**, targeting **Vulkan 1.0**, entry point `main`, at
**optimization level zero**. With the same shaderc release those settings
reproduce the committed artifacts byte-for-byte, so a regeneration that only
reformats the source should produce no diff in the `.spv`. Another release
changes only the generator word and debug-only `OpSourceExtension`
instructions:

```rust
let mut options = shaderc::CompileOptions::new().unwrap();
options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_0 as u32);
options.set_optimization_level(shaderc::OptimizationLevel::Zero);
compiler.compile_into_spirv(&source, kind, path, "main", Some(&options))
```

Equivalently, with the standalone compiler:

```sh
glslc --target-env=vulkan1.0 -O0 -fentry-point=main -o texture.frag.spv texture.frag
```

Optimization must stay at zero. Higher levels produce a smaller module that works
but makes every future diff unreviewable against the committed history.

## Colour space

The renderer blends in encoded sRGB (`VulkanRenderer::BLEND_SPACE`). Colour
attachments are viewed through their UNORM storage format, and every sampled
texel is already the encoded premultiplied value SourceOver expects, so neither
`texture.frag` nor `kawase.frag` converts colour: the blur and its CSS filters
work on stored values, as CSS `backdrop-filter` does. See
`docs/rendering-plan.md` in the Avio repo for the full contract.

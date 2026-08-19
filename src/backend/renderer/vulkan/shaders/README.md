# Vulkan renderer shaders

The `.spv` files next to each source are checked in and pulled into the binary by
`include_bytes!` in `pipeline.rs`. There is no build script, so editing a `.frag`
or `.vert` has no effect until its `.spv` is regenerated and committed with it.

## Regenerating

Compile with **shaderc**, targeting **Vulkan 1.0**, entry point `main`, at
**optimization level zero**. Those settings reproduce the committed artifacts
byte-for-byte, so a regeneration that only reformats the source should produce no
diff in the `.spv`:

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

`texture.frag` is the single authority that converts sampled texels into
premultiplied linear light; `kawase.frag` decodes its taps for the same reason.
Colour attachments are viewed as `_SRGB`, so the hardware performs the
destination decode and the encode on store — shaders must not encode as well.
See `docs/rendering-plan.md` in the Avio repo for the full contract.

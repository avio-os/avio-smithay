#version 450

// Dual-Kawase blur pass (KWin formulation). One module serves both pyramid
// directions via `mode`. Taps, filtering and the CSS colour transform all work
// on the stored encoded-sRGB premultiplied values, the renderer's blend space,
// which is what CSS `backdrop-filter` specifies; neither side is converted.

layout(set = 0, binding = 0) uniform sampler2D texture_sampler;

layout(push_constant) uniform KawasePushConstants {
    vec2 halfpixel; // 0.5 / size of the SMALLER pyramid level, in UV units
    float offset;   // kawase spread multiplier
    uint mode;      // 0 = downsample (5 taps), 1 = upsample (8 taps)
    float saturation; // post-blur saturation; 1 = identity
    float contrast;   // post-blur CSS contrast(), applied before brightness; 1 = identity
    float brightness; // post-blur CSS brightness(), applied before saturation; 1 = identity
    uint _pad0;
    vec2 source_uv_scale;
    vec2 source_uv_min;
    vec2 source_uv_max;
    vec2 source_uv_offset;
} constants;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

vec4 tap(vec2 uv) {
    // Preserve the existing full-image path exactly. A region-backed source
    // needs its own clamp boundary so uninitialized backing pixels cannot leak.
    if (any(notEqual(constants.source_uv_scale, vec2(1.0))) || any(notEqual(constants.source_uv_offset, vec2(0.0)))) {
        uv = clamp(constants.source_uv_offset + uv * constants.source_uv_scale, constants.source_uv_min, constants.source_uv_max);
    }
    return texture(texture_sampler, uv);
}

void main() {
    vec2 hp = constants.halfpixel * constants.offset;
    vec4 sum;
    if (constants.mode == 0u) {
        sum = tap(in_uv) * 4.0;
        sum += tap(in_uv - hp);
        sum += tap(in_uv + hp);
        sum += tap(in_uv + vec2(hp.x, -hp.y));
        sum += tap(in_uv - vec2(hp.x, -hp.y));
        sum /= 8.0;
    } else {
        sum = tap(in_uv + vec2(-hp.x * 2.0, 0.0));
        sum += tap(in_uv + vec2(-hp.x, hp.y)) * 2.0;
        sum += tap(in_uv + vec2(0.0, hp.y * 2.0));
        sum += tap(in_uv + vec2(hp.x, hp.y)) * 2.0;
        sum += tap(in_uv + vec2(hp.x * 2.0, 0.0));
        sum += tap(in_uv + vec2(hp.x, -hp.y)) * 2.0;
        sum += tap(in_uv + vec2(0.0, -hp.y * 2.0));
        sum += tap(in_uv + vec2(-hp.x, -hp.y)) * 2.0;
        sum /= 12.0;
    }
    // CSS filter-list order: contrast(), brightness(), saturate(), each
    // clamped before the next reads it. All three are affine in straight
    // colour, so they apply to premultiplied RGB with alpha standing in for
    // one: contrast pivots around 0.5 * alpha and every clamp ceiling is
    // alpha, which preserves the premultiplied invariant before blending.
    vec3 pivot = vec3(0.5 * sum.a);
    sum.rgb = clamp((sum.rgb - pivot) * constants.contrast + pivot, vec3(0.0), vec3(sum.a));
    sum.rgb = clamp(sum.rgb * constants.brightness, vec3(0.0), vec3(sum.a));
    // The rows of the SVG/CSS saturate matrix sum to one, so applying its
    // luma form directly to premultiplied RGB is equivalent to applying it to
    // straight RGB and premultiplying afterwards.
    float luma = dot(sum.rgb, vec3(0.213, 0.715, 0.072));
    sum.rgb = clamp(vec3(luma) + (sum.rgb - vec3(luma)) * constants.saturation,
                    vec3(0.0), vec3(sum.a));
    out_color = sum;
}

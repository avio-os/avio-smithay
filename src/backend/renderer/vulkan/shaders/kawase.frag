#version 450

// Dual-Kawase blur pass (KWin formulation). One module serves both pyramid
// directions via `mode`. A pass explicitly selects linear-light or
// CSS-compatible encoded-sRGB filtering. `encode_output` re-encodes linear
// output only when the destination is not an _SRGB attachment doing that for
// us — encoding in both places would darken every pyramid level.
//
// Blur inputs are the compositor's own offscreens, which store premultiplied-linear
// values in sRGB encoding, so a plain decode is exact; there is no unpremultiply
// step here, unlike texture.frag's electrical-premultiplied client buffers.

layout(set = 0, binding = 0) uniform sampler2D texture_sampler;

layout(push_constant) uniform KawasePushConstants {
    vec2 halfpixel; // 0.5 / size of the SMALLER pyramid level, in UV units
    float offset;   // kawase spread multiplier
    uint mode;      // 0 = downsample (5 taps), 1 = upsample (8 taps)
    uint encode_output; // 1 = shader must sRGB-encode; 0 = the _SRGB attachment does
    float saturation; // post-blur saturation; 1 = identity
    uint encoded_srgb; // 1 = CSS-compatible filtering on encoded channel values
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

vec3 srgb_to_linear(vec3 c) {
    bvec3 lo = lessThanEqual(c, vec3(0.04045));
    vec3 linear_lo = c / 12.92;
    vec3 linear_hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(linear_hi, linear_lo, vec3(lo));
}

vec3 linear_to_srgb(vec3 c) {
    bvec3 lo = lessThanEqual(c, vec3(0.0031308));
    vec3 srgb_lo = c * 12.92;
    vec3 srgb_hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(srgb_hi, srgb_lo, vec3(lo));
}

vec4 tap(vec2 uv) {
    // Preserve the existing full-image path exactly. A region-backed source
    // needs its own clamp boundary so uninitialized backing pixels cannot leak.
    if (any(notEqual(constants.source_uv_scale, vec2(1.0))) || any(notEqual(constants.source_uv_offset, vec2(0.0)))) {
        uv = clamp(constants.source_uv_offset + uv * constants.source_uv_scale, constants.source_uv_min, constants.source_uv_max);
    }
    vec4 s = texture(texture_sampler, uv);
    if (constants.encoded_srgb == 0u) {
        s.rgb = srgb_to_linear(s.rgb);
    }
    return s;
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
    if (constants.encoded_srgb == 0u && constants.encode_output != 0u) {
        sum.rgb = linear_to_srgb(sum.rgb);
    }
    out_color = sum;
}

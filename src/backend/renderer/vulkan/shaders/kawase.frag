#version 450

// Dual-Kawase blur pass (KWin formulation). One module serves both pyramid
// directions via `mode`; `linearize` decodes sRGB-encoded storage to linear
// before averaging and re-encodes on write, so 8-bit storage keeps sRGB
// precision while the blur itself averages physically-linear light.

layout(set = 0, binding = 0) uniform sampler2D texture_sampler;

layout(push_constant) uniform KawasePushConstants {
    vec2 halfpixel; // 0.5 / size of the SMALLER pyramid level, in UV units
    float offset;   // kawase spread multiplier
    uint mode;      // 0 = downsample (5 taps), 1 = upsample (8 taps)
    uint linearize; // 1 = average in linear space
    uint _pad0;
    uint _pad1;
    uint _pad2;
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
    vec4 s = texture(texture_sampler, uv);
    if (constants.linearize != 0u) {
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
    if (constants.linearize != 0u) {
        sum.rgb = linear_to_srgb(sum.rgb);
    }
    out_color = sum;
}

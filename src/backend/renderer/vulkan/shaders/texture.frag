#version 450

layout(set = 0, binding = 0) uniform sampler2D texture_sampler;

layout(push_constant) uniform TexturePushConstants {
    float alpha;
    uint transform;
    uint y_inverted;
    uint _pad0;
    vec2 src_offset;
    vec2 src_scale;
} constants;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

vec2 apply_transform(vec2 uv, uint transform) {
    if (transform == 0u) {
        return uv;
    }
    if (transform == 1u) {
        return vec2(uv.y, 1.0 - uv.x);
    }
    if (transform == 2u) {
        return vec2(1.0 - uv.x, 1.0 - uv.y);
    }
    if (transform == 3u) {
        return vec2(1.0 - uv.y, uv.x);
    }
    if (transform == 4u) {
        return vec2(1.0 - uv.x, uv.y);
    }
    if (transform == 5u) {
        return vec2(uv.y, uv.x);
    }
    if (transform == 6u) {
        return vec2(uv.x, 1.0 - uv.y);
    }
    if (transform == 7u) {
        return vec2(1.0 - uv.y, 1.0 - uv.x);
    }

    return uv;
}

void main() {
    vec2 uv = apply_transform(in_uv, constants.transform);

    if (constants.y_inverted != 0u) {
        uv.y = 1.0 - uv.y;
    }

    uv = constants.src_offset + (uv * constants.src_scale);

    vec4 sampled = texture(texture_sampler, uv);
    out_color = vec4(sampled.rgb, sampled.a * constants.alpha);
}

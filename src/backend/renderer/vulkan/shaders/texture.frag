#version 450

layout(set = 0, binding = 0) uniform sampler2D texture_sampler;

layout(push_constant) uniform TexturePushConstants {
    float alpha;
    uint transform;
    uint y_inverted;
    uint rounded_clip_flags;
    vec2 src_offset;
    vec2 src_scale;
    vec4 clip_rect;
    vec4 clip_params;
    vec4 effect;
    vec4 effect_params;
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

float rounded_clip_alpha(vec2 frag_pos) {
    if (constants.rounded_clip_flags == 0u) {
        return 1.0;
    }

    vec2 clip_pos = frag_pos - constants.clip_rect.xy;
    vec2 clip_size = constants.clip_rect.zw;
    if (clip_pos.x < 0.0 || clip_pos.y < 0.0 ||
        clip_pos.x > clip_size.x || clip_pos.y > clip_size.y) {
        return 0.0;
    }

    float radius = min(constants.clip_params.x, min(clip_size.x, clip_size.y) * 0.5);
    if (radius <= 0.0) {
        return 1.0;
    }

    float exponent = max(constants.clip_params.y, 2.0);
    float aa_width = max(constants.clip_params.z, 0.001);
    vec2 center;
    uint corner_flag = 0u;

    if (clip_pos.x < radius && clip_pos.y < radius) {
        center = vec2(radius, radius);
        corner_flag = 1u;
    } else if (clip_pos.x > clip_size.x - radius && clip_pos.y < radius) {
        center = vec2(clip_size.x - radius, radius);
        corner_flag = 2u;
    } else if (clip_pos.x > clip_size.x - radius && clip_pos.y > clip_size.y - radius) {
        center = vec2(clip_size.x - radius, clip_size.y - radius);
        corner_flag = 4u;
    } else if (clip_pos.x < radius && clip_pos.y > clip_size.y - radius) {
        center = vec2(radius, clip_size.y - radius);
        corner_flag = 8u;
    } else {
        return 1.0;
    }

    if ((constants.rounded_clip_flags & corner_flag) == 0u) {
        return 1.0;
    }

    vec2 delta = abs(clip_pos - center);
    float dist = pow(pow(delta.x, exponent) + pow(delta.y, exponent), 1.0 / exponent) - radius;
    return 1.0 - smoothstep(-aa_width, aa_width, dist);
}

vec2 genie_effect_uv(vec2 uv, out float coverage) {
    float intensity = clamp(constants.effect.y, 0.0, 1.0);
    if (intensity <= 0.001) {
        coverage = 1.0;
        return uv;
    }

    float anchor_x = clamp(constants.effect.z, -1.0, 2.0);
    float anchor_y = clamp(constants.effect.w, 0.0, 1.0);
    bool toward_bottom = anchor_y >= 0.5;
    float vertical = toward_bottom ? uv.y : (1.0 - uv.y);
    float curve = smoothstep(0.0, 1.0, vertical);
    float influence = intensity * curve;
    float neck_width = clamp(constants.effect_params.x, 0.035, 0.35);
    float width = mix(1.0, neck_width, influence);

    float shoulder_wave = sin(vertical * 3.14159265) * 0.06 * intensity * (1.0 - curve);
    float center = mix(0.5, anchor_x, influence) + shoulder_wave;
    float half_width = max(width * 0.5, 0.0001);
    float left = center - half_width;
    float right = center + half_width;
    float aa = max(fwidth(uv.x) * 2.0, 0.0015);
    float left_alpha = smoothstep(left - aa, left + aa, uv.x);
    float right_alpha = 1.0 - smoothstep(right - aa, right + aa, uv.x);
    coverage = left_alpha * right_alpha;

    vec2 warped = uv;
    warped.x = (uv.x - left) / max(width, 0.0001);

    float pull = intensity * curve * (1.0 - curve) * 0.12;
    warped.y = toward_bottom ? uv.y - pull : uv.y + pull;
    return clamp(warped, vec2(0.0), vec2(1.0));
}

vec2 fullscreen_effect_uv(vec2 uv) {
    float progress = clamp(constants.effect.y, 0.0, 1.0);
    float pulse = sin(progress * 3.14159265);
    float depth_strength = clamp(constants.effect_params.y, 0.0, 0.24);
    float flow_strength = clamp(constants.effect_params.w, 0.0, 0.18);
    if (pulse <= 0.001 || (depth_strength <= 0.001 && flow_strength <= 0.001)) {
        return uv;
    }

    vec2 centered = uv - vec2(0.5);
    float radial = dot(centered, centered);
    float zoom = 1.0 + depth_strength * pulse * (1.0 + radial * 0.75);
    vec2 warped = vec2(0.5) + centered / zoom;

    vec2 flow_dir = vec2(constants.effect_params.x, constants.effect_params.z);
    float flow_len = length(flow_dir);
    if (flow_len > 0.001 && flow_strength > 0.001) {
        flow_dir /= flow_len;
        float flow = flow_strength * pulse * (1.0 - smoothstep(0.72, 1.0, progress));
        float along = dot(centered, flow_dir);
        float leading_edge = smoothstep(-0.45, 0.55, along);
        float radial_weight = 0.35 + radial * 1.4;
        warped -= flow_dir * flow * mix(0.45, 1.0, leading_edge) * radial_weight;

        vec2 cross_dir = vec2(-flow_dir.y, flow_dir.x);
        warped -= cross_dir * dot(centered, cross_dir) * flow * 0.1;
    }

    return clamp(warped, vec2(0.0), vec2(1.0));
}

vec2 apply_texture_effect(vec2 uv, out float coverage) {
    coverage = 1.0;
    int kind = int(constants.effect.x + 0.5);
    if (kind == 1) {
        return genie_effect_uv(uv, coverage);
    }
    if (kind == 2 || kind == 3) {
        return fullscreen_effect_uv(uv);
    }
    return uv;
}

void main() {
    float effect_coverage = 1.0;
    vec2 uv = apply_texture_effect(in_uv, effect_coverage);
    uv = apply_transform(uv, constants.transform);

    if (constants.y_inverted != 0u) {
        uv.y = 1.0 - uv.y;
    }

    uv = constants.src_offset + (uv * constants.src_scale);

    vec4 sampled = texture(texture_sampler, uv);
    float coverage = rounded_clip_alpha(gl_FragCoord.xy);
    out_color = vec4(sampled.rgb * constants.alpha, sampled.a * constants.alpha) * coverage * effect_coverage;
}

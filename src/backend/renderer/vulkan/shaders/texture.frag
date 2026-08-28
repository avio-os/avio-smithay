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
    uint source_encoding;
    float clip_scale;
} constants;

// Must match the SOURCE_ENCODING_* constants in pipeline.rs.
const uint SOURCE_ELECTRICAL_PREMULTIPLIED = 0u;
const uint SOURCE_LINEAR_PREMULTIPLIED = 1u;
const uint SOURCE_PASSTHROUGH = 2u;
const uint BOTTOM_EDGE_CLIP_FLAG = 0x80000000u;
const uint CLIP_TRANSFORM_SHIFT = 8u;

vec3 srgb_to_linear(vec3 c) {
    bvec3 lo = lessThanEqual(c, vec3(0.04045));
    vec3 linear_lo = c / 12.92;
    vec3 linear_hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(linear_hi, linear_lo, vec3(lo));
}

// Convert a sampled texel into the premultiplied-LINEAR value the blend expects.
//
// The compositor blends in linear light: the colour attachment is viewed as _SRGB, so
// the hardware decodes the destination and re-encodes the result on store. Only the
// source side is left, and it has two shapes:
//
//   ELECTRICAL_PREMULTIPLIED - client and shell buffers, premultiplied in gamma space
//     (`encode(colour) * alpha`). Decoding that product directly is wrong: it yields
//     roughly `linear(colour) * alpha^2.4`, which visibly darkens translucent glass.
//     Unpremultiply first, decode, then premultiply again in linear.
//   LINEAR_PREMULTIPLIED - our own offscreens, already premultiplied in linear and
//     merely sRGB-encoded for 8-bit precision. A plain decode is exact here, and
//     unpremultiplying would corrupt it.
//
// Both collapse to a plain decode when alpha is 1, which is the opaque fast path.
vec4 to_linear_premultiplied(vec4 texel) {
    if (constants.source_encoding == SOURCE_PASSTHROUGH) {
        return texel;
    }
    if (constants.source_encoding == SOURCE_LINEAR_PREMULTIPLIED) {
        return vec4(srgb_to_linear(texel.rgb), texel.a);
    }
    if (texel.a <= 0.0) {
        return vec4(0.0);
    }
    vec3 straight = min(texel.rgb / texel.a, vec3(1.0));
    return vec4(srgb_to_linear(straight) * texel.a, texel.a);
}

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
    uint corner_flags = constants.rounded_clip_flags & 0xFu;
    if (corner_flags == 0u) {
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

    if ((corner_flags & corner_flag) == 0u) {
        return 1.0;
    }

    vec2 delta = abs(clip_pos - center);
    float dist = pow(pow(delta.x, exponent) + pow(delta.y, exponent), 1.0 / exponent) - radius;
    return 1.0 - smoothstep(-aa_width, aa_width, dist);
}

float cubic_component(float p0, float p1, float p2, float p3, float t) {
    float inverse = 1.0 - t;
    return inverse * inverse * inverse * p0 +
        3.0 * inverse * inverse * t * p1 +
        3.0 * inverse * t * t * p2 +
        t * t * t * p3;
}

float bottom_edge_left_boundary(
    float y,
    float baseline,
    float plateau_top,
    float plateau_left,
    float foot_left,
    float foot_radius,
    float foot_tangent,
    float top_extension,
    float cosine,
    float sine,
    float corner_handle
) {
    float foot_end_y = baseline - foot_tangent * sine;
    float cubic_start_y = plateau_top + top_extension * sine;

    if (y >= foot_end_y) {
        vec2 center = vec2(foot_left - foot_tangent, baseline - foot_radius);
        float dy = y - center.y;
        return center.x + sqrt(max(0.0, foot_radius * foot_radius - dy * dy));
    }

    vec2 foot_end = vec2(
        foot_left + foot_tangent * cosine,
        foot_end_y
    );
    vec2 cubic_start = vec2(
        plateau_left - top_extension * cosine,
        cubic_start_y
    );
    if (y >= cubic_start_y) {
        float line_progress = (y - cubic_start.y) /
            max(foot_end.y - cubic_start.y, 0.0001);
        return mix(cubic_start.x, foot_end.x, line_progress);
    }

    vec2 p0 = cubic_start;
    vec2 p1 = vec2(
        plateau_left - corner_handle * top_extension * cosine,
        plateau_top + corner_handle * top_extension * sine
    );
    vec2 p2 = vec2(
        plateau_left + corner_handle * top_extension,
        plateau_top
    );
    vec2 p3 = vec2(plateau_left + top_extension, plateau_top);
    float low = 0.0;
    float high = 1.0;
    for (int iteration = 0; iteration < 10; iteration++) {
        float candidate = (low + high) * 0.5;
        float candidate_y = cubic_component(p0.y, p1.y, p2.y, p3.y, candidate);
        if (candidate_y > y) {
            low = candidate;
        } else {
            high = candidate;
        }
    }
    float t = (low + high) * 0.5;
    return cubic_component(p0.x, p1.x, p2.x, p3.x, t);
}

float bottom_edge_clip_alpha(vec2 frag_pos) {
    if ((constants.rounded_clip_flags & BOTTOM_EDGE_CLIP_FLAG) == 0u) {
        return 1.0;
    }

    vec2 clip_size = constants.clip_rect.zw;
    vec2 transformed_position = frag_pos - constants.clip_rect.xy;
    if (transformed_position.x < 0.0 || transformed_position.y < 0.0 ||
        transformed_position.x > clip_size.x || transformed_position.y > clip_size.y) {
        return 0.0;
    }

    uint clip_transform =
        (constants.rounded_clip_flags >> CLIP_TRANSFORM_SHIFT) & 0x7u;
    vec2 transformed_uv = transformed_position / max(clip_size, vec2(0.0001));
    vec2 source_uv = apply_transform(transformed_uv, clip_transform);
    bool swaps_axes = clip_transform == 1u || clip_transform == 3u ||
        clip_transform == 5u || clip_transform == 7u;
    vec2 source_size = swaps_axes ? clip_size.yx : clip_size;
    vec2 point = source_uv * source_size;

    const float ANGLE_LOW = 9.0;
    const float ANGLE_FULL = 89.6;
    const float ANGLE_EASE = 0.55;
    const float SPLAY = 52.0;
    const float SPLAY_EASE = 1.2;
    const float MINIMUM_STRAIGHT = 6.0;
    const float JOIN_SMOOTH = 16.0;
    const float SQUIRCLE_EXTENSION = 1.6;
    const float SQUIRCLE_HANDLE = 0.26;
    const float CIRCULAR_HANDLE = 0.4477;
    const float TOP_RADIUS_MAXIMUM = 20.0;
    const float FOOT_RADIUS_MAXIMUM = 8.0;
    const float FOOT_RADIUS_SOFT = 20.0;
    const float TOP_RADIUS_SOFT = 16.0;

    float geometry_scale = max(constants.clip_scale, 0.001);
    float progress = max(constants.clip_params.y, 0.0);
    float raised_height = constants.clip_params.z * progress;
    if (raised_height < 0.5 * geometry_scale) {
        return 0.0;
    }

    float settled = clamp(progress, 0.0, 1.0);
    float angle_degrees = mix(ANGLE_LOW, ANGLE_FULL, pow(settled, ANGLE_EASE));
    float angle_radians = radians(angle_degrees);
    float side_run = angle_radians > radians(89.5) ? 0.0 :
        raised_height / tan(angle_radians);
    float top_radius = min(
        raised_height * 1.3,
        (TOP_RADIUS_MAXIMUM + TOP_RADIUS_SOFT * (1.0 - settled)) * geometry_scale
    );
    float foot_radius = min(
        raised_height * 1.1,
        (FOOT_RADIUS_MAXIMUM + FOOT_RADIUS_SOFT * (1.0 - settled)) * geometry_scale
    );
    float half_tangent = tan(angle_radians * 0.5);
    float span = raised_height / sin(angle_radians);
    float overlap = (foot_radius + top_radius) * half_tangent -
        (span - MINIMUM_STRAIGHT * geometry_scale * settled);
    if (overlap > 0.0) {
        float denominator = max((foot_radius + top_radius) * half_tangent, 0.0001);
        float shrink = max(0.0, 1.0 - overlap / denominator);
        top_radius *= shrink;
        foot_radius *= shrink;
    }

    float straight = span - (foot_radius + top_radius) * half_tangent;
    float curvature_blend = clamp(
        1.0 - straight / (JOIN_SMOOTH * geometry_scale),
        0.0,
        1.0
    );
    if (curvature_blend > 0.0) {
        float mean_radius = (top_radius + foot_radius) * 0.5;
        top_radius = mix(top_radius, mean_radius, curvature_blend);
        foot_radius = mix(foot_radius, mean_radius, curvature_blend);
    }

    float foot_tangent = foot_radius * half_tangent;
    float top_tangent = top_radius * half_tangent;
    float extension = 1.0 + (SQUIRCLE_EXTENSION - 1.0) * (1.0 - settled);
    float corner_handle = mix(CIRCULAR_HANDLE, SQUIRCLE_HANDLE, 1.0 - settled);
    float plateau_half = constants.clip_params.x * 0.5 + constants.clip_params.w;
    float top_extension = min(
        min(top_tangent * extension, top_tangent + max(0.0, straight) * 0.85),
        plateau_half * 0.9
    );

    float baseline = source_size.y;
    float plateau_top = baseline - raised_height;
    float center_x = source_size.x * 0.5;
    float half_width = plateau_half + SPLAY * geometry_scale *
        pow(1.0 - settled, SPLAY_EASE);
    float plateau_left = center_x - half_width;
    float foot_left = plateau_left - side_run;
    float cosine = cos(angle_radians);
    float sine = sin(angle_radians);
    if (point.y < plateau_top || point.y > baseline) {
        return 0.0;
    }

    float left = bottom_edge_left_boundary(
        point.y,
        baseline,
        plateau_top,
        plateau_left,
        foot_left,
        foot_radius,
        foot_tangent,
        top_extension,
        cosine,
        sine,
        corner_handle
    );
    float right = source_size.x - left;
    float signed_inside = min(
        min(point.x - left, right - point.x),
        min(point.y - plateau_top, baseline - point.y)
    );
    float aa_width = max(fwidth(signed_inside) * 0.75, 0.5 * geometry_scale);
    return smoothstep(-aa_width, aa_width, signed_inside);
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

    vec4 sampled = to_linear_premultiplied(texture(texture_sampler, uv));
    float coverage = rounded_clip_alpha(gl_FragCoord.xy) *
        bottom_edge_clip_alpha(gl_FragCoord.xy);
    out_color = vec4(sampled.rgb * constants.alpha, sampled.a * constants.alpha) * coverage * effect_coverage;
}

// SRGB image storage auto-decodes before filtering. The common shader samples
// encoded storage, so reconstruct encoded taps BEFORE the bilinear interpolation.
// This preserves the Vulkan UNORM sampling convention, including CSS filters.
layout(std140) uniform FrameUniforms {
    vec4 destination;
    vec2 target_size;
    uint storage_is_srgb;
    uint sample_nearest;
    uint encode_terminal;
    uint target_is_srgb;
    vec2 source_size;
} frame;

vec3 linear_to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, vec3(lessThanEqual(c, vec3(0.0031308))));
}
vec4 encoded_texel(ivec2 coordinate) {
    ivec2 extent = textureSize(texture_sampler, 0);
    vec4 value = texelFetch(texture_sampler, clamp(coordinate, ivec2(0), extent - 1), 0);
    value.rgb = linear_to_srgb(value.rgb);
    return value;
}
vec4 sample_encoded(vec2 uv) {
    if (frame.storage_is_srgb == 0u) { return texture(texture_sampler, uv); }
    vec2 coordinate = uv * vec2(textureSize(texture_sampler, 0)) - 0.5;
    if (frame.sample_nearest != 0u) { return encoded_texel(ivec2(floor(coordinate + 0.5))); }
    ivec2 base = ivec2(floor(coordinate));
    vec2 weight = fract(coordinate);
    return mix(mix(encoded_texel(base), encoded_texel(base + ivec2(1,0)), weight.x),
               mix(encoded_texel(base + ivec2(0,1)), encoded_texel(base + ivec2(1,1)), weight.x), weight.y);
}
uint count_mask(uint mask) {
    return (mask & 1u) + ((mask >> 1u) & 1u) + ((mask >> 2u) & 1u) + ((mask >> 3u) & 1u);
}

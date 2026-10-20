#version 300 es
precision highp float;
precision highp int;
layout(std140) uniform FrameUniforms {
    vec4 destination;
    vec2 target_size;
    uint storage_is_srgb;
    uint sample_nearest;
    uint encode_terminal;
    uint target_is_srgb;
    vec2 source_size;
} frame;

out vec2 in_uv;
void main() {
    vec2 unit = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    in_uv = unit;
    vec2 position = frame.destination.xy + unit * frame.destination.zw;
    gl_Position = vec4(position / frame.target_size * 2.0 - 1.0, 0.0, 1.0);
}

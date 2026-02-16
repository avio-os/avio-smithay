#version 450

layout(push_constant) uniform SolidPushConstants {
    vec4 color;
} constants;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = constants.color;
}

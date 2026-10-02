#version 450

layout(push_constant) uniform SolidPushConstants {
    vec4 color;
    vec4 owner_rect;
    float owner_radius;
} constants;

layout(location = 0) out vec4 out_color;

void main() {
    if (constants.owner_radius >= 0.0) {
        vec2 p = gl_FragCoord.xy - constants.owner_rect.xy;
        vec2 size = constants.owner_rect.zw;
        float r = min(constants.owner_radius, min(size.x, size.y) * 0.5);
        vec2 d = max(max(vec2(r) - p, p - (size - vec2(r))), vec2(0.0));
        if (any(lessThan(p, vec2(0.0))) || any(greaterThanEqual(p, size)) || dot(d,d) > r*r) { discard; }
    }
    out_color = vec4(constants.color.rgb * constants.color.a, constants.color.a);
}

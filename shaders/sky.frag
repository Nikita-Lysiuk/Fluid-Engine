#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 v_dir;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_hdri;

const vec2 inv_atan = vec2(0.1591, 0.3183);

vec2 sample_spherical_map(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(-v.y));
    uv *= inv_atan;
    uv += 0.5;
    return uv;
}

void main() {
    vec3 dir = normalize(v_dir);
    vec2 uv = sample_spherical_map(dir);
    vec3 color = texture(u_hdri, uv).rgb;

    float exposure = 0.3;
    color *= exposure;

    f_color = vec4(color, 1.0);
}

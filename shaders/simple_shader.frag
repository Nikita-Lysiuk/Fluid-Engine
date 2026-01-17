#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 in_color;
layout(location = 0) out vec4 f_color;

layout(buffer_reference, scalar) readonly buffer CameraDataRef {
    mat4 view;
    mat4 proj;
    mat4 inv_view_proj;
    vec3 camera_pos;
};

layout(push_constant) uniform PushConstants {
    uint64_t camera_addr;
    uint64_t particles_addr;
} push;

void main() {
    CameraDataRef camera = CameraDataRef(push.camera_addr);
    vec2 circ_coord = gl_PointCoord * 2.0 - 1.0;
    float dist_sq = dot(circ_coord, circ_coord);

    if (dist_sq > 1.0) discard;

    float z = sqrt(1.0 - dist_sq);
    vec3 normal = vec3(circ_coord, z);

    vec3 world_light_pos = normalize(vec3(0.5, 1.0, 0.5));

    vec3 light_dir = normalize(mat3(camera.view) * world_light_pos);

    float ambient_strength = 0.15;
    vec3 ambient = ambient_strength * vec3(1.0);

    float diff_strength = max((dot(normal, light_dir) + 1) / 2, 0.0);
    vec3 diffuse = diff_strength * vec3(1.0);

    vec3 view_dir = normalize(camera.camera_pos - vec3(gl_FragCoord));
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec3 specular = vec3(0.5) * spec * vec3(1.0);

    f_color = vec4((ambient + diffuse) * in_color + specular, 1.0);
}
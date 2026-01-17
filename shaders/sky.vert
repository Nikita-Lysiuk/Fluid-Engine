#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, scalar) readonly buffer CameraDataRef {
    mat4 view;
    mat4 proj;
};

layout(buffer_reference, scalar) readonly buffer SkyVerticesRef {
    vec3 positions[];
};

layout(push_constant) uniform PushConstants {
    uint64_t camera_addr;
    uint64_t sky_vertices_addr;
} push;

layout(location = 0) out vec3 v_dir;

void main() {
    CameraDataRef camera = CameraDataRef(push.camera_addr);
    SkyVerticesRef sky = SkyVerticesRef(push.sky_vertices_addr);

    vec3 pos = sky.positions[gl_VertexIndex];
    v_dir = pos;

    mat4 static_view = mat4(mat3(camera.view));

    vec4 clip_pos = camera.proj * static_view * vec4(pos, 1.0);
    gl_Position = clip_pos.xyww;
}
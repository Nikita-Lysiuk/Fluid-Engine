#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 position;
layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outCameraPos;

layout(buffer_reference, scalar) readonly buffer CameraDataRef {
    mat4 view;
    mat4 proj;
    mat4 inv_view_proj;
    vec3 camera_pos;
};

layout(push_constant) uniform PC {
    uint64_t camera_addr;
    uint64_t _pad;
    mat4 model;
} push;

void main() {
    CameraDataRef camera = CameraDataRef(push.camera_addr);

    vec4 worldPos = push.model * vec4(position, 1.0);
    outWorldPos = worldPos.xyz;

    outCameraPos = camera.camera_pos;

    gl_Position = camera.proj * camera.view * worldPos;
}
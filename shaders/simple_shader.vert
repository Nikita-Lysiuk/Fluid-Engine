#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, scalar) readonly buffer CameraDataRef {
    mat4 view;
    mat4 proj;
    mat4 inv_view_proj;
    vec3 camera_pos;
};

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in float inRadius;

layout(location = 0) out vec3 fragColor;
layout(push_constant) uniform PushConstants {
    uint64_t camera_addr;
} push;

void main() {
    CameraDataRef camera = CameraDataRef(push.camera_addr);
    
    gl_Position = camera.proj * camera.view * vec4(inPosition.xyz, 1.0);

    float dist = length(camera.camera_pos - inPosition.xyz);
    gl_PointSize = inRadius * 1000.0 / dist;

    fragColor = inColor.xyz;
}
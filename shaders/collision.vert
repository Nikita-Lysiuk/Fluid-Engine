#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, scalar) readonly buffer CameraDataRef {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

layout(buffer_reference, scalar) readonly buffer BoxVerticesRef {
    vec3 vertices[8];
};

layout(push_constant) uniform PushConstants {
    uint64_t camera_addr;
    uint64_t box_vertices_addr;
} push;

void main() {
    CameraDataRef camera = CameraDataRef(push.camera_addr);
    BoxVerticesRef box = BoxVerticesRef(push.box_vertices_addr);
    
    vec3 pos = box.vertices[gl_VertexIndex];
    gl_Position = camera.projectionMatrix * camera.viewMatrix * vec4(pos, 1.0);
}
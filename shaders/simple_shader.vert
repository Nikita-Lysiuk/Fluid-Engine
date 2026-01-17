


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

struct Particle {
    vec3 position;
    float radius;
    vec3 color;
};

layout(buffer_reference, scalar) readonly buffer ParticlesRef {
    Particle data[];
};

layout(push_constant) uniform PushConstants {
    uint64_t camera_addr;
    uint64_t particles_addr;
} push;

layout(location = 0) out vec3 out_color;

void main() {
    CameraDataRef camera = CameraDataRef(push.camera_addr);
    ParticlesRef particles = ParticlesRef(push.particles_addr);
    
    Particle p = particles.data[gl_VertexIndex];
    
    gl_Position = camera.proj * camera.view * vec4(p.position, 1.0);
    
    float dist = length(camera.camera_pos - p.position);
    gl_PointSize = (p.radius * 1000.0) / dist;
    
    out_color = p.color;
}
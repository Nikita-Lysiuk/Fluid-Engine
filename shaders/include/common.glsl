#ifndef COMMON_GLSL
#define COMMON_GLSL

#define PI 3.1415926535897932384626433832795

struct Entry {
    uint hash;
    uint index;
};

layout(std140, set = 0, binding = 2) uniform SimulationParams {
    float particle_radius;
    float particle_mass;
    float smoothing_radius;
    float target_density;
    float viscosity;
    float relax_factor;
    float dt;
    uint density_iterations;
    uint divergence_iterations;
    float _pad0;
    float _pad1;
    float _pad2;
    vec4 gravity;
    vec4 box_min;
    vec4 box_max;
} sim_params;

uint get_cell_hash(ivec3 grid_pos, uint table_size) {
    uint p1 = 73856093;
    uint p2 = 19349663;
    uint p3 = 83492791;
    return (uint(grid_pos.x) * p1 ^ uint(grid_pos.y) * p2 ^ uint(grid_pos.z) * p3) % table_size;
}

// --- WENDLAND QUINTIC C2 KERNEL ---
float kernel_w(float r, float h) {
    float q = r / h;
    if (q >= 1.0) return 0.0;

    float h3 = h * h * h;
    float k = 21.0 / (2.0 * PI * h3);

    float one_minus_q = 1.0 - q;
    float one_minus_q_4 = one_minus_q * one_minus_q * one_minus_q * one_minus_q;

    return k * one_minus_q_4 * (4.0 * q + 1.0);
}

vec3 kernel_grad(vec3 r_vec, float r, float h) {
    float q = r / h;
    if (q >= 1.0 || r < 1e-6) return vec3(0.0);

    float h3 = h * h * h;
    float l = -210.0 / (PI * h3);

    float one_minus_q = 1.0 - q;
    float one_minus_q_3 = one_minus_q * one_minus_q * one_minus_q;

    vec3 grad_q = r_vec / (r * h);

    return l * q * one_minus_q_3 * grad_q;
}

#endif
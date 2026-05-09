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
    ivec4 grid_res;
} sim_params;

uint get_cell_hash(ivec3 grid_pos, uint table_size) {
    uint p1 = 73856093;
    uint p2 = 19349663;
    uint p3 = 83492791;
    return (uint(grid_pos.x) * p1 ^ uint(grid_pos.y) * p2 ^ uint(grid_pos.z) * p3) % table_size;
}

// --- CUBIC SPLINE ---
float kernel_w(float r, float h) {
    float q = r / h;
    if (q >= 1.0) return 0.0;

    float h3 = h * h * h;
    float k = 8.0 / (PI * h3);

    if (q <= 0.5) {
        float q2 = q * q;
        float q3 = q2 * q;
        return k * (6.0 * q3 - 6.0 * q2 + 1.0);
    } else {
        float one_minus_q = 1.0 - q;
        float one_minus_q_3 = one_minus_q * one_minus_q * one_minus_q;
        return k * 2.0 * one_minus_q_3;
    }
}

vec3 kernel_grad(vec3 r_vec, float r, float h) {
    float q = r / h;
    if (q >= 1.0 || r < 1e-6) return vec3(0.0);

    float h3 = h * h * h;
    float k = 8.0 / (PI * h3);

    float grad_factor;

    if (q <= 0.5) {
        grad_factor = k * (18.0 * q * q - 12.0 * q);
    } else {
        float one_minus_q = 1.0 - q;
        grad_factor = k * -6.0 * one_minus_q * one_minus_q;
    }

    return (grad_factor / (h * r)) * r_vec;
}

#endif
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
    float piston_velocity;   // velocity along +x of the box_min.x wall
    float _pad1;
    float _pad2;
    vec4 gravity;
    vec4 box_min;
    vec4 box_max;
    ivec4 grid_res;
    vec4 attractor;          // xyz = centre, w = strength (0 = off)
    float attractor_radius;
    float _pad3;
    float _pad4;
    float _pad5;
    vec4 pillar;             // x, z of the axis, radius, w = 1 when present
} sim_params;

// Warm start of the pressure solvers (DFSPH paper, Sec. 4.4; SPlisHSPlasH
// `USE_WARMSTART`): a solve starts from half of the previous step's pressure,
// for compressed particles only. The previous pressure is stored as p*dt^2
// (density solver) or p*dt (divergence solver) so it stays valid when dt
// changes. The caps are SPlisHSPlasH's limits on p/rho^2 * dt^k.
#define WARM_START_CAP_DENSITY 0.00025
#define WARM_START_CAP_DIVERGENCE 0.5

// Neighbour lists are built once per substep by density_and_alpha.comp and
// read by every later pass. Entry k of particle i lives at
// neighbor_list[k * num_particles + i], so that a warp reads consecutive
// addresses. A particle never stores itself.
#define MAX_NEIGHBORS 64u

// Spreads the low 10 bits of x so that two zero bits follow each bit.
uint part1by2(uint x) {
    x &= 0x3ffu;
    x = (x | (x << 16)) & 0x030000ffu;
    x = (x | (x << 8)) & 0x0300f00fu;
    x = (x | (x << 4)) & 0x030c30c3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

// Bucket of a grid cell: its Morton (Z-order) code modulo the table size,
// which is a power of two. Particles are sorted by this key, so the low
// Morton bits keep nearby cells close in memory; that matters once the
// particle data no longer fits in the GPU's L2 cache. Cells that share a
// bucket are far apart and are rejected by the distance test.
uint get_cell_hash(ivec3 grid_pos, uint table_size) {
    uvec3 c = uvec3(grid_pos + ivec3(512));
    uint morton = part1by2(c.x) | (part1by2(c.y) << 1) | (part1by2(c.z) << 2);
    return morton & (table_size - 1u);
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

// --- Box walls as a continuous boundary ------------------------------------
//
// Everything outside [box_min, box_max] is solid. Instead of sampling the
// walls with boundary particles (Akinci et al. 2012), each wall contributes
// the kernel integrated over the solid half-space behind it, which for a
// plane depends only on the distance d to it (the planar case of the volume
// maps of Bender et al. 2019):
//
//   V(d) = integral of W(x - y) over the wall,   grad V = -A(d) n,
//   A(d) = 2 pi * integral_d^h W(r) r dr,
//
// where n is the wall normal pointing into the fluid. A wall then acts like
// a neighbour with mass rho0 * V and kernel gradient grad V. Closed forms for
// the cubic spline (q = d / h), derived symbolically; V(0) = 1/2.

float wall_volume(float d, float h) {
    float q = clamp(d / h, 0.0, 1.0);
    float q2 = q * q;
    float q3 = q2 * q;
    if (q <= 0.5) {
        return 16.0/5.0 * q3 * q3 - 24.0/5.0 * q3 * q2 + 8.0/3.0 * q3 - 7.0/5.0 * q + 0.5;
    }
    return -16.0/15.0 * q3 * q3 + 24.0/5.0 * q3 * q2 - 8.0 * q2 * q2 + 16.0/3.0 * q3 - 8.0/5.0 * q + 8.0/15.0;
}

// A(d): magnitude of grad V, units 1/m.
float wall_volume_slope(float d, float h) {
    float q = clamp(d / h, 0.0, 1.0);
    float q2 = q * q;
    if (q <= 0.5) {
        return (-96.0/5.0 * q2 * q2 * q + 24.0 * q2 * q2 - 8.0 * q2 + 7.0/5.0) / h;
    }
    return (32.0/5.0 * q2 * q2 * q - 24.0 * q2 * q2 + 32.0 * q2 * q - 16.0 * q2 + 8.0/5.0) / h;
}

// Sum of the six walls at position x: V and grad V. Corners count the overlap
// of two or three walls more than once, which slightly overestimates the
// boundary there.
// Distance from x to the surface of the vertical pillar and the surface
// normal pointing away from it. Returns false when there is no pillar.
bool pillar_distance(vec3 x, out float distance_to_surface, out vec3 normal) {
    distance_to_surface = 0.0;
    normal = vec3(1.0, 0.0, 0.0);
    if (sim_params.pillar.w == 0.0) {
        return false;
    }
    vec2 from_axis = x.xz - sim_params.pillar.xy;
    float r = length(from_axis);
    if (r > 1e-6) {
        normal = vec3(from_axis.x / r, 0.0, from_axis.y / r);
    }
    distance_to_surface = r - sim_params.pillar.z;
    return true;
}

void box_boundary(vec3 x, out float volume, out vec3 grad) {
    float h = sim_params.smoothing_radius;
    vec3 to_min = x - sim_params.box_min.xyz;
    vec3 to_max = sim_params.box_max.xyz - x;
    volume = 0.0;
    grad = vec3(0.0);
    for (int axis = 0; axis < 3; axis++) {
        vec3 n = vec3(0.0);
        n[axis] = 1.0;
        if (to_min[axis] < h) {
            volume += wall_volume(to_min[axis], h);
            grad -= wall_volume_slope(to_min[axis], h) * n;
        }
        if (to_max[axis] < h) {
            volume += wall_volume(to_max[axis], h);
            grad += wall_volume_slope(to_max[axis], h) * n;
        }
    }

    // The pillar's surface is treated as locally planar, which is accurate
    // while its radius is a few times larger than h.
    float to_pillar;
    vec3 pillar_normal;
    if (pillar_distance(x, to_pillar, pillar_normal) && to_pillar < h) {
        volume += wall_volume(to_pillar, h);
        grad -= wall_volume_slope(to_pillar, h) * pillar_normal;
    }
}

// sum over walls of (v - v_wall) . grad V_wall: the walls' share of the
// velocity divergence. Only the box_min.x wall moves (along +x).
float box_boundary_divergence(vec3 x, vec3 v) {
    float volume;
    vec3 grad;
    box_boundary(x, volume, grad);
    float result = dot(v, grad);

    float h = sim_params.smoothing_radius;
    float to_piston = x.x - sim_params.box_min.x;
    if (to_piston < h) {
        // grad V of the piston wall is -A n with n = +x.
        result += sim_params.piston_velocity * wall_volume_slope(to_piston, h);
    }
    return result;
}

// Neighbour count of particle i including the walls, which count as the
// number of fluid particles that would fill the part of the kernel support
// behind them (spherical cap of a sphere of radius h cut at distance d).
float effective_neighbor_count(vec3 x, uint fluid_neighbors) {
    float h = sim_params.smoothing_radius;
    float spacing = 2.0 * sim_params.particle_radius;
    vec3 to_min = x - sim_params.box_min.xyz;
    vec3 to_max = sim_params.box_max.xyz - x;
    float wall_volume = 0.0;
    for (int axis = 0; axis < 3; axis++) {
        float d_min = clamp(to_min[axis], 0.0, h);
        float d_max = clamp(to_max[axis], 0.0, h);
        wall_volume += PI * (h - d_min) * (h - d_min) * (2.0 * h + d_min) / 3.0;
        wall_volume += PI * (h - d_max) * (h - d_max) * (2.0 * h + d_max) / 3.0;
    }
    float to_pillar;
    vec3 pillar_normal;
    if (pillar_distance(x, to_pillar, pillar_normal)) {
        float d = clamp(to_pillar, 0.0, h);
        wall_volume += PI * (h - d) * (h - d) * (2.0 * h + d) / 3.0;
    }
    return float(fluid_neighbors) + wall_volume / (spacing * spacing * spacing);
}

// SPlisHSPlasH skips the divergence solve for particles with fewer than 20
// neighbours (3D): their velocity divergence is unreliable.
#define MIN_DIVERGENCE_NEIGHBORS 20.0

#endif
#ifndef SOLVER_GLSL
#define SOLVER_GLSL

// Shared by the pressure solver passes and `solver_check.comp`.
//
// `error_sum` accumulates the compression residual of one Jacobi iteration in
// fixed point, 1 unit = 1 / SOLVER_ERROR_SCALE kg/m^3 (float atomics are an
// optional Vulkan feature). `solver_check.comp` turns it into an average.
#define SOLVER_ERROR_SCALE 100.0

// Upper bound for one workgroup's contribution, so that the sum of all
// workgroups cannot overflow 32 bits even for 1M particles.
#define SOLVER_MAX_GROUP_ERROR 1000000u

struct SolverState {
    uint error_sum;
    uint iterations;
    uint converged;
    float avg_error;
};

#endif

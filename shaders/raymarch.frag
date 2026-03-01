#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable
#include "include/common.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inCameraPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform usampler3D densityTex;
layout(set = 0, binding = 1) uniform sampler2D skyboxTex;

layout(push_constant) uniform PC {
    uint64_t camera_addr;
    uint64_t _pad;
    mat4 model;
} push;

const float DENSITY_THRESHOLD  = 0.01;
const float DENSITY_OFFSET     = 300.0;
const float DENSITY_MULTIPLIER = 1.0;
const int   MAX_STEPS          = 96;        // менше кроків = більше FPS
const float STEP_SIZE          = 0.007;

// Фізично коректний IOR морської води
const float IOR_WATER = 1.333;
const float IOR_AIR   = 1.0;

const vec3  ABSORPTION_COEFF = vec3(0.45, 0.085, 0.025);

// Scattering — колір розсіювання всередині товщі
const vec3  SCATTER_COLOR    = vec3(0.04, 0.18, 0.28);
const float SCATTER_STRENGTH = 0.35;

// Sонце
const vec3  SUN_DIR          = normalize(vec3(0.4, 1.0, 0.3));
const vec3  SUN_COLOR        = vec3(1.0, 0.95, 0.85);

// Foam — з'являється там де нормаль дуже "рівна" (горизонтальна поверхня з хвилями)
const vec3  FOAM_COLOR       = vec3(0.92, 0.96, 1.0);

// ============================================================
// UTILITY
// ============================================================

vec2 intersectAABB(vec3 ro, vec3 rd, vec3 bMin, vec3 bMax) {
    vec3 t1 = (bMin - ro) / rd;
    vec3 t2 = (bMax - ro) / rd;
    return vec2(max(max(min(t1.x,t2.x), min(t1.y,t2.y)), min(t1.z,t2.z)),
    min(min(max(t1.x,t2.x), max(t1.y,t2.y)), max(t1.z,t2.z)));
}

float getDensity(vec3 worldPos, vec3 boxMin, vec3 boxMax) {
    vec3 uvw = (worldPos - boxMin) / (boxMax - boxMin);
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return 0.0;

    vec3 gridPos = uvw * vec3(sim_params.grid_res.xyz) - 0.5;
    ivec3 iobase = ivec3(floor(gridPos));
    vec3  f      = fract(gridPos);

    float v000 = float(texelFetch(densityTex, clamp(iobase + ivec3(0,0,0), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);
    float v100 = float(texelFetch(densityTex, clamp(iobase + ivec3(1,0,0), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);
    float v010 = float(texelFetch(densityTex, clamp(iobase + ivec3(0,1,0), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);
    float v110 = float(texelFetch(densityTex, clamp(iobase + ivec3(1,1,0), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);
    float v001 = float(texelFetch(densityTex, clamp(iobase + ivec3(0,0,1), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);
    float v101 = float(texelFetch(densityTex, clamp(iobase + ivec3(1,0,1), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);
    float v011 = float(texelFetch(densityTex, clamp(iobase + ivec3(0,1,1), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);
    float v111 = float(texelFetch(densityTex, clamp(iobase + ivec3(1,1,1), ivec3(0), ivec3(sim_params.grid_res.xyz)-1), 0).r);

    float v0 = mix(mix(v000, v100, f.x), mix(v010, v110, f.x), f.y);
    float v1 = mix(mix(v001, v101, f.x), mix(v011, v111, f.x), f.y);
    return max(0.0, (mix(v0, v1, f.z) - DENSITY_OFFSET) * DENSITY_MULTIPLIER);
}

// Нормаль через центральні різниці — eps менший для чіткішої поверхні
vec3 calcNormal(vec3 p, vec3 bMin, vec3 bMax) {
    // Адаптивний epsilon: менший = більш деталізований, більший = більш гладкий
    float eps = 0.018;
    vec2  h   = vec2(eps, 0.0);
    return normalize(vec3(
                     getDensity(p + h.xyy, bMin, bMax) - getDensity(p - h.xyy, bMin, bMax),
                     getDensity(p + h.yxy, bMin, bMax) - getDensity(p - h.yxy, bMin, bMax),
                     getDensity(p + h.yyx, bMin, bMax) - getDensity(p - h.yyx, bMin, bMax)
                     ));
}

// Schlick Fresnel
float fresnel(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 sampleSkybox(vec3 dir) {
    dir = normalize(dir);
    vec2 uv = vec2(atan(dir.z, dir.x) * 0.15915 + 0.5,
    asin(clamp(dir.y, -1.0, 1.0)) * 0.31831 + 0.5);
    return texture(skyboxTex, uv).rgb;
}

// Thickness raymarching — без break, рахує весь шлях крізь рідину
// Використовує великі кроки (дешево)
float calcThickness(vec3 startP, vec3 dir, float tMax, vec3 bMin, vec3 bMax) {
    float thickness = 0.0;
    float dStep = 0.06;
    float t = 0.0;
    // Обмежуємо кількість кроків щоб не вбивати FPS
    for (int i = 0; i < 32 && t < tMax; i++, t += dStep) {
if (getDensity(startP + dir * t, bMin, bMax) > DENSITY_THRESHOLD)
thickness += dStep;
}
return thickness;
}

// GGX NDF для більш фізично коректного specularu
float ggxD(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159 * d * d);
}

void main() {
    vec3 rayDir    = normalize(inWorldPos - inCameraPos);
    vec3 rayOrigin = inCameraPos;

    // Bounding box з матриці моделі
    vec3 boxCenter = vec3(push.model[3]);
    vec3 boxScale  = vec3(length(push.model[0]), length(push.model[1]), length(push.model[2]));
    vec3 boxMin    = boxCenter - boxScale * 0.5;
    vec3 boxMax    = boxCenter + boxScale * 0.5;

    vec2 tHit = intersectAABB(rayOrigin, rayDir, boxMin, boxMax);
    if (tHit.x > tHit.y || tHit.y < 0.0) discard;

    float stepWorld = STEP_SIZE * max(max(boxScale.x, boxScale.y), boxScale.z);
    float tCurrent  = max(tHit.x, 0.0);

    // --- Raymarching: знайти першу точку поверхні ---
    bool hit = false;
    for (int i = 0; i < MAX_STEPS; i++) {
        if (tCurrent > tHit.y) break;
        if (getDensity(rayOrigin + rayDir * tCurrent, boxMin, boxMax) > DENSITY_THRESHOLD) {
            hit = true;
            break;
        }
        tCurrent += stepWorld;
    }
    if (!hit) discard;

    // --- Binary search refinement (8 ітерацій замість 4 — точніша поверхня) ---
    float t0 = tCurrent - stepWorld, t1 = tCurrent;
    for (int i = 0; i < 8; i++) {
        float m = (t0 + t1) * 0.5;
        if (getDensity(rayOrigin + rayDir * m, boxMin, boxMax) > DENSITY_THRESHOLD)
        t1 = m;
        else
        t0 = m;
    }
    vec3 surfacePos = rayOrigin + rayDir * t1;
    vec3 N = -calcNormal(surfacePos, boxMin, boxMax);
    N = normalize(N);
    if (dot(N, N) < 0.5) N = vec3(0.0, 1.0, 0.0);

    vec3 V = -rayDir;
    vec3 L = SUN_DIR;
    vec3 H = normalize(L + V);

    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    float F0 = pow((IOR_AIR - IOR_WATER) / (IOR_AIR + IOR_WATER), 2.0); // ~0.02
    float F  = fresnel(NdotV, F0);
    F = clamp(F, 0.0, 1.0);

    vec3 reflDir    = reflect(rayDir, N);
    vec3 reflection = sampleSkybox(reflDir);
    float specGGX = ggxD(NdotH, 0.04) * NdotL;
    reflection += SUN_COLOR * clamp(specGGX * 0.15, 0.0, 3.0);


    float thickness  = calcThickness(surfacePos, rayDir, tHit.y - t1, boxMin, boxMax);
    vec3  absorption = exp(-ABSORPTION_COEFF * thickness * 8.0);

    vec3 refrDir = refract(rayDir, N, IOR_AIR / IOR_WATER);
    if (dot(refrDir, refrDir) < 0.01) refrDir = rayDir;
    vec3 background = sampleSkybox(normalize(refrDir));

    float sssWrapped = max(0.0, dot(N, L) * 0.5 + 0.5);
    float sssBack    = max(0.0, dot(-N, L));
    float sssFactor  = (sssWrapped * 0.6 + sssBack * 0.4) * (1.0 - exp(-thickness * 3.0));
    vec3  sss        = SCATTER_COLOR * SUN_COLOR * sssFactor * SCATTER_STRENGTH;


    float foamMask = smoothstep(0.0, 0.3, thickness);
    foamMask *= smoothstep(0.5, 0.95, N.y);


    vec3 refracted = background * absorption + sss;

    vec3 waterColor = mix(refracted, reflection, F);


    waterColor = mix(waterColor, FOAM_COLOR * (NdotL * 0.7 + 0.3), foamMask * 0.7);

    waterColor += SCATTER_COLOR * 0.04;

    waterColor = waterColor / (waterColor + vec3(1.0));

    outColor = vec4(waterColor, 1.0);
}

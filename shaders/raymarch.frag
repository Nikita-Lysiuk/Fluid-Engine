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

// --- ПАРАМЕТРИ З ПРЕЗЕНТАЦІЇ ---
const float DENSITY_THRESHOLD = 0.01;
const float DENSITY_OFFSET = 300.0;
const float DENSITY_MULTIPLIER = 1.0;
const int MAX_STEPS = 128;
const float STEP_SIZE = 0.008;       // Трішки менший крок для деталізації
const vec3 WATER_COLOR = vec3(0.1, 0.5, 0.9);
const vec3 SUN_DIR = normalize(vec3(0.5, 0.8, 0.2));
const float IOR_AIR = 1.0;
const float IOR_WATER = 1.33;
const vec3 ABSORPTION_K = vec3(1.2, 0.6, 0.2); // Коефіцієнти для R, G, B

// --- ДОПОМІЖНІ ФУНКЦІЇ ---

vec2 intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax) {
    vec3 tMin = (boxMin - rayOrigin) / rayDir;
    vec3 tMax = (boxMax - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    return vec2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

float getDensity(vec3 worldPos, vec3 boxMin, vec3 boxMax) {
    vec3 uvw = (worldPos - boxMin) / (boxMax - boxMin);
    if(any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return 0.0;

    // Використовуємо вбудовану лінійну фільтрацію (sampler3D), якщо можливо,
    // або залишаємо твій ручний mix для точності:
    vec3 gridPos = uvw * vec3(sim_params.grid_res.xyz) - 0.5;
    ivec3 iobase = ivec3(floor(gridPos));
    vec3 f = fract(gridPos);

    float v000 = float(texelFetch(densityTex, iobase + ivec3(0,0,0), 0).r);
    float v100 = float(texelFetch(densityTex, iobase + ivec3(1,0,0), 0).r);
    float v010 = float(texelFetch(densityTex, iobase + ivec3(0,1,0), 0).r);
    float v110 = float(texelFetch(densityTex, iobase + ivec3(1,1,0), 0).r);
    float v001 = float(texelFetch(densityTex, iobase + ivec3(0,0,1), 0).r);
    float v101 = float(texelFetch(densityTex, iobase + ivec3(1,0,1), 0).r);
    float v011 = float(texelFetch(densityTex, iobase + ivec3(0,1,1), 0).r);
    float v111 = float(texelFetch(densityTex, iobase + ivec3(1,1,1), 0).r);

    float v0 = mix(mix(v000, v100, f.x), mix(v010, v110, f.x), f.y);
    float v1 = mix(mix(v001, v101, f.x), mix(v011, v111, f.x), f.y);

    return max(0.0, (mix(v0, v1, f.z) - DENSITY_OFFSET) * DENSITY_MULTIPLIER);
}

vec3 calculateNormal(vec3 p, vec3 boxMin, vec3 boxMax) {
    float eps = 0.04;
    vec2 h = vec2(eps, 0.0);
    return normalize(vec3(
                     getDensity(p + h.xyy, boxMin, boxMax) - getDensity(p - h.xyy, boxMin, boxMax),
                     getDensity(p + h.yxy, boxMin, boxMax) - getDensity(p - h.yxy, boxMin, boxMax),
                     getDensity(p + h.yyx, boxMin, boxMax) - getDensity(p - h.yyx, boxMin, boxMax)
                     ));
}

vec3 sampleSkybox(vec3 dir) {
    vec2 uv = vec2(atan(dir.z, dir.x) * 0.1591 + 0.5, asin(dir.y) * 0.3183 + 0.5);
    return texture(skyboxTex, uv).rgb;
}

float calculateThickness(vec3 startP, vec3 dir, float tMax, vec3 bMin, vec3 bMax) {
    float thickness = 0.0;
    float dStep = 0.05;

    for(float t = 0.0; t < tMax; t += dStep) {
        vec3 p = startP + dir * t;
        if (getDensity(p, bMin, bMax) < DENSITY_THRESHOLD)
        break;

        thickness += dStep;
    }
    return thickness;
}

void main() {
    vec3 rayDir = normalize(inWorldPos - inCameraPos);
    vec3 rayOrigin = inCameraPos;

    vec3 boxCenter = vec3(push.model[3]);
    vec3 boxScale = vec3(length(push.model[0]), length(push.model[1]), length(push.model[2]));
    vec3 boxMin = boxCenter - boxScale * 0.5;
    vec3 boxMax = boxCenter + boxScale * 0.5;

    vec2 tHit = intersectAABB(rayOrigin, rayDir, boxMin, boxMax);
    if (tHit.x > tHit.y || tHit.y < 0.0) discard;

    float stepWorld = STEP_SIZE * max(max(boxScale.x, boxScale.y), boxScale.z);
    float tCurrent = max(tHit.x, 0.0);

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

    // Binary Search Refinement
    float t0 = tCurrent - stepWorld, t1 = tCurrent;
    for (int i = 0; i < 4; i++) {
        float m = (t0 + t1) * 0.5;
        if (getDensity(rayOrigin + rayDir * m, boxMin, boxMax) > DENSITY_THRESHOLD) t1 = m; else t0 = m;
    }
    vec3 surfacePos = rayOrigin + rayDir * t1;

    vec3 N = -calculateNormal(surfacePos, boxMin, boxMax);
    if (length(N) < 0.1) N = vec3(0, 1, 0); else N = normalize(N);

    vec3 V = -rayDir;
    vec3 L = SUN_DIR;
    vec3 H = normalize(L + V);

    // 1. Schlick's Fresnel
    float R0 = pow((IOR_AIR - IOR_WATER) / (IOR_AIR + IOR_WATER), 2.0);
    float F = R0 + (1.0 - R0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);

    // 2. Thickness & Beer's Law (Volumetric Absorption)
    float d = calculateThickness(surfacePos, rayDir, tHit.y, boxMin, boxMax);
    vec3 absorption = exp(-ABSORPTION_K * d * 2.0);

    // 3. Diffuse (Wrapped) & Specular
    float diffuse = max(0.0, dot(N, L) * 0.5 + 0.5);
    float spec = pow(max(dot(N, H), 0.0), 256.0);

    // 4. Refraction (Background)
    vec3 refrDir = refract(rayDir, N, IOR_AIR / IOR_WATER);
    vec3 background = sampleSkybox(normalize(refrDir));

    // 5. Final Composition
    vec3 reflection = sampleSkybox(reflect(rayDir, N));

    // Внутрішня частина води (Заломлення + Поглинання + Колір води)
    vec3 waterInterior = background * absorption * WATER_COLOR;
    waterInterior += WATER_COLOR * diffuse * 0.1; // Невелике розсіювання всередині

    // Змішуємо за Френелем
    vec3 finalColor = mix(waterInterior, reflection, F);

    // Додаємо сонячний блік
    finalColor += vec3(1.0, 0.9, 0.8) * spec * 1.5;

    outColor = vec4(finalColor, 1.0);
}
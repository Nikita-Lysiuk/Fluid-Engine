#version 460

layout(location = 0) in vec3 in_color;
layout(location = 0) out vec4 f_color;

void main() {
    vec2 circ_coord = gl_PointCoord * 2.0 - 1.0; 
    float dist_sq = dot(circ_coord, circ_coord);
    
    if (dist_sq > 1.0) {
        discard;
    }
    
    float z = sqrt(1.0 - dist_sq);
    vec3 normal = vec3(circ_coord, z);
    
    vec3 light_dir = normalize(vec3(0.5, 0.5, 1.0));
    float diffuse = max(dot(normal, light_dir), 0.2);

    f_color = vec4(in_color * diffuse, 1.0);
}
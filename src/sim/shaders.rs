pub(super) mod density_alpha_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\density_and_alpha.comp");
}

pub(super) mod viscosity_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\viscosity.comp");
}

pub(super) mod density_source_term_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\density_source_term.comp");
}

pub(super) mod divergence_source_term_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\divergence_source_term.comp");
}

pub(super) mod pressure_integration_shader {
    use vulkano_shaders::shader;
    shader! {
        ty: "compute",
        path: "shaders\\compute\\pressure_integration.comp"
    }
}

pub(super) mod divergence_integration_shader {
    use vulkano_shaders::shader;
    shader! {
        ty: "compute",
        path: "shaders\\compute\\divergence_integration.comp"
    }
}

pub(super) mod pressure_force_shader {
    use vulkano_shaders::shader;
    shader! { ty: "compute", path: "shaders\\compute\\pressure_force.comp" }
}

pub(super) mod pressure_update_shader {
    use vulkano_shaders::shader;
    shader!(
        ty: "compute",
        path: "shaders\\compute\\pressure_update.comp"
    );
}

pub(super) mod stats_shader {
    use vulkano_shaders::shader;
    shader! { ty: "compute", path: "shaders\\compute\\stats.comp" }
}

pub(super) mod solver_check_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders/compute/solver_check.comp");
}


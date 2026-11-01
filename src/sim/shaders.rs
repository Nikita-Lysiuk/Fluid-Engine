pub(crate) mod density_alpha_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\density_and_alpha.comp");
}

pub(crate) mod viscosity_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\viscosity.comp");
}

pub(crate) mod density_source_term_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\density_source_term.comp");
}

pub(crate) mod divergence_source_term_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\divergence_source_term.comp");
}

pub(crate) mod pressure_integration_shader {
    use vulkano_shaders::shader;
    shader! {
        ty: "compute",
        path: "shaders\\compute\\pressure_integration.comp"
    }
}

pub(crate) mod divergence_integration_shader {
    use vulkano_shaders::shader;
    shader! {
        ty: "compute",
        path: "shaders\\compute\\divergence_integration.comp"
    }
}

pub(crate) mod pressure_force_shader {
    use vulkano_shaders::shader;
    shader! { ty: "compute", path: "shaders\\compute\\pressure_force.comp" }
}

pub(crate) mod pressure_update_shader {
    use vulkano_shaders::shader;
    shader!(
        ty: "compute",
        path: "shaders\\compute\\pressure_update.comp"
    );
}

pub(crate) mod stats_shader {
    use vulkano_shaders::shader;
    shader! { ty: "compute", path: "shaders\\compute\\stats.comp" }
}

pub(crate) mod solver_check_shader {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders/compute/solver_check.comp");
}

pub(crate) mod cs_hash {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\spatial_hash.comp");
}
pub(crate) mod cs_offsets {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\grid_offsets.comp");
}
pub(crate) mod cs_reorder {
    use vulkano_shaders::shader;
    shader!(ty: "compute", path: "shaders\\compute\\reorder.comp");
}


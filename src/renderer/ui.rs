use egui::{Context, Slider, Window};
use glam::Vec4;
use crate::core::scene::Scene;
use crate::renderer::pipelines::SortAlgorithm;

#[derive(PartialEq, Clone, Copy)]
pub enum RenderMode {
    Raymarching,
    Particles,
}

pub struct AppUI {
    pub show_controls: bool,
    pub render_mode: RenderMode,
    pub sort_algorithm: SortAlgorithm,
    pub use_cfl: bool,
    pub display_max_speed: f32,
    pub display_cfl_dt: f32,

    pub use_solver_error_threshold: bool,
    /// η  — max density error as % of ρ₀  (paper default 0.1 %)
    pub density_error_pct: f32,
    /// ηv — max divergence error as % of ρ₀ (paper default 0.5 %)
    pub divergence_error_pct: f32,
    pub display_avg_density_error: f32,
    pub display_avg_divergence_error: f32,
    pub display_density_iters_used: u32,
    pub display_divergence_iters_used: u32,
}

impl AppUI {
    pub fn new() -> Self {
        Self {
            show_controls: true,
            render_mode: RenderMode::Raymarching,
            sort_algorithm: SortAlgorithm::Radix,
            use_cfl: false,
            display_max_speed: 0.0,
            display_cfl_dt: 0.0,

            use_solver_error_threshold: false,
            density_error_pct: 0.1,
            divergence_error_pct: 0.5,
            display_avg_density_error: 0.0,
            display_avg_divergence_error: 0.0,
            display_density_iters_used: 0,
            display_divergence_iters_used: 0,
        }
    }
    pub fn render(&mut self, ctx: &Context, scene: &mut Scene, fps: u32) {
        if !self.show_controls {
            return;
        }

        Window::new("Fluid Simulation Controls")
            .default_width(150.0)
            .resizable(true)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.heading("Simulation Parameters");
                ui.add(Slider::new(&mut scene.sim_params.viscosity, 0.0..=0.5).text("Viscosity"));
                ui.add(Slider::new(&mut scene.sim_params.target_density, 500.0..=2000.0).text("Target Density"));
                ui.add(Slider::new(&mut scene.sim_params.relax_factor, 0.0..=1.0).text("Relaxation Factor"));
                ui.add(Slider::new(&mut scene.sim_params.smoothing_radius, 0.0001..=0.2).text("Smoothing Radius"));
                ui.add(Slider::new(&mut scene.sim_params.density_solver_iterations, 1..=100).text("Density Max Iters"));
                ui.add(Slider::new(&mut scene.sim_params.divergence_solver_iterations, 1..=100).text("Divergence Max Iters"));
                ui.add(Slider::new(&mut scene.sim_params.dt, 0.0001..=0.1).text("Time Step (dt)"));

                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.use_cfl, "Adaptive CFL dt");
                });
                ui.label(format!("Max speed:  {:.3} m/s", self.display_max_speed));
                ui.label(format!("CFL limit:  {:.4} s", self.display_cfl_dt));

                ui.separator();

                ui.heading("Solver Convergence (DFSPH)");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.use_solver_error_threshold, "Early-exit by error");
                });
                if self.use_solver_error_threshold {
                    let rho0 = scene.sim_params.target_density;
                    ui.add(Slider::new(&mut self.density_error_pct, 0.01..=5.0)
                        .suffix(" %")
                        .text("η  (density)"));
                    ui.label(format!("  = {:.3} kg/m³", self.density_error_pct / 100.0 * rho0));
                    ui.add(Slider::new(&mut self.divergence_error_pct, 0.01..=5.0)
                        .suffix(" %")
                        .text("ηv (divergence)"));
                    ui.label(format!("  = {:.3} kg/m³/s", self.divergence_error_pct / 100.0 * rho0));
                }
                ui.label(format!("Avg Δρ:      {:.4} kg/m³  ({} iters)",
                    self.display_avg_density_error, self.display_density_iters_used));
                ui.label(format!("Avg Dρ/Dt:   {:.4} kg/m³/s ({} iters)",
                    self.display_avg_divergence_error, self.display_divergence_iters_used));

                ui.separator();

                ui.heading("External Forces");
                ui.add(Slider::new(&mut scene.sim_params.gravity[1], -100.0..=100.0).text("Gravity Y"));

                ui.separator();

                ui.heading("Collision Boundary");
                ui.label("Control the simulation box size:");

                ui.group(|ui| {
                    ui.label("Min Coordinates:");
                    ui.vertical(|ui| {
                        ui.add(Slider::new(&mut scene.boundary.min.x, -5.0..=-(scene.sim_params.particle_radius / 2.0)).text("X"));
                        ui.add(Slider::new(&mut scene.boundary.min.y, -2.0..=(scene.sim_params.particle_radius / 2.0)).text("Y"));
                        ui.add(Slider::new(&mut scene.boundary.min.z, -5.0..=-(scene.sim_params.particle_radius / 2.0)).text("Z"));
                    });
                });

                ui.group(|ui| {
                    ui.label("Max Coordinates:");
                    ui.vertical(|ui| {
                        ui.add(Slider::new(&mut scene.boundary.max.x, (scene.sim_params.particle_radius / 2.0)..=5.0).text("X"));
                        ui.add(Slider::new(&mut scene.boundary.max.y, (scene.sim_params.particle_radius / 2.0)..=10.0).text("Y"));
                        ui.add(Slider::new(&mut scene.boundary.max.z, (scene.sim_params.particle_radius / 2.0)..=5.0).text("Z"));
                    });
                });

                scene.sim_params.box_min = Vec4::new(scene.boundary.min.x, scene.boundary.min.y, scene.boundary.min.z, 0.0).into();
                scene.sim_params.box_max = Vec4::new(scene.boundary.max.x, scene.boundary.max.y, scene.boundary.max.z, 0.0).into();

                ui.separator();

                ui.heading("Rendering Parameters");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.render_mode, RenderMode::Raymarching, "Raymarching");
                    ui.selectable_value(&mut self.render_mode, RenderMode::Particles, "Particles");
                });

                ui.separator();
                ui.heading("Sort Algorithm");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.sort_algorithm, SortAlgorithm::Bitonic, "Bitonic");
                    ui.selectable_value(&mut self.sort_algorithm, SortAlgorithm::Radix, "Radix");
                });

                ui.separator();
                ui.label("Adjust the resolution of the water texture:");

                ui.group(|ui| {
                    ui.label("Water texture resolution:");
                    ui.vertical(|ui| {
                        ui.add(Slider::new(&mut scene.sim_params.grid_res[0], 16..=256).text("X"));
                        ui.add(Slider::new(&mut scene.sim_params.grid_res[1], 16..=256).text("Y"));
                        ui.add(Slider::new(&mut scene.sim_params.grid_res[2], 16..=256).text("Z"));
                    });
                });

                ui.separator();
                ui.label(format!("Active Particles: {}", scene.initial_positions.len()));

                ui.separator();

                ui.label(format!("FPS: {}", fps));

                if ui.button("Reset Simulation (Console Log)").clicked() {
                    println!("Reset logic to be implemented!");
                }
            });
    }
}

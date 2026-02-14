use egui::{Context, Slider, Window};
use glam::Vec4;
use crate::core::scene::Scene;

pub struct AppUI {
    pub show_controls: bool,
}

impl AppUI {
    pub fn new() -> Self {
        Self {
            show_controls: true,
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
                ui.add(Slider::new(&mut scene.sim_params.density_solver_iterations, 1..=100).text("Density Solver Iterations"));
                ui.add(Slider::new(&mut scene.sim_params.divergence_solver_iterations, 1..=100).text("Divergence Solver Iterations"));
                ui.add(Slider::new(&mut scene.sim_params.dt, 0.0001..=0.1).text("Time Step (dt)"));

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
                ui.label(format!("Active Particles: {}", scene.initial_positions.len()));

                ui.separator();

                ui.label(format!("FPS: {}", fps));

                if ui.button("Reset Simulation (Console Log)").clicked() {
                    println!("Reset logic to be implemented!");
                }
            });
    }
}
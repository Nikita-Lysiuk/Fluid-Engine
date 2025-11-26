use log::error;
use fluid_engine::engine::engine::Engine;

fn main() {
    let mut engine = Engine::new();
    let event_loop = engine.event_loop();
    if let Err(e) = event_loop.run_app(&mut engine) {
        error!("Event loop error: {}", e);
    }
}

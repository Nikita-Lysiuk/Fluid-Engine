
use fluid_engine::engine::engine::Engine;

fn main() -> anyhow::Result<()> {
    let mut engine =  Engine::new()?;
    let event_loop = engine.event_loop();
    event_loop.run_app(&mut engine)?;
    Ok(())
}

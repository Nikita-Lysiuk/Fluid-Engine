
use fluid_engine::core::engine::Engine;

fn main() -> anyhow::Result<()> {
    let mut engine =  Engine::new().map_err(|e| anyhow::anyhow!("Failed to initialize Engine: {}", e))?;
    engine.run();
    Ok(())
}

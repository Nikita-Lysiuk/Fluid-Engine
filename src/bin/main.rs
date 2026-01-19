use std::alloc::System;
use fluid_engine::core::engine::Engine;

#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<System> =
    tracy_client::ProfiledAllocator::new(System, 100);

fn main() -> anyhow::Result<()> {
    let mut engine =  Engine::new().map_err(|e| anyhow::anyhow!("Failed to initialize Engine: {}", e))?;
    engine.run();
    Ok(())
}

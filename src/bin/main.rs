use fluid_engine::app::App;
use std::alloc::System;

#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<System> =
    tracy_client::ProfiledAllocator::new(System, 100);

fn main() -> anyhow::Result<()> {
    let mut engine = App::new().map_err(|e| anyhow::anyhow!("Failed to initialize App: {}", e))?;
    engine.run();
    Ok(())
}

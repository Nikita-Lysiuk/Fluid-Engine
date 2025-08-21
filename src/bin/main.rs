use log::error;
use fluid_engine::engine::engine::Engine;

fn main() {

    let engine = Engine::new();
    if let Err(err) = engine.game_loop() {
        error!("error in game loop {}", err);
    }
}

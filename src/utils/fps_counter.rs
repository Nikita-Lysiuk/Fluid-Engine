use std::time::{Duration, Instant};

pub struct FpsCounter {
    last_updated_time: Instant,
    last_frame_time: Instant,
    frame_count: u32,
    current_fps: u32,
    target_frame_count: u32,
}

impl FpsCounter {
    pub fn new(preferred_fps: u32) -> Self {
        let now = Instant::now();
        FpsCounter {
            last_updated_time: now,
            last_frame_time: now,
            frame_count: 0,
            current_fps: 0,
            target_frame_count: preferred_fps,
        }
    }
    
    pub fn tick(&mut self) -> Duration {
        let now = Instant::now();
        let delta_time = now.duration_since(self.last_frame_time);
        
        self.last_frame_time = now;
        self.frame_count += 1;
        
        if self.frame_count >= self.target_frame_count {
            let elapsed = now.duration_since(self.last_updated_time);
            
            self.current_fps = (self.target_frame_count as f64 / elapsed.as_secs_f64()).round() as u32;
            
            self.frame_count = 0;
            self.last_updated_time = now;
        }
        
        delta_time
    }
    
    pub fn fps(&self) -> u32 {
        self.current_fps
    }
}
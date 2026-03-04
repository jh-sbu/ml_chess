#![allow(dead_code)]

// Implemented in Phase 7.

use std::time::Duration;

use crate::agents::Agent;
use crate::chess::{GameState, Move};

pub struct MinimaxAgent {
    pub depth: u32,
    name: String,
}

impl MinimaxAgent {
    pub fn new(depth: u32) -> Self {
        Self { depth, name: format!("Minimax(d{})", depth) }
    }
}

impl Agent for MinimaxAgent {
    fn select_move(&mut self, _state: &GameState, _time_budget: Option<Duration>) -> Move {
        unimplemented!("MinimaxAgent is implemented in Phase 7")
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#![allow(dead_code)]

// Implemented in Phase 7.

use std::time::Duration;

use crate::agents::Agent;
use crate::chess::{GameState, Move};

pub struct NegamaxAgent {
    pub depth: u32,
    name: String,
}

impl NegamaxAgent {
    pub fn new(depth: u32) -> Self {
        Self { depth, name: format!("Negamax(d{})", depth) }
    }
}

impl Agent for NegamaxAgent {
    fn select_move(&mut self, _state: &GameState, _time_budget: Option<Duration>) -> Move {
        unimplemented!("NegamaxAgent is implemented in Phase 7")
    }

    fn name(&self) -> &str {
        &self.name
    }
}

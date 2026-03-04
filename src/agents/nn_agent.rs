#![allow(dead_code)]

// Implemented in Phase 10.

use std::time::Duration;

use crate::agents::Agent;
use crate::chess::{GameState, Move};

pub struct NNAgent {
    name: String,
}

impl Agent for NNAgent {
    fn select_move(&mut self, _state: &GameState, _time_budget: Option<Duration>) -> Move {
        unimplemented!("NNAgent is implemented in Phase 10")
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#![allow(dead_code)]

use rand::seq::IndexedRandom;

use crate::agents::Agent;
use crate::chess::{GameState, Move};
use std::time::Duration;

pub struct RandomAgent {
    name: String,
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self {
            name: "Random".to_string(),
        }
    }
}

impl Agent for RandomAgent {
    fn select_move(&mut self, state: &GameState, _time_budget: Option<Duration>) -> Move {
        let moves = state.legal_moves();
        let mut rng = rand::rng();
        moves
            .choose(&mut rng)
            .expect("select_move called on terminal position")
            .clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::GameState;

    #[test]
    fn random_agent_returns_legal_move() {
        let state = GameState::new();
        let legal = state.legal_moves();
        let mut agent = RandomAgent::default();
        let chosen = agent.select_move(&state, None);
        assert!(legal.contains(&chosen));
    }
}

#![allow(dead_code)]

use std::sync::mpsc;
use std::time::Duration;

use crate::agents::Agent;
use crate::chess::{GameState, Move};

pub struct HumanAgent {
    name: String,
    move_rx: mpsc::Receiver<Move>,
}

impl HumanAgent {
    pub fn new(name: impl Into<String>, move_rx: mpsc::Receiver<Move>) -> Self {
        Self {
            name: name.into(),
            move_rx,
        }
    }
}

impl Agent for HumanAgent {
    fn select_move(&mut self, _state: &GameState, _time_budget: Option<Duration>) -> Move {
        self.move_rx
            .recv()
            .expect("HumanAgent move channel disconnected")
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::GameState;
    use crate::chess::moves::parse_uci;
    use std::sync::mpsc;

    #[test]
    fn human_agent_returns_sent_move() {
        let state = GameState::new();
        let (tx, rx) = mpsc::channel();
        let mv = parse_uci(&state.position, "e2e4").unwrap();
        tx.send(mv.clone()).unwrap();
        let mut agent = HumanAgent::new("TestHuman", rx);
        let received = agent.select_move(&state, None);
        assert_eq!(received, mv);
    }
}

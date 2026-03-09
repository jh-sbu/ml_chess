#![allow(dead_code, unused_imports)]

pub mod book;
pub mod human;
pub mod mcts;
pub mod minimax;
pub mod negamax;
pub mod nn_agent;
pub mod random;

pub use book::{BookAgent, OpeningBook};
pub use human::HumanAgent;
pub use mcts::MCTSAgent;
pub use minimax::MinimaxAgent;
pub use negamax::NegamaxAgent;
pub use nn_agent::NNAgent;
pub use random::RandomAgent;

use crate::chess::{GameState, Move};
use std::time::Duration;

pub trait Agent {
    fn select_move(&mut self, state: &GameState, time_budget: Option<Duration>) -> Move;
    fn name(&self) -> &str;
}

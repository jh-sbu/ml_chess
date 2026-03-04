#![allow(dead_code, unused_imports)]

pub mod human;
pub mod minimax;
pub mod negamax;
pub mod nn_agent;
pub mod random;

pub use human::HumanAgent;
pub use random::RandomAgent;

use crate::chess::{GameState, Move};
use std::time::Duration;

pub trait Agent {
    fn select_move(&mut self, state: &GameState, time_budget: Option<Duration>) -> Move;
    fn name(&self) -> &str;
}

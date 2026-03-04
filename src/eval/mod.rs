#![allow(dead_code, unused_imports)]

pub mod classical;
pub mod neural;

pub use classical::evaluate;
pub use neural::evaluate_nn;

pub type Score = i32;
pub const MATE_SCORE: Score = 1_000_000;
pub const DRAW_SCORE: Score = 0;

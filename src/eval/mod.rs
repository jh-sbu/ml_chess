#![allow(dead_code, unused_imports)]

pub mod classical;
pub mod neural;
pub mod policy;

pub use classical::evaluate;
pub use policy::{ChessPolicyNet, ChessPolicyNetConfig, encode_move, policy_net_forward};

pub type Score = i32;
pub const MATE_SCORE: Score = 1_000_000;
pub const DRAW_SCORE: Score = 0;

#![allow(dead_code, unused_imports)]

pub mod board;
pub mod moves;
pub mod notation;
pub mod rules;

pub use board::{BoardError, GameState};
pub use moves::MoveError;
pub use rules::GameResult;
pub use shakmaty::{Chess, Color, Move, Piece, Position, Role, Square};

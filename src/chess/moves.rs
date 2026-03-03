#![allow(dead_code)]

use shakmaty::san::{ParseSanError, San, SanError, SanPlus};
use shakmaty::uci::{IllegalUciMoveError, ParseUciMoveError, UciMove};
use shakmaty::{Chess, Move};

#[derive(Debug, thiserror::Error)]
pub enum MoveError {
    #[error("invalid UCI '{0}': {1}")]
    InvalidUci(String, #[source] ParseUciMoveError),
    #[error("illegal UCI '{0}': {1}")]
    IllegalUci(String, #[source] IllegalUciMoveError),
    #[error("invalid SAN '{0}': {1}")]
    InvalidSan(String, #[source] ParseSanError),
    #[error("illegal SAN '{0}': {1}")]
    IllegalSan(String, #[source] SanError),
}

pub fn parse_uci(position: &Chess, s: &str) -> Result<Move, MoveError> {
    let uci: UciMove = s
        .parse()
        .map_err(|e| MoveError::InvalidUci(s.to_string(), e))?;
    uci.to_move(position)
        .map_err(|e| MoveError::IllegalUci(s.to_string(), e))
}

pub fn parse_san(position: &Chess, s: &str) -> Result<Move, MoveError> {
    let san: San = s
        .parse()
        .map_err(|e| MoveError::InvalidSan(s.to_string(), e))?;
    san.to_move(position)
        .map_err(|e| MoveError::IllegalSan(s.to_string(), e))
}

pub fn move_to_san(position: &Chess, mv: &Move) -> String {
    SanPlus::from_move(position.clone(), mv).to_string()
}

pub fn move_to_uci(mv: &Move) -> String {
    UciMove::from_standard(mv).to_string()
}

#[cfg(test)]
mod tests {
    use shakmaty::Chess;

    fn perft_count(depth: u32) -> u64 {
        shakmaty::perft(&Chess::default(), depth)
    }

    #[test]
    fn perft_depth_1() {
        assert_eq!(perft_count(1), 20);
    }

    #[test]
    fn perft_depth_2() {
        assert_eq!(perft_count(2), 400);
    }

    #[test]
    fn perft_depth_3() {
        assert_eq!(perft_count(3), 8902);
    }
}

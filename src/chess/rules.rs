#![allow(dead_code)]

use shakmaty::{Chess, Color, Outcome, Position};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    WhiteWins,
    BlackWins,
    Draw,
}

impl From<Outcome> for GameResult {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Outcome::Decisive {
                winner: Color::White,
            } => GameResult::WhiteWins,
            Outcome::Decisive {
                winner: Color::Black,
            } => GameResult::BlackWins,
            Outcome::Draw => GameResult::Draw,
        }
    }
}

pub fn game_result(pos: &Chess, halfmove_clock: u32) -> Option<GameResult> {
    pos.outcome()
        .map(GameResult::from)
        .or_else(|| is_fifty_move_draw(halfmove_clock).then_some(GameResult::Draw))
        .or_else(|| pos.is_insufficient_material().then_some(GameResult::Draw))
}

pub fn is_fifty_move_draw(halfmove_clock: u32) -> bool {
    halfmove_clock >= 100
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Chess;

    #[test]
    fn initial_position_is_not_over() {
        let pos = Chess::default();
        assert_eq!(game_result(&pos, 0), None);
    }

    #[test]
    fn fifty_move_threshold() {
        assert!(!is_fifty_move_draw(99));
        assert!(is_fifty_move_draw(100));
        assert!(is_fifty_move_draw(101));
    }
}

#![allow(dead_code)]

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position};

#[derive(Debug, Clone)]
pub struct GameState {
    pub position: Chess,
    pub history: Vec<Move>,
}

#[derive(Debug, thiserror::Error)]
pub enum BoardError {
    #[error("illegal move: {0}")]
    IllegalMove(String),
    #[error("FEN parse error: {0}")]
    FenParse(String),
}

impl GameState {
    pub fn new() -> Self {
        Self {
            position: Chess::default(),
            history: Vec::new(),
        }
    }

    pub fn from_fen(s: &str) -> Result<Self, BoardError> {
        let fen: Fen = s
            .parse()
            .map_err(|e: shakmaty::fen::ParseFenError| BoardError::FenParse(e.to_string()))?;
        let position = fen
            .into_position(CastlingMode::Standard)
            .map_err(|e| BoardError::FenParse(e.to_string()))?;
        Ok(Self {
            position,
            history: Vec::new(),
        })
    }

    pub fn to_fen(&self) -> String {
        Fen::from_position(self.position.clone(), EnPassantMode::Legal).to_string()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        self.position.legal_moves().into_iter().collect()
    }

    pub fn is_checkmate(&self) -> bool {
        self.position.is_checkmate()
    }

    pub fn is_stalemate(&self) -> bool {
        self.position.is_stalemate()
    }

    pub fn outcome(&self) -> Option<shakmaty::Outcome> {
        self.position.outcome()
    }

    pub fn side_to_move(&self) -> Color {
        self.position.turn()
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.position.halfmoves()
    }

    pub fn fullmove_number(&self) -> u32 {
        self.position.fullmoves().get()
    }

    pub fn apply_move(&mut self, mv: Move) -> Result<(), BoardError> {
        let new_pos = self
            .position
            .clone()
            .play(&mv)
            .map_err(|_| BoardError::IllegalMove(mv.to_string()))?;
        self.position = new_pos;
        self.history.push(mv);
        Ok(())
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::moves::parse_uci;

    #[test]
    fn initial_position_has_20_legal_moves() {
        let state = GameState::new();
        assert_eq!(state.legal_moves().len(), 20);
    }

    #[test]
    fn fen_roundtrip() {
        let fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        let state = GameState::from_fen(fen).expect("valid FEN");
        let roundtrip = state.to_fen();
        let state2 = GameState::from_fen(&roundtrip).expect("roundtrip FEN");
        assert_eq!(state2.to_fen(), roundtrip);
    }

    #[test]
    fn fool_mate() {
        let mut state = GameState::new();
        for uci in &["f2f3", "e7e5", "g2g4", "d8h4"] {
            let mv = parse_uci(&state.position, uci).expect("valid move");
            state.apply_move(mv).expect("legal move");
        }
        assert!(state.is_checkmate());
    }
}

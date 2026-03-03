#![allow(dead_code)]

use std::fmt::Write as FmtWrite;

use anyhow::Context as _;
use shakmaty::fen::{Epd, Fen};
use shakmaty::san::SanPlus;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position};

pub fn fen_to_position(fen: &str) -> anyhow::Result<Chess> {
    let parsed: Fen = fen.parse().context("invalid FEN string")?;
    let pos = parsed
        .into_position(CastlingMode::Standard)
        .map_err(|e| anyhow::anyhow!("invalid position: {}", e))?;
    Ok(pos)
}

pub fn position_to_epd(pos: &Chess) -> String {
    Epd::from_position(pos.clone(), EnPassantMode::Legal).to_string()
}

pub fn moves_to_pgn(start: &Chess, moves: &[Move]) -> String {
    let mut pgn = String::new();
    let mut pos = start.clone();

    for mv in moves {
        if pos.turn() == Color::White {
            let _ = write!(pgn, "{}. ", pos.fullmoves().get());
        }
        let san = SanPlus::from_move_and_play_unchecked(&mut pos, mv);
        let _ = write!(pgn, "{} ", san);
    }

    pgn.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::moves::parse_uci;

    #[test]
    fn pgn_e4_e5() {
        let start = Chess::default();
        let mut pos = start.clone();
        let mut moves = Vec::new();
        for uci in &["e2e4", "e7e5"] {
            let mv = parse_uci(&pos, uci).expect("valid");
            pos = pos.play(&mv).expect("legal");
            moves.push(mv);
        }
        let pgn = moves_to_pgn(&start, &moves);
        assert_eq!(pgn, "1. e4 e5");
    }

    #[test]
    fn pgn_fools_mate() {
        let start = Chess::default();
        let mut pos = start.clone();
        let mut moves = Vec::new();
        for uci in &["f2f3", "e7e5", "g2g4", "d8h4"] {
            let mv = parse_uci(&pos, uci).expect("valid");
            pos = pos.play(&mv).expect("legal");
            moves.push(mv);
        }
        let pgn = moves_to_pgn(&start, &moves);
        assert_eq!(pgn, "1. f3 e5 2. g4 Qh4#");
    }
}

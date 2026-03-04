#![allow(dead_code)]

use shakmaty::{Chess, Color, Position, Role};

use super::{DRAW_SCORE, MATE_SCORE, Score};

fn material_value(role: Role) -> Score {
    match role {
        Role::Pawn => 100,
        Role::Knight => 320,
        Role::Bishop => 330,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20_000,
    }
}

#[rustfmt::skip]
const PAWN_PST: [Score; 64] = [
    //  A    B    C    D    E    F    G    H
       0,   0,   0,   0,   0,   0,   0,   0,  // rank 1
       5,  10,  10, -20, -20,  10,  10,   5,  // rank 2
       5,  -5, -10,   0,   0, -10,  -5,   5,  // rank 3
       0,   0,   0,  20,  20,   0,   0,   0,  // rank 4
       5,   5,  10,  25,  25,  10,   5,   5,  // rank 5
      10,  10,  20,  30,  30,  20,  10,  10,  // rank 6
      50,  50,  50,  50,  50,  50,  50,  50,  // rank 7
       0,   0,   0,   0,   0,   0,   0,   0,  // rank 8
];

#[rustfmt::skip]
const KNIGHT_PST: [Score; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
const BISHOP_PST: [Score; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
const ROOK_PST: [Score; 64] = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const QUEEN_PST: [Score; 64] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
];

#[rustfmt::skip]
const KING_MIDGAME_PST: [Score; 64] = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
];

fn pst_for(role: Role) -> &'static [Score; 64] {
    match role {
        Role::Pawn => &PAWN_PST,
        Role::Knight => &KNIGHT_PST,
        Role::Bishop => &BISHOP_PST,
        Role::Rook => &ROOK_PST,
        Role::Queen => &QUEEN_PST,
        Role::King => &KING_MIDGAME_PST,
    }
}

/// Evaluate a position from White's perspective in centipawns.
pub fn evaluate(pos: &Chess) -> Score {
    if pos.is_checkmate() {
        return if pos.turn() == Color::White {
            -MATE_SCORE
        } else {
            MATE_SCORE
        };
    }

    if pos.is_stalemate() || pos.is_insufficient_material() {
        return DRAW_SCORE;
    }

    let mut score: Score = 0;
    for (sq, piece) in pos.board().iter() {
        let pst_idx = if piece.color == Color::White {
            usize::from(sq)
        } else {
            usize::from(sq.flip_vertical())
        };
        let piece_value = material_value(piece.role) + pst_for(piece.role)[pst_idx];
        score += if piece.color == Color::White {
            piece_value
        } else {
            -piece_value
        };
    }

    score
}

#[cfg(test)]
mod tests {
    use shakmaty::{CastlingMode, Chess, fen::Fen};

    use crate::chess::{board::GameState, moves::parse_uci};

    use super::*;

    fn pos_from_fen(fen: &str) -> Chess {
        let fen: Fen = fen.parse().expect("valid FEN");
        fen.into_position(CastlingMode::Standard)
            .expect("legal position")
    }

    #[test]
    fn initial_position_is_zero() {
        let pos = Chess::default();
        assert_eq!(evaluate(&pos), 0);
    }

    #[test]
    fn scholars_mate_white_wins() {
        // After 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7# — Black to move, is mated
        let pos =
            pos_from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
        assert_eq!(evaluate(&pos), MATE_SCORE);
    }

    #[test]
    fn fools_mate_white_loses() {
        // 1.f3 e5 2.g4 Qh4# — White to move, is mated
        let mut state = GameState::new();
        for mv in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            let m = parse_uci(&state.position, mv).expect("valid move");
            state.apply_move(m).expect("legal move");
        }
        assert_eq!(evaluate(&state.position), -MATE_SCORE);
    }

    #[test]
    fn stalemate_is_draw() {
        let pos = pos_from_fen("k7/8/1QK5/8/8/8/8/8 b - - 0 1");
        assert_eq!(evaluate(&pos), DRAW_SCORE);
    }

    #[test]
    fn extra_queen_for_white_is_positive() {
        // Black queen removed from starting position
        let pos = pos_from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(evaluate(&pos) > 800);
    }
}

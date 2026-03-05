#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Position};

use crate::agents::Agent;
use crate::chess::{GameState, Move};
use crate::eval::classical::evaluate;
use crate::eval::{MATE_SCORE, Score};

type Tt = HashMap<u64, (Score, u32)>;

fn negamax(
    pos: &Chess,
    depth: u32,
    mut alpha: Score,
    beta: Score,
    deadline: Instant,
    tt: &mut Tt,
) -> Option<Score> {
    if Instant::now() >= deadline {
        return None;
    }

    let hash = pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0;

    if let Some(&(score, cached_depth)) = tt.get(&hash)
        && cached_depth >= depth
    {
        return Some(score);
    }

    let moves: Vec<Move> = pos.legal_moves().into_iter().collect();

    if depth == 0 || moves.is_empty() {
        let raw = evaluate(pos);
        return Some(if pos.turn() == Color::White {
            raw
        } else {
            -raw
        });
    }

    let mut moves = moves;
    moves.sort_by_key(|mv| if mv.is_capture() { 0 } else { 1 });

    let mut best = -MATE_SCORE;
    for mv in &moves {
        if Instant::now() >= deadline {
            return None;
        }
        let new_pos = pos.clone().play(mv).expect("legal move");
        let score = -negamax(&new_pos, depth - 1, -beta, -alpha, deadline, tt)?;
        if score > best {
            best = score;
        }
        alpha = alpha.max(best);
        if alpha >= beta {
            break;
        }
    }

    tt.insert(hash, (best, depth));
    Some(best)
}

pub struct NegamaxAgent {
    pub depth: u32,
    name: String,
}

impl NegamaxAgent {
    pub fn new(depth: u32) -> Self {
        Self {
            depth,
            name: format!("Negamax(d{})", depth),
        }
    }
}

impl Agent for NegamaxAgent {
    fn select_move(&mut self, state: &GameState, time_budget: Option<Duration>) -> Move {
        let deadline = Instant::now() + time_budget.unwrap_or(Duration::from_secs(5));
        let mut tt: Tt = HashMap::new();

        let mut moves = state.legal_moves();
        moves.sort_by_key(|mv| if mv.is_capture() { 0 } else { 1 });

        let mut best_move = moves[0].clone();

        'outer: for d in 1..=self.depth {
            let mut iter_best_score = -MATE_SCORE;
            let mut iter_best_move = moves[0].clone();
            let mut alpha = -MATE_SCORE;
            let mut completed = true;

            for mv in &moves {
                if Instant::now() >= deadline {
                    completed = false;
                    break;
                }
                let new_pos = state.position.clone().play(mv).expect("legal move");
                let score = negamax(&new_pos, d - 1, -MATE_SCORE, -alpha, deadline, &mut tt);
                let Some(score) = score else {
                    completed = false;
                    break;
                };
                let score = -score;

                if score > iter_best_score {
                    iter_best_score = score;
                    iter_best_move = mv.clone();
                }
                alpha = alpha.max(score);
            }

            if completed {
                best_move = iter_best_move;
            } else {
                break 'outer;
            }
        }

        best_move
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::chess::GameState;

    #[test]
    fn finds_mate_in_one() {
        // Scholar's mate setup — White has Qxf7# available
        let mut state = GameState::from_fen(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        )
        .expect("valid FEN");
        let mut agent = NegamaxAgent::new(2);
        let mv = agent.select_move(&state, Some(Duration::from_secs(5)));
        state.apply_move(mv).expect("legal");
        assert!(state.is_checkmate());
    }
}

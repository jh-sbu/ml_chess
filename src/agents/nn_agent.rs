#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Position};

use crate::agents::Agent;
use crate::chess::{GameState, Move};
use crate::eval::MATE_SCORE;
use crate::eval::Score;
use crate::eval::neural::{ChessValueNet, ChessValueNetConfig, evaluate_nn};

type Tt = HashMap<u64, (Score, u32)>;

fn nn_negamax(
    pos: &Chess,
    model: &ChessValueNet<NdArray>,
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
        let state = GameState::from_pos(pos.clone());
        let raw = evaluate_nn(&state, model);
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
        let score = -nn_negamax(&new_pos, model, depth - 1, -beta, -alpha, deadline, tt)?;
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

pub struct NNAgent {
    model: ChessValueNet<NdArray>,
    search_depth: u32,
    name: String,
}

impl NNAgent {
    /// Create an agent with freshly initialized (random) weights.
    pub fn new_random(depth: u32) -> Self {
        let device = NdArrayDevice::default();
        let model = ChessValueNetConfig::new().init::<NdArray>(&device);
        Self {
            model,
            search_depth: depth,
            name: format!("NNAgent(d{depth})"),
        }
    }

    /// Load a trained model checkpoint. Path is given without extension (Burn convention).
    pub fn load(path: &Path, depth: u32) -> anyhow::Result<Self> {
        let device = NdArrayDevice::default();
        let base = ChessValueNetConfig::new().init::<NdArray>(&device);
        let record = CompactRecorder::new()
            .load(path.into(), &device)
            .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {e}", path.display()))?;
        let model = base.load_record(record);
        Ok(Self {
            model,
            search_depth: depth,
            name: format!("NNAgent(d{depth})"),
        })
    }
}

impl Agent for NNAgent {
    fn select_move(&mut self, state: &GameState, time_budget: Option<Duration>) -> Move {
        let deadline = Instant::now() + time_budget.unwrap_or(Duration::from_secs(5));
        let mut tt: Tt = HashMap::new();

        let mut moves = state.legal_moves();
        moves.sort_by_key(|mv| if mv.is_capture() { 0 } else { 1 });
        let mut best_move = moves[0].clone();

        'outer: for d in 1..=self.search_depth {
            let mut iter_best_score = -MATE_SCORE;
            let mut iter_best_move = moves[0].clone();
            let mut alpha = -MATE_SCORE;

            for mv in &moves {
                if Instant::now() >= deadline {
                    break 'outer;
                }
                let new_pos = state.position.clone().play(mv).expect("legal move");
                let Some(score) = nn_negamax(
                    &new_pos,
                    &self.model,
                    d - 1,
                    -MATE_SCORE,
                    -alpha,
                    deadline,
                    &mut tt,
                ) else {
                    break 'outer;
                };
                let score = -score;
                if score > iter_best_score {
                    iter_best_score = score;
                    iter_best_move = mv.clone();
                }
                alpha = alpha.max(score);
            }

            best_move = iter_best_move;
        }

        best_move
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::GameState;

    #[test]
    fn nn_agent_random_makes_legal_move() {
        let state = GameState::new();
        let legal = state.legal_moves();
        let mut agent = NNAgent::new_random(1);
        let mv = agent.select_move(&state, Some(Duration::from_secs(5)));
        assert!(legal.contains(&mv));
    }

    #[test]
    fn nn_agent_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NNAgent>();
    }
}

use std::time::Duration;

use rayon::prelude::*;
use shakmaty::{Color, Position};

use crate::agents::Agent;
use crate::agents::nn_agent::NNAgent;
use crate::chess::{GameResult, GameState};
use crate::chess::rules::game_result;

use super::dataset::GameRecord;

/// Generate `n` self-play games using NNAgent with `depth` search ply.
/// Games are generated in parallel via rayon.
pub fn generate_games(n: usize, depth: u32) -> Vec<GameRecord> {
    (0..n)
        .into_par_iter()
        .map(|_| play_one_game(depth))
        .collect()
}

fn play_one_game(depth: u32) -> GameRecord {
    let mut white = NNAgent::new_random(depth);
    let mut black = NNAgent::new_random(depth);
    let mut state = GameState::new();
    let mut positions: Vec<String> = Vec::new();
    let time_budget = Some(Duration::from_millis(100));

    loop {
        positions.push(state.to_fen());

        if let Some(result) = game_result(&state.position, state.halfmove_clock()) {
            let outcome = match result {
                GameResult::WhiteWins => 1.0,
                GameResult::BlackWins => -1.0,
                GameResult::Draw => 0.0,
            };
            return GameRecord { positions, outcome };
        }

        if state.history.len() >= 200 {
            return GameRecord { positions, outcome: 0.0 };
        }

        let mv = if state.position.turn() == Color::White {
            white.select_move(&state, time_budget)
        } else {
            black.select_move(&state, time_budget)
        };

        state.apply_move(mv).expect("agent returned illegal move");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_one_game() {
        let records = generate_games(1, 1);
        assert_eq!(records.len(), 1);
        assert!(!records[0].positions.is_empty());
        assert!([-1.0f32, 0.0, 1.0].contains(&records[0].outcome));
    }

    #[test]
    fn game_terminates() {
        let record = play_one_game(1);
        assert!(!record.positions.is_empty());
        assert!(record.outcome.is_finite());
        assert!([-1.0f32, 0.0, 1.0].contains(&record.outcome));
    }
}

use std::time::Duration;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use shakmaty::{Color, Position};

use crate::agents::Agent;
use crate::agents::mcts::MCTSAgent;
use crate::agents::nn_agent::NNAgent;
use crate::chess::{GameResult, GameState};
use crate::chess::rules::game_result;

use super::dataset::{GameRecord, MctsGameRecord, MoveVisit};

pub struct SelfPlayStats {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub avg_game_len: f32,
}

fn make_progress_bar(n: usize, quiet: bool, label: &str) -> ProgressBar {
    if quiet {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(n as u64);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "{label} [{{bar:40}}] {{pos}}/{{len}}  {{elapsed_precise}}  ETA: {{eta}}"
        ))
        .unwrap()
        .progress_chars("=>-"),
    );
    pb
}

/// Generate `n` self-play games using NNAgent with `depth` search ply.
/// Games are generated in parallel via rayon.
pub fn generate_games(n: usize, depth: u32, quiet: bool) -> (Vec<GameRecord>, SelfPlayStats) {
    let pb = make_progress_bar(n, quiet, "Generating games  ");
    let records: Vec<GameRecord> = (0..n)
        .into_par_iter()
        .progress_with(pb)
        .map(|_| play_one_game(depth))
        .collect();
    let stats = compute_stats_value(&records);
    (records, stats)
}

fn compute_stats_value(records: &[GameRecord]) -> SelfPlayStats {
    let wins = records.iter().filter(|r| r.outcome > 0.5).count();
    let losses = records.iter().filter(|r| r.outcome < -0.5).count();
    let draws = records.iter().filter(|r| r.outcome.abs() < 0.5).count();
    let avg_game_len = if records.is_empty() {
        0.0
    } else {
        records.iter().map(|r| r.positions.len() as f32).sum::<f32>() / records.len() as f32
    };
    SelfPlayStats { wins, losses, draws, avg_game_len }
}

fn compute_stats_mcts(records: &[MctsGameRecord]) -> SelfPlayStats {
    let wins = records.iter().filter(|r| r.outcome > 0.5).count();
    let losses = records.iter().filter(|r| r.outcome < -0.5).count();
    let draws = records.iter().filter(|r| r.outcome.abs() < 0.5).count();
    let avg_game_len = if records.is_empty() {
        0.0
    } else {
        records.iter().map(|r| r.positions.len() as f32).sum::<f32>() / records.len() as f32
    };
    SelfPlayStats { wins, losses, draws, avg_game_len }
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

/// Generate `n` MCTS self-play games, each agent running at least `simulations` rollouts.
pub fn generate_mcts_games(
    n: usize,
    simulations: u32,
    quiet: bool,
) -> (Vec<MctsGameRecord>, SelfPlayStats) {
    let pb = make_progress_bar(n, quiet, "Generating MCTS games");
    let records: Vec<MctsGameRecord> = (0..n)
        .into_par_iter()
        .progress_with(pb)
        .map(|_| play_one_mcts_game(simulations))
        .collect();
    let stats = compute_stats_mcts(&records);
    (records, stats)
}

fn play_one_mcts_game(simulations: u32) -> MctsGameRecord {
    let mut white = MCTSAgent::new_random(simulations);
    let mut black = MCTSAgent::new_random(simulations);
    let mut state = GameState::new();
    let mut positions: Vec<String> = Vec::new();
    let mut move_visits_all: Vec<Vec<MoveVisit>> = Vec::new();
    let time_budget = Some(Duration::from_secs(2));

    loop {
        if let Some(result) = game_result(&state.position, state.halfmove_clock()) {
            let outcome = match result {
                GameResult::WhiteWins => 1.0,
                GameResult::BlackWins => -1.0,
                GameResult::Draw => 0.0,
            };
            return MctsGameRecord { positions, move_visits: move_visits_all, outcome };
        }

        if state.history.len() >= 200 {
            return MctsGameRecord { positions, move_visits: move_visits_all, outcome: 0.0 };
        }

        let current_fen = state.to_fen();
        let (mv, visits) = if state.position.turn() == Color::White {
            white.select_move_with_visits(&state, time_budget)
        } else {
            black.select_move_with_visits(&state, time_budget)
        };

        positions.push(current_fen);
        move_visits_all.push(visits);
        state.apply_move(mv).expect("agent returned illegal move");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_one_game() {
        let (records, _stats) = generate_games(1, 1, true);
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

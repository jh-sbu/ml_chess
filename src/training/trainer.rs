use std::path::Path;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;

use crate::chess::GameState;
use crate::chess::moves::parse_uci;
use crate::eval::neural::{ChessValueNetConfig, encode_position};
use crate::eval::policy::{ChessPolicyNetConfig, encode_move};

use super::checkpoint;
use super::dataset::{GameRecord, MctsGameRecord};

type TrainBackend = Autodiff<NdArray>;

pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Learning rate for the policy network. Defaults to `lr` when `None`.
    pub policy_lr: Option<f64>,
}

pub fn train(config: TrainingConfig, records: &[GameRecord], output_path: &Path) -> anyhow::Result<()> {
    // Flatten records into (encoded position, target outcome) pairs.
    let mut pairs: Vec<([f32; 832], f32)> = Vec::new();
    for record in records {
        for fen in &record.positions {
            match GameState::from_fen(fen) {
                Ok(state) => pairs.push((encode_position(&state), record.outcome)),
                Err(_) => continue,
            }
        }
    }

    if pairs.is_empty() {
        eprintln!("No training data; skipping training.");
        return Ok(());
    }

    let device = NdArrayDevice::default();
    let mut model = ChessValueNetConfig::new().init::<TrainBackend>(&device);
    let mut optimizer = AdamWConfig::new().init();

    for epoch in 0..config.epochs {
        let mut total_loss = 0.0f64;
        let mut batch_count = 0usize;

        for chunk in pairs.chunks(config.batch_size) {
            let batch_size = chunk.len();

            let input_data: Vec<f32> = chunk.iter().flat_map(|(enc, _)| enc.iter().copied()).collect();
            let target_data: Vec<f32> = chunk.iter().map(|(_, t)| *t).collect();

            let input = Tensor::<TrainBackend, 1>::from_floats(input_data.as_slice(), &device)
                .reshape([batch_size, 832]);
            let target = Tensor::<TrainBackend, 1>::from_floats(target_data.as_slice(), &device)
                .reshape([batch_size, 1]);

            let predicted = model.forward(input);
            let diff = predicted - target;
            let loss = diff.clone().mul(diff).mean();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.lr, model, grads);

            total_loss += 1.0; // count batches; scalar extraction requires detach on autodiff backend
            batch_count += 1;
        }

        eprintln!("Epoch {}/{}: {} batches processed.", epoch + 1, config.epochs, batch_count);
        let _ = total_loss;
    }

    let inference_model = model.valid();
    checkpoint::save_model(&inference_model, output_path)?;
    Ok(())
}

/// Train both value and policy networks from MCTS self-play data.
/// Policy targets are derived from visit count distributions.
pub fn train_policy(
    config: &TrainingConfig,
    records: &[MctsGameRecord],
    value_output: &Path,
    policy_output: &Path,
) -> anyhow::Result<()> {
    // Build (encoded_position, value_target, policy_target) triples.
    let mut triples: Vec<([f32; 832], f32, [f32; 4096])> = Vec::new();
    for record in records {
        for (i, fen) in record.positions.iter().enumerate() {
            let state = match GameState::from_fen(fen) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let Some(move_visits) = record.move_visits.get(i) else { continue };
            let total_visits: u32 = move_visits.iter().map(|v| v.visits).sum();
            if total_visits == 0 {
                continue;
            }
            let mut policy_target = [0.0f32; 4096];
            for mv_visit in move_visits {
                if let Ok(mv) = parse_uci(&state.position, &mv_visit.uci) {
                    policy_target[encode_move(&mv)] =
                        mv_visit.visits as f32 / total_visits as f32;
                }
            }
            triples.push((encode_position(&state), record.outcome, policy_target));
        }
    }

    if triples.is_empty() {
        eprintln!("No MCTS training data; skipping.");
        return Ok(());
    }

    let device = NdArrayDevice::default();
    let mut value_model = ChessValueNetConfig::new().init::<TrainBackend>(&device);
    let mut policy_model = ChessPolicyNetConfig::new().init::<TrainBackend>(&device);
    let mut value_optimizer = AdamWConfig::new().init();
    let mut policy_optimizer = AdamWConfig::new().init();
    let policy_lr = config.policy_lr.unwrap_or(config.lr);

    for epoch in 0..config.epochs {
        let mut batch_count = 0usize;
        for chunk in triples.chunks(config.batch_size) {
            let batch_size = chunk.len();
            let input_data: Vec<f32> =
                chunk.iter().flat_map(|(enc, _, _)| enc.iter().copied()).collect();
            let value_target_data: Vec<f32> =
                chunk.iter().map(|(_, v, _)| *v).collect();
            let policy_target_data: Vec<f32> =
                chunk.iter().flat_map(|(_, _, p)| p.iter().copied()).collect();

            let input = Tensor::<TrainBackend, 1>::from_floats(input_data.as_slice(), &device)
                .reshape([batch_size, 832]);
            let value_target =
                Tensor::<TrainBackend, 1>::from_floats(value_target_data.as_slice(), &device)
                    .reshape([batch_size, 1]);
            let policy_target =
                Tensor::<TrainBackend, 1>::from_floats(policy_target_data.as_slice(), &device)
                    .reshape([batch_size, 4096]);

            // Value loss: MSE
            let value_pred = value_model.forward(input.clone());
            let diff = value_pred - value_target;
            let value_loss = diff.clone().mul(diff).mean();
            let grads = value_loss.backward();
            let grads = GradientsParams::from_grads(grads, &value_model);
            value_model = value_optimizer.step(config.lr, value_model, grads);

            // Policy loss: cross-entropy via numerically-stable log-sum-exp
            // CE = -dot(target, logits) + logsumexp(logits)
            let logits = policy_model.forward(input);
            let max_logit = logits.clone().max_dim(1); // [batch, 1]
            let shifted = logits.clone() - max_logit.clone();
            let log_sum_exp = shifted.exp().sum_dim(1).log() + max_logit; // [batch, 1]
            let neg_dot = -(policy_target * logits).sum_dim(1); // [batch, 1]
            let policy_loss = (neg_dot + log_sum_exp).mean();
            let grads = policy_loss.backward();
            let grads = GradientsParams::from_grads(grads, &policy_model);
            policy_model = policy_optimizer.step(policy_lr, policy_model, grads);

            batch_count += 1;
        }
        eprintln!(
            "MCTS Epoch {}/{}: {} batches processed.",
            epoch + 1,
            config.epochs,
            batch_count
        );
    }

    let value_inference = value_model.valid();
    let policy_inference = policy_model.valid();
    checkpoint::save_model(&value_inference, value_output)?;
    checkpoint::save_policy_model(&policy_inference, policy_output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::dataset::GameRecord;

    #[test]
    fn train_one_epoch() {
        let starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let records = vec![
            GameRecord { positions: vec![starting_fen.to_string()], outcome: 1.0 },
            GameRecord { positions: vec![starting_fen.to_string()], outcome: 0.0 },
            GameRecord { positions: vec![starting_fen.to_string()], outcome: -1.0 },
        ];

        let dir = std::env::temp_dir().join("ml_chess_trainer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("trained_model");

        let config = TrainingConfig { epochs: 1, batch_size: 2, lr: 1e-3, policy_lr: None };
        train(config, &records, &output).expect("training should not panic");

        std::fs::remove_file(dir.join("trained_model.mpk")).ok();
    }
}

use std::path::Path;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;

use crate::chess::GameState;
use crate::eval::neural::{ChessValueNetConfig, encode_position};

use super::checkpoint;
use super::dataset::GameRecord;

type TrainBackend = Autodiff<NdArray>;

pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::dataset::GameRecord;

    #[test]
    fn train_one_epoch() {
        let starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let records = vec![
            GameRecord {
                positions: vec![starting_fen.to_string()],
                outcome: 1.0,
            },
            GameRecord {
                positions: vec![starting_fen.to_string()],
                outcome: 0.0,
            },
            GameRecord {
                positions: vec![starting_fen.to_string()],
                outcome: -1.0,
            },
        ];

        let dir = std::env::temp_dir().join("ml_chess_trainer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("trained_model");

        let config = TrainingConfig {
            epochs: 1,
            batch_size: 2,
            lr: 1e-3,
        };

        train(config, &records, &output).expect("training should not panic");

        // Clean up
        std::fs::remove_file(dir.join("trained_model.mpk")).ok();
    }
}

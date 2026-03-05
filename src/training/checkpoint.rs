#![allow(dead_code)]

use std::path::Path;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};

use crate::eval::neural::{ChessValueNet, ChessValueNetConfig};

/// Save an NdArray-backend model to disk. The recorder appends `.mpk` to the path.
pub fn save_model(model: &ChessValueNet<NdArray>, path: &Path) -> anyhow::Result<()> {
    CompactRecorder::new()
        .record(model.clone().into_record(), path.to_path_buf())
        .map_err(|e| anyhow::anyhow!("Failed to save model to {}: {e}", path.display()))?;
    Ok(())
}

/// Load a model from a checkpoint. Path should be given without extension.
pub fn load_model(path: &Path) -> anyhow::Result<ChessValueNet<NdArray>> {
    let device = NdArrayDevice::default();
    let base = ChessValueNetConfig::new().init::<NdArray>(&device);
    let record = CompactRecorder::new()
        .load(path.to_path_buf(), &device)
        .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {e}", path.display()))?;
    Ok(base.load_record(record))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::GameState;
    use crate::eval::neural::evaluate_nn;
    use std::fs;

    #[test]
    fn save_and_load_model() {
        let dir = std::env::temp_dir().join("ml_chess_checkpoint_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_model");

        let device = NdArrayDevice::default();
        let model = ChessValueNetConfig::new().init::<NdArray>(&device);

        save_model(&model, &path).expect("save failed");
        let loaded = load_model(&path).expect("load failed");

        let state = GameState::new();
        let score = evaluate_nn(&state, &loaded);
        assert!(score.abs() <= 1000, "score {score} out of expected range");

        // Clean up
        fs::remove_file(dir.join("test_model.mpk")).ok();
    }
}

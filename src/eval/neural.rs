#![allow(dead_code)]

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{
        Tensor,
        activation::{relu, tanh},
        backend::Backend,
    },
};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use shakmaty::{Color, Position, Role};

use crate::chess::GameState;
use super::Score;

/// Encode a chess position as a flat float array of 832 elements:
/// 12 planes (one per piece type per color) + 1 side-to-move plane, each 64 squares.
pub fn encode_position(state: &GameState) -> [f32; 832] {
    let mut planes = [0.0f32; 832];
    for (sq, piece) in state.position.board().iter() {
        let color_offset = match piece.color {
            Color::White => 0,
            Color::Black => 6,
        };
        let role_offset = match piece.role {
            Role::Pawn => 0,
            Role::Knight => 1,
            Role::Bishop => 2,
            Role::Rook => 3,
            Role::Queen => 4,
            Role::King => 5,
        };
        let plane_idx = color_offset + role_offset;
        let sq_idx = usize::from(sq);
        planes[plane_idx * 64 + sq_idx] = 1.0;
    }
    // Plane 12: side to move (all 1.0 if White to move)
    if state.position.turn() == Color::White {
        for i in 0..64 {
            planes[12 * 64 + i] = 1.0;
        }
    }
    planes
}

/// Configuration for the chess value network.
#[derive(Config, Debug)]
pub struct ChessValueNetConfig {
    #[config(default = "256")]
    hidden1: usize,
    #[config(default = "128")]
    hidden2: usize,
    #[config(default = "64")]
    hidden3: usize,
}

/// Four-layer MLP: 832 → 256 → 128 → 64 → 1, output in (-1, +1) via tanh.
#[derive(Module, Debug)]
pub struct ChessValueNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    fc4: Linear<B>,
}

impl ChessValueNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ChessValueNet<B> {
        ChessValueNet {
            fc1: LinearConfig::new(832, self.hidden1).init(device),
            fc2: LinearConfig::new(self.hidden1, self.hidden2).init(device),
            fc3: LinearConfig::new(self.hidden2, self.hidden3).init(device),
            fc4: LinearConfig::new(self.hidden3, 1).init(device),
        }
    }
}

impl<B: Backend> ChessValueNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.fc1.forward(x));
        let x = relu(self.fc2.forward(x));
        let x = relu(self.fc3.forward(x));
        tanh(self.fc4.forward(x))
    }
}

/// Evaluate a position using the value network. Returns a score in [-1000, 1000].
pub fn evaluate_nn(state: &GameState, model: &ChessValueNet<NdArray>) -> Score {
    let device = NdArrayDevice::default();
    let encoded = encode_position(state);
    let tensor = Tensor::<NdArray, 1>::from_floats(encoded.as_slice(), &device)
        .reshape([1, 832]);
    let output = model.forward(tensor);
    let val: f32 = output.flatten::<1>(0, 1).into_scalar();
    (val * 1000.0) as Score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::GameState;

    #[test]
    fn encode_starting_position_has_32_pieces() {
        let state = GameState::new();
        let encoded = encode_position(&state);
        let piece_sum: f32 = encoded[..12 * 64].iter().sum();
        assert_eq!(piece_sum as u32, 32);
    }

    #[test]
    fn encode_side_to_move_plane() {
        let state = GameState::new();
        let encoded = encode_position(&state);
        let side_plane_sum: f32 = encoded[12 * 64..].iter().sum();
        assert_eq!(side_plane_sum as u32, 64);
    }

    #[test]
    fn inference_returns_score() {
        let device = NdArrayDevice::default();
        let model = ChessValueNetConfig::new().init::<NdArray>(&device);
        let state = GameState::new();
        let score = evaluate_nn(&state, &model);
        assert!(score.abs() <= 1000);
    }
}

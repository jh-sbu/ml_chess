#![allow(dead_code)]

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{
        Tensor,
        activation::relu,
        backend::Backend,
    },
};

use crate::chess::{GameState, Move};
use super::neural::encode_position;

/// Encode a move as a flat index in [0, 4096): `from_sq * 64 + to_sq`.
/// Underpromotions collapse to the same index as queen promotion because
/// from/to squares are identical.
pub fn encode_move(mv: &Move) -> usize {
    let from = usize::from(mv.from().expect("chess move has from square"));
    let to = usize::from(mv.to());
    from * 64 + to
}

/// Configuration for the chess policy network.
#[derive(Config, Debug)]
pub struct ChessPolicyNetConfig {
    #[config(default = "256")]
    hidden1: usize,
    #[config(default = "128")]
    hidden2: usize,
}

/// Three-layer MLP: 832 → 256 → 128 → 4096 (raw logits).
#[derive(Module, Debug)]
pub struct ChessPolicyNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl ChessPolicyNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ChessPolicyNet<B> {
        ChessPolicyNet {
            fc1: LinearConfig::new(832, self.hidden1).init(device),
            fc2: LinearConfig::new(self.hidden1, self.hidden2).init(device),
            fc3: LinearConfig::new(self.hidden2, 4096).init(device),
        }
    }
}

impl<B: Backend> ChessPolicyNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.fc1.forward(x));
        let x = relu(self.fc2.forward(x));
        self.fc3.forward(x) // raw logits, shape [batch, 4096]
    }
}

/// Run the policy network on a position. Returns `(move, prior)` pairs for all
/// legal moves, sorted descending by probability (numerically-stable softmax over
/// legal-move logits only).
pub fn policy_net_forward(
    state: &GameState,
    model: &ChessPolicyNet<NdArray>,
) -> Vec<(Move, f32)> {
    let legal = state.legal_moves();
    if legal.is_empty() {
        return vec![];
    }
    let device = NdArrayDevice::default();
    let encoded = encode_position(state);
    let tensor =
        Tensor::<NdArray, 1>::from_floats(encoded.as_slice(), &device).reshape([1, 832]);
    let logits_tensor = model.forward(tensor); // [1, 4096]
    let logits_vec: Vec<f32> = logits_tensor
        .flatten::<1>(0, 1)
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_else(|_| vec![0.0f32; 4096]);

    // Gather logits for legal moves and compute numerically-stable softmax.
    let raw: Vec<f32> = legal.iter().map(|mv| logits_vec[encode_move(mv)]).collect();
    let max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = raw.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum::<f32>().max(f32::EPSILON);
    let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    let mut pairs: Vec<(Move, f32)> = legal.into_iter().zip(probs).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;

    #[test]
    fn encode_move_is_in_range() {
        let state = GameState::new();
        for mv in state.legal_moves() {
            let idx = encode_move(&mv);
            assert!(idx < 4096, "encode_move out of range: {idx}");
        }
    }

    #[test]
    fn policy_forward_priors_sum_to_one() {
        let device = NdArrayDevice::default();
        let model = ChessPolicyNetConfig::new().init::<NdArray>(&device);
        let state = GameState::new();
        let priors = policy_net_forward(&state, &model);
        assert!(!priors.is_empty());
        let total: f32 = priors.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5, "priors sum = {total}");
    }

    #[test]
    fn policy_net_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ChessPolicyNet<NdArray>>();
    }
}

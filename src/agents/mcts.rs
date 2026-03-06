#![allow(dead_code)]

use std::path::Path;
use std::time::{Duration, Instant};

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use serde::{Deserialize, Serialize};
use shakmaty::{Chess, Color, Position};

use crate::agents::Agent;
use crate::chess::{GameResult, GameState, Move};
use crate::chess::moves::move_to_uci;
use crate::chess::rules::game_result;
use crate::eval::neural::{ChessValueNet, ChessValueNetConfig, evaluate_nn};
use crate::eval::policy::{ChessPolicyNet, ChessPolicyNetConfig, policy_net_forward};

/// Visit count for a single move, used to build policy training targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveVisit {
    pub uci: String,
    pub visits: u32,
}

// ── Arena-based MCTS tree ────────────────────────────────────────────────────

struct MctsNode {
    parent: usize,
    incoming_move: Option<Move>,
    /// Chess position AT this node (side to move is the player about to move here).
    position: Chess,
    n: u32,
    /// Accumulated value from the perspective of the player who MOVED INTO this node.
    w: f32,
    /// Prior probability assigned by parent's policy network.
    p: f32,
    children: Vec<(Move, usize)>,
    expanded: bool,
    terminal: bool,
    terminal_value: f32,
}

struct MctsTree {
    nodes: Vec<MctsNode>,
    c_puct: f32,
}

impl MctsTree {
    fn new(root_pos: Chess, c_puct: f32) -> Self {
        let root = MctsNode {
            parent: 0,
            incoming_move: None,
            position: root_pos,
            n: 0,
            w: 0.0,
            p: 1.0,
            children: vec![],
            expanded: false,
            terminal: false,
            terminal_value: 0.0,
        };
        MctsTree { nodes: vec![root], c_puct }
    }

    fn run_simulation(
        &mut self,
        value_model: &ChessValueNet<NdArray>,
        policy_model: &ChessPolicyNet<NdArray>,
    ) {
        let (path, leaf_pos) = self.select();
        let leaf_idx = *path.last().unwrap();

        let value = if self.nodes[leaf_idx].terminal {
            self.nodes[leaf_idx].terminal_value
        } else if let Some(result) = game_result(&leaf_pos, leaf_pos.halfmoves()) {
            // Terminal: annotate and use exact outcome.
            let v = terminal_value(result, leaf_pos.turn());
            self.nodes[leaf_idx].terminal = true;
            self.nodes[leaf_idx].terminal_value = v;
            v
        } else {
            let leaf_state = GameState::from_pos(leaf_pos.clone());
            let priors = policy_net_forward(&leaf_state, policy_model);
            if priors.is_empty() {
                // Shouldn't happen if game_result is None, but guard anyway.
                self.nodes[leaf_idx].terminal = true;
                0.0
            } else {
                self.expand(leaf_idx, &leaf_pos, priors);
                // Value from current player's perspective.
                let abs_val = evaluate_nn(&leaf_state, value_model) as f32 / 1000.0;
                if leaf_pos.turn() == Color::White { abs_val } else { -abs_val }
            }
        };

        self.backpropagate(&path, value);
    }

    /// Traverse the tree via PUCT until reaching an unexpanded or terminal node.
    /// Returns the path (root…leaf indices) and a clone of the leaf's position.
    fn select(&self) -> (Vec<usize>, Chess) {
        let mut path = vec![0usize];
        loop {
            let node = &self.nodes[*path.last().unwrap()];
            if !node.expanded || node.terminal || node.children.is_empty() {
                break;
            }
            let (_, child_idx) = self.select_child(*path.last().unwrap());
            path.push(child_idx);
        }
        let leaf_pos = self.nodes[*path.last().unwrap()].position.clone();
        (path, leaf_pos)
    }

    fn select_child(&self, node_idx: usize) -> (Move, usize) {
        let node = &self.nodes[node_idx];
        let n_sqrt = (node.n as f32).sqrt();
        node.children
            .iter()
            .map(|(mv, child_idx)| {
                let child = &self.nodes[*child_idx];
                let q = if child.n > 0 { child.w / child.n as f32 } else { 0.0 };
                let u = self.c_puct * child.p * n_sqrt / (1.0 + child.n as f32);
                (mv.clone(), *child_idx, q + u)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(mv, idx, _)| (mv, idx))
            .unwrap()
    }

    fn expand(&mut self, node_idx: usize, parent_pos: &Chess, priors: Vec<(Move, f32)>) {
        let children: Vec<(Move, usize)> = priors
            .into_iter()
            .map(|(mv, p)| {
                let child_pos = parent_pos.clone().play(&mv).expect("legal MCTS expand move");
                let child_idx = self.nodes.len();
                self.nodes.push(MctsNode {
                    parent: node_idx,
                    incoming_move: Some(mv.clone()),
                    position: child_pos,
                    n: 0,
                    w: 0.0,
                    p,
                    children: vec![],
                    expanded: false,
                    terminal: false,
                    terminal_value: 0.0,
                });
                (mv, child_idx)
            })
            .collect();
        self.nodes[node_idx].children = children;
        self.nodes[node_idx].expanded = true;
    }

    /// Backpropagate `value` (from the leaf's current-player perspective) up to root,
    /// alternating sign at each step so each node stores value from its mover's perspective.
    fn backpropagate(&mut self, path: &[usize], value: f32) {
        let mut v = value;
        for &idx in path.iter().rev() {
            self.nodes[idx].n += 1;
            self.nodes[idx].w += v;
            v = -v;
        }
    }

    fn best_move(&self) -> Option<Move> {
        self.nodes[0]
            .children
            .iter()
            .max_by_key(|(_, child_idx)| self.nodes[*child_idx].n)
            .map(|(mv, _)| mv.clone())
    }

    fn visit_counts(&self) -> Vec<MoveVisit> {
        self.nodes[0]
            .children
            .iter()
            .map(|(mv, child_idx)| MoveVisit {
                uci: move_to_uci(mv),
                visits: self.nodes[*child_idx].n,
            })
            .collect()
    }
}

fn terminal_value(result: GameResult, turn: Color) -> f32 {
    match result {
        GameResult::WhiteWins => {
            // White won; current player (turn) is Black (checkmated).
            if turn == Color::White { 1.0 } else { -1.0 }
        }
        GameResult::BlackWins => {
            if turn == Color::Black { 1.0 } else { -1.0 }
        }
        GameResult::Draw => 0.0,
    }
}

// ── MCTSAgent ────────────────────────────────────────────────────────────────

pub struct MCTSAgent {
    value_model: ChessValueNet<NdArray>,
    policy_model: ChessPolicyNet<NdArray>,
    c_puct: f32,
    min_simulations: u32,
    name: String,
}

impl MCTSAgent {
    /// Create an agent with freshly initialized (random) weights.
    pub fn new_random(min_simulations: u32) -> Self {
        let device = NdArrayDevice::default();
        let value_model = ChessValueNetConfig::new().init::<NdArray>(&device);
        let policy_model = ChessPolicyNetConfig::new().init::<NdArray>(&device);
        Self {
            value_model,
            policy_model,
            c_puct: 1.0,
            min_simulations,
            name: format!("MCTSAgent(s{min_simulations})"),
        }
    }

    /// Load checkpointed value and policy models. Paths are given without extension.
    pub fn load(
        value_path: &Path,
        policy_path: &Path,
        min_simulations: u32,
    ) -> anyhow::Result<Self> {
        let device = NdArrayDevice::default();

        let value_base = ChessValueNetConfig::new().init::<NdArray>(&device);
        let value_record = CompactRecorder::new()
            .load(value_path.to_path_buf(), &device)
            .map_err(|e| anyhow::anyhow!("Failed to load value model: {e}"))?;
        let value_model = value_base.load_record(value_record);

        let policy_base = ChessPolicyNetConfig::new().init::<NdArray>(&device);
        let policy_record = CompactRecorder::new()
            .load(policy_path.to_path_buf(), &device)
            .map_err(|e| anyhow::anyhow!("Failed to load policy model: {e}"))?;
        let policy_model = policy_base.load_record(policy_record);

        Ok(Self {
            value_model,
            policy_model,
            c_puct: 1.0,
            min_simulations,
            name: format!("MCTSAgent(s{min_simulations})"),
        })
    }

    /// Run MCTS and return the chosen move together with per-move visit counts.
    /// The visit counts are used as policy training targets.
    pub fn select_move_with_visits(
        &mut self,
        state: &GameState,
        time_budget: Option<Duration>,
    ) -> (Move, Vec<MoveVisit>) {
        let deadline = Instant::now() + time_budget.unwrap_or(Duration::from_secs(5));
        let mut tree = MctsTree::new(state.position.clone(), self.c_puct);

        let mut sim_count = 0u32;
        loop {
            tree.run_simulation(&self.value_model, &self.policy_model);
            sim_count += 1;
            if sim_count >= self.min_simulations && Instant::now() >= deadline {
                break;
            }
        }

        let visits = tree.visit_counts();
        let best = tree
            .best_move()
            .unwrap_or_else(|| state.legal_moves().into_iter().next().unwrap());
        (best, visits)
    }
}

impl Agent for MCTSAgent {
    fn select_move(&mut self, state: &GameState, time_budget: Option<Duration>) -> Move {
        self.select_move_with_visits(state, time_budget).0
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
    fn mcts_agent_random_makes_legal_move() {
        let state = GameState::new();
        let legal = state.legal_moves();
        let mut agent = MCTSAgent::new_random(10);
        let mv = agent.select_move(&state, Some(Duration::from_secs(5)));
        assert!(legal.contains(&mv));
    }

    #[test]
    fn mcts_agent_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MCTSAgent>();
    }

    #[test]
    fn mcts_runs_min_simulations() {
        let state = GameState::new();
        let mut agent = MCTSAgent::new_random(10);
        let (_, visits) = agent.select_move_with_visits(&state, Some(Duration::from_secs(10)));
        let total_visits: u32 = visits.iter().map(|v| v.visits).sum();
        assert!(total_visits >= 10, "Expected at least 10 simulations, got {total_visits}");
    }
}

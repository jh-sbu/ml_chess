# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goals

A terminal-based chess application written in Rust with the following pillars:

1. **Playable TUI chess game** — human-vs-human and human-vs-AI via Ratatui
2. **Traditional AI agents** — search-based opponents (minimax, negamax, alpha-beta pruning, iterative deepening, etc.)
3. **ML/DL agents** — neural network opponents trained with the `burn` deep learning library
4. **Training infrastructure** — self-play pipelines, dataset generation, model checkpointing
5. **AI spectator mode** — watch two AI agents play each other in the TUI

---

## Build & Run

```bash
# Debug build
cargo build

# Run — no subcommand opens the TUI main menu
cargo run
cargo run -- play [--depth N] [--model PATH]   # Human vs AI
cargo run -- watch [--depth N] [--model PATH]  # AI vs AI spectator mode
cargo run -- train [--games N] [--epochs N] [--output PATH]

# Training loop requires the feature flag
cargo run --features train -- train --games 100 --epochs 10 --output chess_model

# Tests
cargo test

# Single test
cargo test <test_name>

# Clippy (must pass before committing)
cargo clippy -- -D warnings

# Format
cargo fmt
```

---

## Architecture Overview

All modules are implemented. Module dependency direction: `training` → `agents` → `chess`; `tui` → `chess` + `agents`.

```
src/
├── main.rs              # clap CLI dispatch (play/watch/train subcommands)
├── app.rs               # App state + ratatui event loop; AI runs on background thread
├── chess/               # shakmaty wrapper — GameState, Move, rules, notation
├── tui/                 # Ratatui widgets: board, game UI, main menu
├── agents/              # Agent trait + Random, Human, Minimax, Negamax, NNAgent
├── eval/                # Classical evaluator (material + PST) + neural.rs (burn MLP)
└── training/            # dataset.rs, self_play.rs, checkpoint.rs, trainer.rs (feature-gated)
```

### Agent Trait

```rust
pub trait Agent {
    fn select_move(&mut self, state: &GameState, time_budget: Option<Duration>) -> Move;
    fn name(&self) -> &str;
}
```

Agents are interchangeable in both game play and training. `NNAgent` is the default AI (negamax + NN eval); falls back to random weights if no model file is provided.

### Feature Flags

- `train` — enables `burn/train`, `burn/autodiff`; required for `training::trainer`. Default build uses inference-only `burn/ndarray`.

---

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `ratatui 0.30` | TUI rendering |
| `burn 0.20.1` | Deep learning (inference: `ndarray` feature; training: `train` feature) |
| `shakmaty 0.27` | Chess rules and move generation |
| `clap 4` | CLI with derive macros |
| `rayon` | Parallel self-play game generation |
| `rand 0.9` | Stochastic move selection |
| `serde_json` | Game record serialization |

---

## Development Conventions

- **Edition 2024** — prefer `let-else`, `?` chains, and `impl Trait` in function signatures.
- **No `unwrap()`/`expect()` in library code** — use `thiserror` / `anyhow`; `unwrap` is acceptable only in tests.
- **No unsafe** unless critically necessary; document every `unsafe` block.
- **Commits** — atomic, single-concern commits with present-tense imperative messages.
- All chess files use `#![allow(dead_code)]` to suppress "library in progress" warnings.

---

## Critical shakmaty 0.27 API Notes

- Use `UciMove` (not deprecated `Uci`); errors are `ParseUciMoveError` / `IllegalUciMoveError`.
- FEN: `Fen::from_position(pos, EnPassantMode::Legal)` / `Epd::from_position(pos, EnPassantMode::Legal)`.
- SAN: `SanPlus::from_move(pos.clone(), &mv)` (consumes pos — clone first); `from_move_and_play_unchecked(&mut pos, &mv)` advances in-place.
- `pos.play(&mv)` returns `Result<Self, PlayError<Self>>` — use `.clone().play(&mv)` to avoid `mem::replace`.
- `pos.fullmoves()` returns `NonZeroU32`; call `.get()` for `u32`.
- `pos.is_insufficient_material()` checks both colors (provided method on `Position` trait).
- Iterate squares: `for rank in Rank::ALL { for file in File::ALL { let sq = Square::from_coords(file, rank); ... } }`.
- Index: `usize::from(sq.file())` / `usize::from(sq.rank())`; Black PST uses `sq.flip_vertical()`.
- `use shakmaty::Color as ChessColor` to avoid clash with `ratatui::style::Color`.

## Critical burn 0.20.1 API Notes

- `AdamWConfig::new().init()` returns `OptimizerAdaptor<AdamW,_,_>` — do not annotate the type explicitly.
- MSE loss: `diff.clone().mul(diff).mean()` (no built-in `mse_loss` in this version).
- `model.valid()` converts from autodiff backend to inference mode for export.
- `CompactRecorder` used for checkpoint save/load.

## Critical ratatui 0.28+ API Notes

- `buf.cell_mut(Position::new(x, y))` (not deprecated `get_mut`).
- Call `frame.area()` before `frame.buffer_mut()` to avoid borrow conflict.
- `MainMenu` needs `#[derive(Clone)]` for `Widget::render(self, ...)`.
- `rand 0.9` API: `rand::rng()` + `IndexedRandom::choose`.

---

## ML Agent Notes

- Input: 13 planes × 64 squares (12 piece bitplanes + side-to-move), encoded by `encode_position` in `eval/neural.rs`.
- Architecture: MLP 832→256→128→64→1, tanh output (value in [-1, 1]).
- Training signal: outcome labels from self-play (win=1, loss=-1, draw=0).
- Self-play games capped at 200 moves to prevent infinite games.
- Save model configs alongside weights for reproducibility (`checkpoint.rs` wraps `CompactRecorder`).

---

## Roadmap

- [x] Chess engine with legal move generation and rule enforcement
- [x] Basic TUI board rendering and human input
- [x] Random and minimax agents (classical eval)
- [x] Alpha-beta with iterative deepening and move ordering
- [x] CLI subcommands (`play`, `watch`, `train`)
- [x] Value network agent (burn)
- [x] Self-play training pipeline
- [ ] Policy network / MCTS hybrid
- [ ] Opening book support
- [ ] ELO tracking across agent versions

A planning document is available in `plans/overview.md`.

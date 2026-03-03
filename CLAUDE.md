# ml_chess — Claude Code Project Guide

## Project Goals

A terminal-based chess application written in Rust with the following pillars:

1. **Playable TUI chess game** — human-vs-human and human-vs-AI via Ratatui
2. **Traditional AI agents** — search-based opponents (minimax, negamax, alpha-beta pruning, iterative deepening, etc.)
3. **ML/DL agents** — neural network opponents trained with the `burn` deep learning library (value networks, policy networks, MCTS + NN, etc.)
4. **Training infrastructure** — self-play pipelines, dataset generation, model checkpointing, and evaluation harnesses
5. **AI spectator mode** — watch two AI agents play each other in the TUI

---

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `ratatui` | TUI rendering and layout |
| `burn` | Deep learning framework for ML agents |

Additional crates to add as needed:
- `crossterm` — terminal backend for ratatui (already a ratatui dependency)
- `shakmaty` or `chess` — battle-tested move generation / rule enforcement (consider before rolling your own)
- `serde` / `serde_json` — serializing game records, model configs, training datasets
- `clap` — CLI argument parsing (subcommands: `play`, `train`, `watch`)
- `rayon` — parallelism for search and self-play data generation
- `rand` / `rand_distr` — stochastic rollouts, noise injection for training

---

## Architecture Overview

### Planned Module Structure

```
src/
├── main.rs              # Entry point; CLI dispatch
├── app.rs               # Top-level application state + event loop
├── chess/
│   ├── mod.rs
│   ├── board.rs         # Board representation (bitboards or mailbox)
│   ├── moves.rs         # Move generation and validation
│   ├── rules.rs         # Check, checkmate, stalemate, draw conditions
│   └── notation.rs      # FEN / PGN parsing and serialization
├── tui/
│   ├── mod.rs
│   ├── board_widget.rs  # Ratatui widget for the chess board
│   ├── game_ui.rs       # Full game screen layout
│   └── menu.rs          # Main menu and settings screens
├── agents/
│   ├── mod.rs           # Agent trait definition
│   ├── human.rs         # Passes input from TUI to move selection
│   ├── random.rs        # Random legal move agent (baseline)
│   ├── minimax.rs       # Minimax with alpha-beta pruning
│   ├── negamax.rs       # Negamax variant
│   └── nn_agent.rs      # Burn-powered neural network agent
├── eval/
│   ├── mod.rs
│   ├── classical.rs     # Hand-crafted evaluation (material, PST, mobility)
│   └── neural.rs        # Neural network inference wrapper
└── training/
    ├── mod.rs
    ├── self_play.rs     # Self-play game generation
    ├── dataset.rs       # Position/label dataset construction
    ├── trainer.rs       # Burn training loop
    └── checkpoint.rs    # Model saving and loading
```

### Agent Trait

All agents should implement a common trait so they are interchangeable in both game play and training:

```rust
pub trait Agent {
    fn select_move(&mut self, board: &Board, time_budget: Duration) -> Move;
    fn name(&self) -> &str;
}
```

---

## Build & Run

```bash
# Debug build
cargo build

# Run (once CLI is wired up)
cargo run -- play          # Human vs AI
cargo run -- watch         # AI vs AI spectator mode
cargo run -- train         # Start a training run

# Tests
cargo test

# Clippy (always fix before committing)
cargo clippy -- -D warnings

# Format
cargo fmt
```

---

## Development Conventions

- **Edition 2024** — use modern Rust idioms; prefer `let-else`, `?` chains, and `impl Trait` in function signatures.
- **No `unwrap()`/`expect()` in library code** — propagate errors with `thiserror` or `anyhow`; `unwrap` is acceptable only in tests and one-off scripts.
- **No unsafe** unless absolutely required for performance-critical board representation; document every `unsafe` block.
- **Keep the chess engine, TUI, agents, and training infrastructure as separate concerns** — cross-cutting dependencies should flow one way: `training` → `agents` → `chess`; `tui` → `chess` + `agents`.
- **Avoid premature abstraction** — don't generalize until there are at least two concrete use cases.
- **Tests** — unit-test move generation with known positions (perft counts), and agent correctness with forced-mate puzzles.
- **Commits** — atomic, single-concern commits with present-tense imperative messages.

---

## ML Agent Notes

- Start with a simple value network (board position → scalar evaluation) as a drop-in replacement for the classical evaluator.
- Use `burn`'s `autodiff` backend for training and a fast inference backend (e.g., `burn-ndarray` or `burn-wgpu`) for play.
- Input representation: 12 bitplane encoding (one per piece type per color) + auxiliary planes (castling rights, en passant, side to move).
- Training signal: outcome labels from self-play games (win/loss/draw) or from tablebases for endgame positions.
- Save model configs alongside weights so experiments are reproducible.

---

## Roadmap (High Level)


- [ ] Chess engine with legal move generation and rule enforcement
- [ ] Basic TUI board rendering and human input
- [ ] Random and minimax agents (classical eval)
- [ ] Alpha-beta with iterative deepening and move ordering
- [ ] CLI subcommands (`play`, `watch`, `train`)
- [ ] Value network agent (burn)
- [ ] Self-play training pipeline
- [ ] Policy network / MCTS hybrid
- [ ] Opening book support
- [ ] ELO tracking across agent versions

A planning document for this project has been generated and is available in plans/overview.md
